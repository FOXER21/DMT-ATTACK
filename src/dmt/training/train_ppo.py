#!/usr/bin/env python3
"""Train the generator bottleneck with a clipped PPO actor-critic."""
import sys,os,json,time,argparse,pickle
import numpy as np

LATENT_DIM=4*64*64;SQRT_D=np.sqrt(LATENT_DIM)
from dmt.paths import configure_runtime, resolve, scene_root, sd_model_dir  # noqa: E402

configure_runtime()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-records", default="data/teacher/records.jsonl")
    ap.add_argument("--checkpoint", default="checkpoints/generator.pt")
    ap.add_argument("--baselines", default="data/baselines/clean_rollouts.pkl")
    ap.add_argument("--episodes", type=int, default=3); ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--prompts-per-iter", type=int, default=3); ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--iters", type=int, default=50); ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01); ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--output-dir", default="outputs/online_ppo"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    import torch,torch.nn as nn,torch.nn.functional as F
    import dmt.diffusion.sampler as sd
    from dmt.rollout.surrogate_data import Pi05Evaluator
    from dmt.scene.tablecloth import replace_tablecloth
    from dmt.training.train_generator import build_generator_model, token_from_prompt

    # ---- Load data ----
    ckpt = torch.load(resolve(args.checkpoint), map_location=args.device, weights_only=False)
    model = build_generator_model(1024, ckpt["heads"], ckpt["bottleneck"], ckpt["hidden"], 1024).to(args.device)
    model.load_state_dict(ckpt["model"])
    # Add learnable log_std per head for PPO
    model.log_std = nn.ParameterList([nn.Parameter(torch.tensor(-2.0)) for _ in range(ckpt["heads"])])
    model.log_std.to(args.device)
    print(f"Actor: {sum(p.numel()/1e6 for p in model.parameters()):.1f}M + {len(model.log_std)} log_std")

    # Critic: V(prompt_embedding) → scalar
    class Critic(nn.Module):
        def __init__(self, in_dim=1024):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
        def forward(self,x): return self.net(x).squeeze(-1)
    critic = Critic(1024).to(args.device)
    print(f"Critic: {sum(p.numel()/1e6 for p in critic.parameters()):.1f}M")

    # Load baselines + teachers
    records_path = resolve(args.teacher_records)
    records = [json.loads(l) for l in records_path.read_text().strip().split('\n') if l.strip()]
    rec_by_id = {r["teacher_id"]: r for r in records}
    bl_path = resolve(args.baselines)
    clean_rollouts = pickle.load(open(bl_path,"rb")) if bl_path.exists() else {}
    print(f"Baselines: {len(clean_rollouts)}")

    emb_data = np.load(resolve("data/teacher/sd_embeddings.npz"), allow_pickle=True)
    teacher_ids = list(emb_data["teacher_ids"])
    tid_to_pkey = {r.get('teacher_id',''): r.get('prompt_key','?') for r in records}

    train_prompts = []
    for i,tid in enumerate(teacher_ids):
        r=rec_by_id.get(tid,{}); gap=(r.get("quick",{}) or {}).get("gap")
        if gap and gap>=0.2: train_prompts.append((float(gap),r.get("prompt",""),tid_to_pkey.get(tid, tid),emb_data["embeddings"][i]))

    train_prompts.sort(key=lambda x: -x[0]); train_prompts = train_prompts[:20]
    print(f"Train prompts: {len(train_prompts)}")

    # Setup
    pipe = sd.load_sd_pipeline(sd_model_dir(), args.device)
    thresholds = json.loads(resolve("data/gates/naturalness.json").read_text())
    source = np.asarray(sd.Image.open(resolve("data/textures/tablecloth_gingham.png")).convert("RGB").resize((512,512)),np.float32)/255.0
    source_tv=sd.total_variation(source)
    evaluator=Pi05Evaluator(); evaluator.policy.config.chunk_size=1
    evaluator.policy.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation="eager"
    evaluator.policy.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation="eager"
    scene_root=scene_root() / "1"

    token_cache={}
    def get_tokens(prompt):
        if prompt not in token_cache: token_cache[prompt]=token_from_prompt(pipe, prompt, args.device)
        return token_cache[prompt]

    # Optimizers
    opt_actor = torch.optim.AdamW(list(model.parameters())+[p for p in model.log_std], lr=args.lr)
    opt_critic = torch.optim.AdamW(critic.parameters(), lr=args.lr)

    # ---- PPO Loop ----
    for it in range(args.iters):
        np.random.shuffle(train_prompts); batch = train_prompts[:args.prompts_per_iter]

        # Storage
        mb_states=[]; mb_actions=[]; mb_log_probs=[]; mb_rewards=[]; mb_head_idx=[]

        for teacher_gap, prompt, pkey, emb in batch:
            tokens,pooled = get_tokens(prompt); tokens=tokens.to(args.device); pooled=pooled.to(args.device)
            clean = clean_rollouts.get(pkey)

            model.eval()
            with torch.no_grad(): z_all,scores,gw = model(tokens,pooled)
            best_k = scores[0].argmax().item()

            for k in [best_k, (best_k+1) % ckpt["heads"]][:args.candidates]:
                # Forward to get bottleneck mean
                h = model.backbone(pooled); gw_out = model.gate(h)
                b_mean = model.experts[k](h) * gw_out[:,k:k+1] + model.head_emb[k].unsqueeze(0)*0.1

                # Sample from Gaussian with learned std
                std = torch.exp(model.log_std[k])
                noise = torch.randn_like(b_mean)*std; b_sample = b_mean + noise

                # Decode to z_T
                zs = model.decoder(b_sample, tokens, pooled)
                zs = zs*(SQRT_D/zs.norm(dim=-1,keepdim=True).clamp(1e-8))
                zn = zs.detach().cpu().numpy().squeeze(0)

                # Generate texture
                img=sd.generate_from_latent(pipe,prompt,zn,device=args.device)
                m=sd.texture_metrics(img,source,source_tv); gate,_=sd.natural_gate(m,thresholds)

                gap=0.0; dense_r=0.0
                if gate:
                    ip=scene_root/f'ppo_{it}_{k}.png'; img.save(ip)
                    scene=replace_tablecloth(ip, scene_root/f'ppo_{it}_{k}')
                    if clean:
                        res=evaluator.eval(scene,num_episodes=args.episodes,start_seed=0,detail=True)
                        gap=float(1.0-res["success_rate"])
                        adv_eps=res.get("episodes",[]); cln_eps=clean.get("episodes",[])
                        for ep_idx in range(min(len(adv_eps),len(cln_eps))):
                            a_adv=np.array(adv_eps[ep_idx].get("action_trace",[])); a_cln=np.array(cln_eps[ep_idx].get("action_trace",[]))
                            e_adv=np.array(adv_eps[ep_idx].get("eef_positions",[])); e_cln=np.array(cln_eps[ep_idx].get("eef_trace",[]))
                            mlen=min(len(a_adv),len(a_cln),len(e_adv),len(e_cln))
                            if mlen<2: continue
                            a_n=np.linalg.norm(a_cln[:mlen],axis=-1)+1e-8
                            r_act=np.mean(np.linalg.norm(a_adv[:mlen]-a_cln[:mlen],axis=-1)/a_n)
                            r_trj=np.mean(np.linalg.norm(e_adv[:mlen]-e_cln[:mlen],axis=-1))
                            tail=min(5,mlen); r_gl=np.linalg.norm(e_adv[-tail:].mean(0)-e_cln[-tail:].mean(0))
                            dense_r += 1.5*r_act + 6.0*r_trj + 3.0*r_gl
                        dense_r /= max(min(len(adv_eps),len(cln_eps)),1)
                    else:
                        res=evaluator.eval(scene,num_episodes=args.episodes,start_seed=0)
                        gap=float(1.0-res["success_rate"])

                reward = gap + 2.0*dense_r
                noise_vec = (b_sample - b_mean) / std
                log_prob = -0.5*(noise_vec.pow(2).sum(dim=-1)) - args.bottleneck*(0.5*np.log(2*np.pi) + model.log_std[k])

                mb_states.append(pooled.squeeze(0)); mb_actions.append(b_sample.squeeze(0))
                mb_log_probs.append(log_prob); mb_rewards.append(reward); mb_head_idx.append(k)

        # ---- PPO Update ----
        if len(mb_rewards)==0: continue

        # Stack batch
        old_log_probs = torch.stack(mb_log_probs)
        rewards_t = torch.tensor(mb_rewards,device=args.device,dtype=torch.float32)
        states_t = torch.stack(mb_states).detach()

        # Advantage: reward - V(s)
        with torch.no_grad():
            values = critic(states_t)
            advantages = rewards_t - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize

        for _ in range(args.ppo_epochs):
            # Re-forward bottlenecks for ratio computation
            model.train(); critic.train()
            all_new_log_probs = []; all_log_probs = []; all_head_idx = []; total_entropy=0.0

            for i in range(len(mb_states)):
                s, a, old_lp, k = mb_states[i], mb_actions[i], old_log_probs[i], mb_head_idx[i]
                s_t = s.unsqueeze(0); h = model.backbone(s_t); gw = model.gate(h)
                b_mean = model.experts[k](h) * gw[:,k:k+1] + model.head_emb[k].unsqueeze(0)*0.1
                std = torch.exp(model.log_std[k])
                noise_vec = (a.unsqueeze(0) - b_mean) / std
                new_lp = -0.5*noise_vec.pow(2).sum(dim=-1) - args.bottleneck*(0.5*np.log(2*np.pi) + model.log_std[k])
                all_new_log_probs.append(new_lp)

                entropy = model.log_std[k] + 0.5 + 0.5*np.log(2*np.pi*np.e)  # Gaussian entropy
                total_entropy += entropy

            new_lp_t = torch.stack(all_new_log_probs)
            ratio = torch.exp(new_lp_t.detach() - old_log_probs.detach())

            # PPO clipped loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-args.clip, 1+args.clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            new_values = critic(states_t)
            vf_loss = F.mse_loss(new_values, rewards_t)

            # Entropy bonus
            ent_loss = -args.ent_coef * total_entropy / len(mb_head_idx)

            loss = actor_loss + args.vf_coef*vf_loss + ent_loss

            opt_actor.zero_grad(); opt_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_actor.step(); opt_critic.step()

        avg_r = rewards_t.mean().item()
        print(f"ppo {it+1}/{args.iters}: reward={avg_r:.3f} actor={actor_loss.item():.3f} vf={vf_loss.item():.3f}", flush=True)

        if (it+1)%20==0:
            out=resolve(args.output_dir)/"checkpoint.pt"; out.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model":{k:v.cpu() for k,v in model.state_dict().items() if 'log_std' not in k},
                        "log_std":[s.item() for s in model.log_std],"critic":{k:v.cpu() for k,v in critic.state_dict().items()}},out)

    out=resolve(args.output_dir)/"best.pt"; out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":{k:v.cpu() for k,v in model.state_dict().items() if 'log_std' not in k},
                "log_std":[s.item() for s in model.log_std],"critic":{k:v.cpu() for k,v in critic.state_dict().items()}},out)
    print(f"Saved {out}")
    return 0

if __name__=="__main__": raise SystemExit(main())
