#!/usr/bin/env python3
"""Train the cross-attention MoE texture generator with offline and online objectives.

Architecture:
  prompt → SD2.1 text encoder → tokens[77,1024] + pooled[1024]
  pooled → backbone MLP → MoE(8 experts) + gate → 8×128d bottlenecks
  tokens → Cross-Attention at each decoder block
  pooled → AdaLN-Zero modulation per block
  128d → Linear→4×8×8 → Upsample+Conv2d×3 → 4×64×64 → whiten → z_T
  DPP loss maximizes inter-head diversity. LoadBalance loss prevents expert collapse.

Phase 1: supervised distillation + DPP + LoadBalance + pseudo high-gap samples
Phase 2: REINFORCE with KL+L2 attention dense reward (v2: KL unbounded, more sensitive)
"""
from __future__ import annotations
import json, sys, time, argparse, pickle, os, hashlib, atexit, shutil
import numpy as np

LATENT_DIM = 4*64*64; SQRT_D = np.sqrt(LATENT_DIM)
from dmt.paths import (  # noqa: E402
    configure_runtime,
    resolve,
    scene_root,
    sd_model_dir,
)

configure_runtime()

# ═══════════════════════════ Architecture ═══════════════════════════

def build_generator_model(in_dim, num_heads, bottleneck_dim, hidden_dim, text_dim=1024):
    import torch, torch.nn as nn
    import torch.nn.functional as F

    class AdaLNZero(nn.Module):
        """Zero-initialized adaptive layer norm: scale, shift, gate from text pooled."""
        def __init__(self, dim, text_dim):
            super().__init__()
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.proj = nn.Linear(text_dim, dim*3)
            # Zero init for residual gating
            nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        def forward(self, x, text):
            # x: [B, C, H, W]; text: [B, text_dim]
            scale, shift, gate = self.proj(text).chunk(3, dim=-1)  # each [B, C]
            # LayerNorm over C: permute to [B, H, W, C]
            x_norm = x.permute(0, 2, 3, 1).contiguous()
            x_norm = self.norm(x_norm)  # [B, H, W, C]
            x_norm = x_norm.permute(0, 3, 1, 2)  # back to [B, C, H, W]
            scale = scale.view(-1, scale.shape[-1], 1, 1)
            shift = shift.view(-1, shift.shape[-1], 1, 1)
            gate = gate.view(-1, gate.shape[-1], 1, 1)
            return x + gate * (x_norm * (1 + scale) + shift)

    class CrossAttnBlock(nn.Module):
        """Cross-attend to text tokens at current spatial resolution."""
        def __init__(self, channels, text_dim=1024):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=channels, kdim=text_dim, vdim=text_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(channels)
        def forward(self, x, text_tokens):
            # x: [B, C, H, W]; text_tokens: [77, B, 1024] (for nn.MultiheadAttention batch_first)
            B, C, H, W = x.shape
            x_flat = x.reshape(B, C, H*W).transpose(1,2)  # [B, H*W, C]
            if text_tokens.dim() == 3 and text_tokens.shape[0] == 77:
                text = text_tokens.transpose(0,1)  # [B, 77, 1024]
            else:
                text = text_tokens  # assume [B, 77, 1024]
            attn_out, _ = self.attn(x_flat, text, text)
            x_flat = self.norm(x_flat + attn_out)
            return x_flat.transpose(1,2).reshape(B, C, H, W)

    class UpsampleBlock(nn.Module):
        def __init__(self, in_ch, out_ch, text_dim=1024, use_cross_attn=True):
            super().__init__()
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
            self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
            self.act = nn.SiLU()
            self.cross_attn = CrossAttnBlock(out_ch, text_dim) if use_cross_attn else None
            self.adaln = AdaLNZero(out_ch, text_dim)
        def forward(self, x, text_tokens, text_pooled):
            x = self.upsample(x); x = self.conv(x); x = self.norm(x); x = self.act(x)
            if self.cross_attn is not None:
                x = self.cross_attn(x, text_tokens)
            x = self.adaln(x, text_pooled)
            return x

    class ConvDecoder(nn.Module):
        """Decode bottleneck(128) → z_T(4,64,64) with cross-attn + AdaLN."""
        def __init__(self, bottleneck_dim, text_dim=1024):
            super().__init__()
            self.fc = nn.Linear(bottleneck_dim, 4*8*8)
            self.block1 = UpsampleBlock(4, 128, text_dim)   # 8→16
            self.block2 = UpsampleBlock(128, 64, text_dim)   # 16→32
            self.block3 = UpsampleBlock(64, 4, text_dim)     # 32→64
        def forward(self, bottleneck, text_tokens, text_pooled):
            h = self.fc(bottleneck).reshape(-1, 4, 8, 8)
            h = self.block1(h, text_tokens, text_pooled)
            h = self.block2(h, text_tokens, text_pooled)
            h = self.block3(h, text_tokens, text_pooled)
            return h.reshape(-1, 4*64*64)

    class MoEGate(nn.Module):
        """Gating network for 8 experts."""
        def __init__(self, hidden_dim, num_experts=8):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_experts))
        def forward(self, x):
            return F.softmax(self.net(x), dim=-1)

    class Generator(nn.Module):
        def __init__(self, in_dim, num_heads=8, bottleneck_dim=128, hidden_dim=512, text_dim=1024):
            super().__init__()
            self.num_heads = num_heads
            # Backbone
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            )
            # MoE: 8 experts
            self.experts = nn.ModuleList([nn.Linear(hidden_dim, bottleneck_dim) for _ in range(num_heads)])
            self.gate = MoEGate(hidden_dim, num_heads)
            # Shared decoder
            self.decoder = ConvDecoder(bottleneck_dim, text_dim)
            # Score heads
            self.score_head = nn.ModuleList([
                nn.Sequential(nn.Linear(bottleneck_dim, 64), nn.ReLU(), nn.Linear(64, 1))
                for _ in range(num_heads)
            ])
            # Per-head embeddings for diversity
            self.head_emb = nn.Parameter(torch.randn(num_heads, bottleneck_dim) * 0.02)

        def forward(self, text_tokens, text_pooled):
            """text_tokens: [B, 77, 1024], text_pooled: [B, 1024]"""
            h = self.backbone(text_pooled)
            gate_w = self.gate(h)  # [B, 8]

            z_all = []; scores = []
            for i in range(self.num_heads):
                # MoE: each expert conditioned by gate
                expert_out = self.experts[i](h)
                bottleneck = expert_out * gate_w[:, i:i+1] + self.head_emb[i].unsqueeze(0) * 0.1
                z = self.decoder(bottleneck, text_tokens, text_pooled)
                z = z * (SQRT_D / z.norm(dim=-1, keepdim=True).clamp(1e-8))
                z_all.append(z)
                scores.append(self.score_head[i](bottleneck))

            return torch.stack(z_all, dim=1), torch.cat(scores, dim=-1), gate_w

    return Generator(in_dim, num_heads, bottleneck_dim, hidden_dim, text_dim)

# ═══════════════════════════ DPP Loss ═══════════════════════════

def dpp_loss(z_stack):
    """Determinantal Point Process: maximize determinant of cosine similarity matrix."""
    import torch
    z_norm = z_stack / (z_stack.norm(dim=-1, keepdim=True).clamp(1e-8))  # [B, H, D]
    cos = torch.einsum('bhd,bkd->bhk', z_norm, z_norm)  # [B, H, H]
    eye = torch.eye(z_stack.shape[1], device=z_stack.device).unsqueeze(0)
    det = torch.linalg.det(cos + 0.1 * eye)  # regularize for stability
    return -torch.log(det.clamp(min=1e-6)).mean()

def token_from_prompt(pipe, prompt, device):
    """Extract SD2.1 text tokens [77, 1024] and pooled [1024]."""
    import torch
    tok = pipe.tokenizer; enc = pipe.text_encoder
    inputs = tok(prompt, return_tensors="pt", padding="max_length", max_length=tok.model_max_length, truncation=True)
    ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        h = enc(ids).last_hidden_state  # [1, 77, 1024]
        eos_idx = (ids == tok.eos_token_id).float().argmax(dim=-1)
        pooled = h[torch.arange(len(ids)), eos_idx].float()
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True).clamp(1e-8))
    return h.float(), pooled  # [1, 77, 1024], [1, 1024]

def load_balance_loss(gate_probs):
    """Switch Transformer style: encourage equal expert usage across batch.
    gate_probs: [B, num_experts] softmax. Minimized when each expert gets ~1/N share.
    Returns scalar ≥ 1.0; 1.0 = perfect balance, >1.0 = imbalance."""
    N = gate_probs.shape[-1]
    f = gate_probs.mean(dim=0)  # [N] mean gate prob per expert across batch
    return N * (f ** 2).sum()   # range [1, N] — 1 when all f_i = 1/N

# ═══════════════════════════ Training ═══════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-records", default="data/teacher/records.jsonl")
    ap.add_argument("--phase1-epochs", type=int, default=5000)
    ap.add_argument("--phase2-iters", type=int, default=2000)
    ap.add_argument("--phase2-candidates", type=int, default=3)
    ap.add_argument("--phase2-episodes", type=int, default=15)
    ap.add_argument("--phase2-prompts", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--output-dir", default="outputs/training"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--lr-phase2", type=float, default=3e-4)
    ap.add_argument("--w-dpp", type=float, default=0.01, help="DPP diversity loss weight")
    ap.add_argument("--w-lb", type=float, default=0.05, help="MoE load balance loss weight")
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    ap.add_argument("--start-iter", type=int, default=-1, help="Global iter offset; -1 = read from checkpoint")
    ap.add_argument("--phase2-total", type=int, default=0, help="Global curriculum horizon for curriculum and soft-gate schedules (0=use --phase2-iters). Keep it fixed across resumed chunks.")
    ap.add_argument("--cleanup-low-gap-scenes", action="store_true",
                    help="Delete generated candidate scene dirs after evaluation unless their gap is high enough to keep.")
    ap.add_argument("--keep-gap-threshold", type=float, default=0.2,
                    help="When --cleanup-low-gap-scenes is set, keep scene dirs with gap above this threshold.")
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()
    np.random.seed(args.seed)

    import torch, torch.nn as nn, torch.nn.functional as F
    import dmt.diffusion.sampler as sd
    torch.set_grad_enabled(True)
    # Belt-and-suspenders: if any compiled path survives, don't let dynamo endlessly
    # recompile (the attn_hook's growing stored_attn list triggers guard failures).
    # Falling back to eager on cache miss is far cheaper than a recompile storm.
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.suppress_errors = True

    # ---- Data ----
    teacher_records_path = resolve(args.teacher_records)
    records = [json.loads(l) for l in teacher_records_path.read_text().strip().split('\n') if l.strip()]
    rec_by_id = {r["teacher_id"]: r for r in records}
    emb_data = np.load(resolve("data/teacher/sd_embeddings.npz"), allow_pickle=True)
    teacher_ids = list(emb_data["teacher_ids"])

    train_prompts = []; Z_list = []; token_cache = {}
    pipe = sd.load_sd_pipeline(sd_model_dir(), args.device)

    for i, tid in enumerate(teacher_ids):
        r = rec_by_id.get(tid, {}); gap = (r.get("quick",{}) or {}).get("gap"); lp = r.get("latent_path","")
        if gap is not None and gap>=0.1 and lp:
            lp_path = resolve(lp)
            if lp_path.exists():
                prompt = r.get("prompt","")
                Z_list.append(np.load(lp_path)["z"].astype(np.float32))
                train_prompts.append((prompt, float(gap), tid))
                if prompt not in token_cache:
                    token_cache[prompt] = token_from_prompt(pipe, prompt, args.device)

    Z = torch.tensor(np.stack(Z_list), dtype=torch.float32, device=args.device)
    print(f"Teachers: {len(Z)} | text tokens precomputed: {len(token_cache)}")

    # Train/val split
    idx = np.random.permutation(len(Z))
    n_train = int(len(Z)*0.7); train_idx = torch.tensor(idx[:n_train], device=args.device)
    val_idx = torch.tensor(idx[n_train:], device=args.device)

    # ---- Model ----
    model = build_generator_model(1024, args.heads, args.bottleneck, args.hidden, 1024).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params | Cross-Attn + MoE + DPP")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf"); best_state = None; no_improve = 0; patience = 300

    # ═══════════════════ Phase 1: Distill + DPP + LB ═══════════════════
    print(f"\n=== Phase 1: Distill + DPP + LB ({args.phase1_epochs} epochs) ===")

    # Gap buffer: store (tokens, pooled, z_vector, gap) for high-gap pseudo-samples
    gap_buffer = []  # populated from teacher data initially, then from Phase 2
    for i in range(len(train_prompts)):
        _, g, _ = train_prompts[i]
        if g >= 0.15:  # pre-populate from above-average teacher gaps
            p = train_prompts[i][0]
            if p in token_cache:
                gap_buffer.append((token_cache[p][0], token_cache[p][1], Z[i:i+1], g))
    gap_buffer.sort(key=lambda x: -x[3])  # sort by gap descending
    gap_buffer = gap_buffer[:max(10, len(gap_buffer)//3)]  # keep top ~1/3
    print(f"Gap buffer: {len(gap_buffer)} pseudo-samples (top gap={gap_buffer[0][3]:.3f})" if gap_buffer else "Gap buffer: empty")

    for ep in range(args.phase1_epochs):
        model.train()
        total_loss = torch.tensor(0.0, device=args.device)

        for idx_in_batch, data_idx in enumerate(train_idx.cpu().numpy()):
            prompt, teacher_gap, tid = train_prompts[data_idx]
            tokens, pooled = token_cache[prompt]
            tokens = tokens.to(args.device); pooled = pooled.to(args.device)
            z_teacher = Z[data_idx:data_idx+1]

            z_all, scores, gate_w = model(tokens, pooled)  # [1,8,16384], [1,8], [1,8]

            # Match loss: each head against teacher
            z_pred = z_all[0]; z_t = z_teacher[0]
            cos_sim = F.cosine_similarity(z_pred, z_t.unsqueeze(0).expand(args.heads,-1), dim=-1)
            mse_per_head = ((z_pred - z_t.unsqueeze(0).expand(args.heads,-1))**2).mean(dim=-1)
            match_cost = mse_per_head + 0.3*(1-cos_sim)
            best_head = match_cost.argmin()
            best_z = z_pred[best_head]
            cos = F.cosine_similarity(best_z, z_t, dim=-1)
            mse = ((best_z - z_t)**2).mean()
            match_loss = mse + 0.3*(1-cos)

            # Score loss
            score_loss = F.cross_entropy(scores, best_head.unsqueeze(0))

            # DPP diversity loss
            dpp = dpp_loss(z_all)  # [1,8,16384]

            loss = match_loss + 0.1*score_loss + args.w_dpp*dpp + args.w_lb*load_balance_loss(gate_w)
            total_loss += loss

        # Insert 1 pseudo high-gap sample per epoch
        pseudo_count = 0
        if gap_buffer:
            tokens_f, pooled_f, z_fake, fake_gap = gap_buffer[np.random.randint(len(gap_buffer))]
            tokens_f = tokens_f.to(args.device); pooled_f = pooled_f.to(args.device)
            z_fake = z_fake.to(args.device)
            z_all_f, scores_f, gate_w_f = model(tokens_f, pooled_f)
            z_pred_f = z_all_f[0]; z_t_f = z_fake[0]
            cos_sim_f = F.cosine_similarity(z_pred_f, z_t_f.unsqueeze(0).expand(args.heads,-1), dim=-1)
            mse_f = ((z_pred_f - z_t_f.unsqueeze(0).expand(args.heads,-1))**2).mean(dim=-1)
            best_f = (mse_f + 0.3*(1-cos_sim_f)).argmin()
            match_f = mse_f[best_f] + 0.3*(1-cos_sim_f[best_f])
            score_f = F.cross_entropy(scores_f, best_f.unsqueeze(0))
            dpp_f = dpp_loss(z_all_f)
            lb_f = load_balance_loss(gate_w_f)
            pseudo_loss = match_f + 0.1*score_f + args.w_dpp*dpp_f + args.w_lb*lb_f
            total_loss += 0.5 * pseudo_loss  # half weight for pseudo-sample
            pseudo_count = 1

        avg_loss = total_loss / (len(train_idx) + pseudo_count)
        opt.zero_grad(); avg_loss.backward(); opt.step()

        if (ep+1)%200==0:
            model.eval(); val_losses = []
            with torch.no_grad():
                for data_idx in val_idx.cpu().numpy():
                    prompt, _, tid = train_prompts[data_idx]
                    tokens, pooled = token_cache[prompt]
                    tokens=tokens.to(args.device); pooled=pooled.to(args.device)
                    z_all, scores, _ = model(tokens, pooled)
                    z_t = Z[data_idx:data_idx+1]
                    z_pred = z_all[0]; z_t_exp = z_t.unsqueeze(1).expand(-1, args.heads, -1).squeeze(0)
                    cos_sim = F.cosine_similarity(z_pred, z_t_exp, dim=-1)
                    mse = ((z_pred - z_t_exp)**2).mean(dim=-1)
                    val_losses.append((mse+0.3*(1-cos_sim)).min().item())
            val_mean = np.mean(val_losses)
            # Gate usage diagnostics: compute mean gate prob per expert across val set
            gate_means = []
            with torch.no_grad():
                for data_idx in val_idx.cpu().numpy():
                    _, pooled = token_cache[train_prompts[data_idx][0]]
                    pooled = pooled.to(args.device)
                    h = model.backbone(pooled); gw = model.gate(h)
                    gate_means.append(gw.squeeze(0).cpu().numpy())
            gm = np.stack(gate_means, axis=0)  # [val_size, 8]
            gm_mean = gm.mean(axis=0); gm_std = gm.std(axis=0)
            lb_val = 8.0 * (gm_mean ** 2).sum()  # same formula as load_balance_loss
            print(f"  ep {ep+1}: loss={avg_loss.item():.4f} val={val_mean:.4f} | "
                  f"gate_mean={np.array2string(gm_mean, precision=2, suppress_small=True)} "
                  f"lb={lb_val:.3f}", flush=True)
            if val_mean < best_val:
                best_val = val_mean; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                no_improve = 0
            else: no_improve += 1
            if no_improve >= patience: break

    if best_state: model.load_state_dict(best_state)
    print(f"Phase 1 done. Best val={best_val:.4f}")

    # ═══════════════════ Phase 2: PPO + Dense Reward ═══════════════════
    if args.phase2_iters == 0:
        out = resolve(args.output_dir)/"best.pt"; out.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"model":{k:v.cpu() for k,v in model.state_dict().items()},"heads":args.heads,"bottleneck":args.bottleneck,"hidden":args.hidden},out)
        print(f"Saved {out} (Phase 1 only)"); return 0

    from dmt.rollout.surrogate_data import Pi05Evaluator
    from dmt.scene.tablecloth import replace_tablecloth
    from lerobot.utils.constants import ACTION

    # Load clean baselines for paired comparison
    bl_path = resolve("data/baselines/clean_rollouts.pkl")
    clean_rollouts = {}
    if bl_path.exists():
        with open(bl_path, "rb") as f: clean_rollouts = pickle.load(f)
        print(f"Loaded {len(clean_rollouts)} clean baselines for dense reward")
    else:
        print("WARNING: No clean baselines found, using gap-only reward")

    print(f"\n=== Phase 2: PPO + Dense Reward ({args.phase2_iters} iters) ===")
    evaluator = Pi05Evaluator()
    atexit.register(evaluator.close)
    evaluator.policy.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    evaluator.policy.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    print(f"PPO + three-phase curriculum (2:4:4 attention→failure→full) | KL+L2 | tasks=0-9 | gap_buffer={len(gap_buffer)}")

    # Attention hooks
    gemma = evaluator.policy.model.paligemma_with_expert.gemma_expert
    stored_attn = []; hook_handles = []
    def attn_hook(m,i,o):
        if isinstance(o,tuple) and len(o)>=2 and o[1] is not None: stored_attn.append(o[1].detach().cpu())
    for lid in [14,15,16,17]: hook_handles.append(gemma.model.layers[lid].self_attn.register_forward_hook(attn_hook))
    print(f"Attention hooks: {len(hook_handles)} layers")

    # Helper: extract saliency
    def extract_saliency(attn_list):
        if not attn_list: return None,None
        ag,wb=[],[]
        for aw in attn_list:
            af=aw.float().mean(dim=[0,1,2]).cpu().numpy()
            ag.append(af[0:256].reshape(16,16));wb.append(af[256:512].reshape(16,16))
        ags=np.mean(ag,axis=0);wbs=np.mean(wb,axis=0)
        for s in [ags,wbs]:
            mn,mx=s.min(),s.max()
            if mx>mn: s[:]=(s-mn)/(mx-mn)
        return ags.astype(np.float32),wbs.astype(np.float32)

    def kl_div(p, q, eps=1e-6):
        """KL(p||q): asymmetric, sensitive to small differences. Per-element (D=256)."""
        p = np.clip(p.ravel(), eps, None); q = np.clip(q.ravel(), eps, None)
        p /= p.sum(); q /= q.sum()
        return np.sum(p * np.log(p / q))

    def attn_reward(ag_sal, cln_ag):
        """reward_A = 4.0*KL/D + 2.0*L2/sqrt(D). D=256 for 16x16 saliency.
        Divisor normalization keeps scale comparable to original JS divergence."""
        D = float(ag_sal.size)  # 256
        kl = kl_div(ag_sal, cln_ag)
        l2 = np.linalg.norm(ag_sal.ravel() - cln_ag.ravel())
        return 4.0 * kl / D + 2.0 * l2 / np.sqrt(D)

    thresholds = json.loads(resolve("data/gates/naturalness.json").read_text())
    source = np.asarray(sd.Image.open(resolve("data/textures/tablecloth_gingham.png")).convert("RGB").resize((512,512)),np.float32)/255.0
    source_tv = sd.total_variation(source)
    # 10 scene-task pairs: scene_id → libero_spatial task_id
    SCENE_TASKS = [(scene_root() / str(i), i) for i in range(10)]
    print(f"Scene-task pool: {len(SCENE_TASKS)} (task_id 0-9)")

    # ---- PPO Setup: Critic + learnable log_std ----
    class Critic(nn.Module):
        def __init__(self, d=1024):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(d,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
        def forward(self,x): return self.net(x).squeeze(-1)
    critic = Critic(1024).to(args.device)
    model.log_std = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(args.heads)])  # σ≈1.65, explosive exploration
    for p in model.log_std: p.to(args.device)
    opt_critic = torch.optim.AdamW(critic.parameters(), lr=args.lr_phase2)

    # Resume from checkpoint
    resume_history = []
    start_iter = args.start_iter
    if args.resume:
        ckpt_path = resolve(args.resume)
        print(f"Resuming from {ckpt_path}")
        resume_ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        if "critic" in resume_ckpt:
            critic.load_state_dict(resume_ckpt["critic"])
        if "log_std" in resume_ckpt:
            for i, v in enumerate(resume_ckpt["log_std"]):
                model.log_std[i].data = torch.tensor(v, device=args.device)
        resume_history = resume_ckpt.get("history", [])
        # --start-iter == -1 (default) → infer next iter from the checkpoint's history;
        # an explicit >=0 value overrides (e.g. auto-restart wrapper passes the exact offset).
        if args.start_iter < 0:
            start_iter = (int(resume_history[-1].get("iter", len(resume_history) - 1)) + 1
                          if resume_history else 0)
        print(f"  Loaded: model + critic + log_std | history: {len(resume_history)} iters | start_iter={start_iter}", flush=True)
    if start_iter < 0:
        start_iter = 0
    print(f"PPO: Critic {sum(p.numel()/1e6 for p in critic.parameters()):.1f}M + {len(model.log_std)} log_std | dense reward | clip={0.2} ent={0.01}")

    # Top prompts for REINFORCE - include prompt_key for baseline lookup
    train_data = [(g, p, tid) for p, g, tid in train_prompts]
    # Add prompt_key lookup: teacher_id → prompt_key
    tid_to_pkey = {}
    for r in records:
        tid = r.get('teacher_id','')
        tid_to_pkey[tid] = r.get("prompt_key") or tid
    train_data = [(g, p, tid_to_pkey.get(tid, tid)) for g, p, tid in train_data if g >= 0.2]
    train_data.sort(key=lambda x: -x[0])
    top_prompts = train_data[:20]

    opt_actor = torch.optim.AdamW(list(model.parameters()) + [p for p in model.log_std], lr=args.lr_phase2)
    # Carry only history entries that precede this run's global start_iter. This keeps a
    # real cross-chunk timeline while discarding a warm-start checkpoint's foreign history
    # (e.g. nocompile's ~100 iters when we launch a fresh run at start_iter=0).
    history = [h for h in resume_history if int(h.get("iter", -1)) < start_iter]
    rew_ema_mean = 0.01; rew_ema_var = 0.1
    # Total horizon for curriculum schedules and soft gate. Stays fixed
    # across restart chunks so the documented curriculum stays on one timeline.
    global_total = args.phase2_total if args.phase2_total > 0 else args.phase2_iters

    # ═══════════════════ Buffer Supervision Pretraining ═══════════════════
    # Only on a FRESH run (start_iter==0). On chunked restarts the model already
    # carries PPO progress; re-pretraining toward buffer MSE each chunk would
    # repeatedly undo it. A resumed run skips straight to PPO.
    if gap_buffer and start_iter == 0:
        print(f"Buffer supervision: {len(gap_buffer)} samples, 300 steps...")
        opt_sup = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for bs_step in range(300):
            idx_b = np.random.randint(len(gap_buffer))
            t_b, p_b, z_b, _ = gap_buffer[idx_b]
            t_b, p_b, z_b = t_b.to(args.device), p_b.to(args.device), z_b.to(args.device)
            z_all_b, scores_b, gw_b = model(t_b, p_b)
            mse_b = ((z_all_b - z_b.unsqueeze(1).expand(-1, args.heads, -1))**2).mean()
            lb_b = load_balance_loss(gw_b)
            loss_b = mse_b + args.w_lb * lb_b
            opt_sup.zero_grad(); loss_b.backward(); opt_sup.step()
        print(f"  Buffer pretraining done", flush=True)

    # ═══════════════════ Phase 2: PPO + Dense Reward ═══════════════════
    print(f"\n=== Phase 2: PPO + Dense (action_div + buffer sup) ===")
    for it_local in range(args.phase2_iters):
        # Global iter = resume offset + local step. All schedules (curriculum ratio, soft
        # gate, task cycling) key off `it` so periodic restarts (to reclaim leaked
        # MuJoCo/EGL memory) resume seamlessly instead of restarting the curriculum.
        it = start_iter + it_local
        # Soft gate factor: early iters relax thresholds for exploration
        pct = it / max(global_total, 1)
        soft_factor = 1.0 if pct > 0.2 else (0.5 + 0.5 * pct / 0.2)  # 0.5→1.0 linearly

        np.random.shuffle(top_prompts); batch = top_prompts[:args.phase2_prompts]
        all_gaps = []
        total_candidates = 0; n_buffer = 0; n_random = 0
        # PPO storage
        ppo_states=[]; ppo_actions=[]; ppo_rewards=[]; ppo_heads=[]; ppo_old_lps=[]

        for prompt_idx, (teacher_gap, prompt, pkey) in enumerate(batch):
            tokens, pooled = token_cache.get(prompt, (None, None))
            if tokens is None: tokens, pooled = token_from_prompt(pipe, prompt, args.device)
            tokens = tokens.to(args.device); pooled = pooled.to(args.device)

            clean = clean_rollouts.get(pkey) if clean_rollouts else None
            # Cycle scene-task every 10 iters. With texture hot-swap the per-candidate env
            # rebuild is gone; only a task *change* rebuilds the env (~20s), so 10-iter
            # cycling keeps full task diversity at negligible amortized cost.
            scene_root, task_id = SCENE_TASKS[(it // 10) % len(SCENE_TASKS)]

            # Generate candidates (model experts + buffer perturbation + random injection)
            model.eval()
            with torch.no_grad():
                z_all, scores, gate_w = model(tokens, pooled)
            best_k = scores[0].argmax().item()

            # Build candidate list
            candidate_specs = []
            for k in range(min(args.phase2_candidates, args.heads)):
                candidate_specs.append(('model', (best_k + k) % args.heads))
            # Gap buffer perturbation (70% probability, if buffer non-empty)
            if gap_buffer and np.random.random() < 0.7:
                candidate_specs.append(('buffer', best_k))
            # Random noise injection (20% probability)
            if np.random.random() < 0.2:
                candidate_specs.append(('random', best_k))

            total_candidates += len(candidate_specs)
            n_buffer += sum(1 for c in candidate_specs if c[0]=='buffer')
            n_random += sum(1 for c in candidate_specs if c[0]=='random')

            for cand_type, k_head in candidate_specs:
                h = model.backbone(pooled)
                gw = model.gate(h)

                if cand_type == 'model':
                    expert_out = model.experts[k_head](h)
                    b_mean = expert_out * gw[:, k_head:k_head+1] + model.head_emb[k_head].unsqueeze(0) * 0.1
                    std = torch.exp(model.log_std[k_head])
                    noise = torch.randn_like(b_mean) * std; bs = b_mean + noise
                    zs = model.decoder(bs, tokens, pooled)
                    zs = zs * (SQRT_D / zs.norm(dim=-1, keepdim=True).clamp(1e-8))
                    store_ppo = True
                elif cand_type == 'buffer':
                    # Perturb successful buffer z, use directly
                    buf_idx = np.random.randint(len(gap_buffer))
                    _, _, z_buf, _ = gap_buffer[buf_idx]
                    z_buf = z_buf.to(args.device)
                    noise_scale = 0.3 * torch.exp(model.log_std[best_k]).item()
                    zs = z_buf + torch.randn_like(z_buf) * noise_scale
                    zs = zs * (SQRT_D / zs.norm(dim=-1, keepdim=True).clamp(1e-8))
                    b_mean = model.experts[k_head](h) * gw[:, k_head:k_head+1] + model.head_emb[k_head].unsqueeze(0)*0.1
                    std = torch.exp(model.log_std[k_head])
                    bs = b_mean  # placeholder
                    store_ppo = False  # exploration-only, skip PPO
                else:  # random
                    zs = torch.randn(1, LATENT_DIM, device=args.device)
                    zs = zs * (SQRT_D / zs.norm(dim=-1, keepdim=True).clamp(1e-8))
                    b_mean = model.experts[k_head](h) * gw[:, k_head:k_head+1] + model.head_emb[k_head].unsqueeze(0)*0.1
                    std = torch.exp(model.log_std[k_head])
                    bs = b_mean  # placeholder
                    store_ppo = False  # exploration-only

                zn = zs.detach().cpu().numpy().squeeze(0)
                img = sd.generate_from_latent(pipe, prompt, zn, device=args.device)
                m = sd.texture_metrics(img, source, source_tv)
                # Soft gate: relaxed thresholds in early iterations
                if soft_factor < 1.0:
                    gate_keys = ['tv_ratio_p90', 'hf_energy_p90', 'saturation_p95_p95', 'color_entropy_p05']
                    soft_th = dict(thresholds)  # copy
                    for gk in gate_keys:
                        if gk in soft_th and isinstance(soft_th[gk], (int, float)):
                            soft_th[gk] *= soft_factor
                    gate, _ = sd.natural_gate(m, soft_th)
                else:
                    gate, _ = sd.natural_gate(m, thresholds)

                gap = 0.0; attn_r = 0.0; act_r = 0.0; trj_r = 0.0; gl_r = 0.0
                if gate:
                    scene_dir = scene_root/f'candidate_{it}_{k_head}'  # unique per candidate → each gets its own texture
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    ip = scene_dir/'tex.png'; img.save(ip)
                    scene = replace_tablecloth(ip, scene_dir)
                    if clean:
                        stored_attn.clear()
                        res = evaluator.eval(scene, num_episodes=args.phase2_episodes, start_seed=0, detail=True, task_id=task_id)
                        n_attn_debug = len(stored_attn)  # diagnostic: did hooks fire?
                        gap = float(1.0 - res["success_rate"])
                        adv_eps = res.get("episodes", [])
                        cln_eps = clean.get("episodes", [])
                        n_pair = min(len(adv_eps), len(cln_eps))
                        for ep_idx in range(n_pair):
                            a_adv = np.array(adv_eps[ep_idx].get("action_trace", []))
                            a_cln = np.array(cln_eps[ep_idx].get("action_trace", []))
                            e_adv = np.array(adv_eps[ep_idx].get("eef_positions", []))
                            e_cln = np.array(cln_eps[ep_idx].get("eef_trace", []))
                            mlen = min(len(a_adv), len(a_cln), len(e_adv), len(e_cln))
                            if mlen < 2: continue
                            a_n = np.linalg.norm(a_cln[:mlen], axis=-1) + 1e-8
                            act_r += np.mean(np.linalg.norm(a_adv[:mlen] - a_cln[:mlen], axis=-1) / a_n)
                            trj_r += np.mean(np.linalg.norm(e_adv[:mlen] - e_cln[:mlen], axis=-1))
                            tail = min(5, mlen); gl_r += np.linalg.norm(e_adv[-tail:].mean(axis=0) - e_cln[-tail:].mean(axis=0))
                            # Attention offset (from chunk=1 hooks)
                            if stored_attn:
                                ag_sal, wb_sal = extract_saliency(stored_attn)
                                cln_at = cln_eps[ep_idx].get("attn_traces", [])
                                if cln_at:
                                    cln_ag = cln_at[0][1] if cln_at[0][1] is not None else np.ones((16,16))*0.02
                                    cln_wb = cln_at[0][2] if len(cln_at)>0 and cln_at[0][2] is not None else np.ones((16,16))*0.02
                                    if ag_sal is not None: attn_r += attn_reward(ag_sal, cln_ag) + attn_reward(wb_sal, cln_wb)
                        n_pair = max(n_pair, 1)
                        act_r /= n_pair; trj_r /= n_pair; gl_r /= n_pair; attn_r /= n_pair
                    else:
                        res = evaluator.eval(scene, num_episodes=args.phase2_episodes, start_seed=0, task_id=task_id)
                        gap = float(1.0 - res["success_rate"])

                all_gaps.append(gap)
                # Add to gap buffer if this is a successful attack
                if gap > 0.2 and gate:
                    z_tensor = torch.tensor(zn, dtype=torch.float32, device='cpu').unsqueeze(0)
                    gap_buffer.append((tokens.cpu(), pooled.cpu(), z_tensor, gap))
                    gap_buffer.sort(key=lambda x: -x[3])
                    # Keep buffer size manageable
                    if len(gap_buffer) > 30:
                        gap_buffer[:] = gap_buffer[:30]
                if gate and args.cleanup_low_gap_scenes and gap < args.keep_gap_threshold:
                    shutil.rmtree(scene_dir, ignore_errors=True)
                # Three-phase reward curriculum from docs/method.md.
                if pct < 0.25:
                    reward = 6.0 * attn_r
                elif pct < 0.55:
                    reward = gap + 6.0 * attn_r
                else:
                    reward = gap + 1.5 * act_r + 6.0 * trj_r + 3.0 * gl_r + 6.0 * attn_r
                # Adaptive reward normalization (EMA)
                rew_ema_mean = 0.99 * rew_ema_mean + 0.01 * reward
                rew_ema_var  = 0.99 * rew_ema_var  + 0.01 * (reward - rew_ema_mean)**2
                reward_norm = (reward - rew_ema_mean) / (np.sqrt(rew_ema_var) + 1e-8)
                # Store for PPO (skip buffer/random exploration-only candidates)
                if store_ppo:
                    noise_vec = (bs - b_mean) / std
                    old_lp = -0.5*noise_vec.pow(2).sum(dim=-1) - args.bottleneck*(0.5*np.log(2*np.pi) + model.log_std[k_head])
                    ppo_states.append(pooled.squeeze(0).detach())
                    ppo_actions.append(bs.squeeze(0).detach())
                    ppo_rewards.append(reward_norm)
                    ppo_heads.append(k_head)
                    ppo_old_lps.append(old_lp.detach())

        # ---- PPO Update ----
        if len(ppo_rewards) > 0:
            states_t = torch.stack([s.detach() for s in ppo_states])
            old_lps_t = torch.stack([lp.detach() for lp in ppo_old_lps])
            rewards_t = torch.tensor(ppo_rewards, device=args.device, dtype=torch.float32)

            # Advantage: reward - V(s)
            with torch.no_grad():
                values = critic(states_t)
                advantages = rewards_t - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ppo_ep = 4
            for _ in range(ppo_ep):
                # Recompute new log_probs
                new_lps = []
                entropies = []
                for i in range(len(ppo_states)):
                    s_t = ppo_states[i].unsqueeze(0); a = ppo_actions[i]
                    k = ppo_heads[i]; h = model.backbone(s_t); gw = model.gate(h)
                    b_mean = model.experts[k](h) * gw[:,k:k+1] + model.head_emb[k].unsqueeze(0)*0.1
                    std = torch.exp(model.log_std[k])
                    nv = (a.unsqueeze(0) - b_mean) / std
                    new_lp = -0.5*nv.pow(2).sum(dim=-1) - args.bottleneck*(0.5*np.log(2*np.pi) + model.log_std[k])
                    new_lps.append(new_lp)
                    entropies.append(model.log_std[k] + 0.5 + 0.5*np.log(2*np.pi*np.e))

                new_lps_t = torch.stack(new_lps)
                ratio = torch.exp(new_lps_t - old_lps_t)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                new_values = critic(states_t)
                vf_loss = F.mse_loss(new_values, rewards_t)
                ent_loss = -0.2 * torch.stack(entropies).mean()  # strong exploration incentive

                # MoE load balance: compute from current batch gate weights
                gate_ws_batch = []
                for i in range(len(ppo_states)):
                    s_t = ppo_states[i].unsqueeze(0); h_ = model.backbone(s_t); gw_ = model.gate(h_)
                    gate_ws_batch.append(gw_.squeeze(0))
                lb_loss = load_balance_loss(torch.stack(gate_ws_batch))
                last_gw_batch = torch.stack(gate_ws_batch).detach()

                # Buffer supervision: pull model toward known high-gap z's
                if gap_buffer and np.random.random() < 0.5:
                    idx_sup = np.random.randint(len(gap_buffer))
                    t_s, p_s, z_s, _ = gap_buffer[idx_sup]
                    t_s, p_s, z_s = t_s.to(args.device), p_s.to(args.device), z_s.to(args.device)
                    z_all_s, _, gw_s = model(t_s, p_s)
                    sup_loss = 0.05 * ((z_all_s - z_s.unsqueeze(1).expand(-1, args.heads, -1))**2).mean()
                else:
                    sup_loss = torch.tensor(0.0, device=args.device)

                loss = actor_loss + 0.5*vf_loss + ent_loss + args.w_lb*lb_loss + sup_loss
                opt_actor.zero_grad(); opt_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_actor.step(); opt_critic.step()

        avg_g = np.mean(all_gaps) if all_gaps else 0
        avg_r = np.mean(ppo_rewards) if ppo_rewards else 0
        # Diagnostic: was stored_attn populated?
        attn_fired = ('n_attn_debug' in dir() and n_attn_debug > 0)
        # Gate usage: mean expert probability across this iteration's samples
        if ppo_states:
            with torch.no_grad():
                gate_means_ppo = []
                for s in ppo_states:
                    h_ppo = model.backbone(s.unsqueeze(0)); gw_ppo = model.gate(h_ppo)
                    gate_means_ppo.append(gw_ppo.squeeze(0).cpu().numpy())
                gm_ppo = np.stack(gate_means_ppo, axis=0).mean(axis=0)  # [8]
                lb_ppo = 8.0 * (gm_ppo ** 2).sum()
                gate_str = f"gm={np.array2string(gm_ppo, precision=2, suppress_small=True)} lb={lb_ppo:.2f}"
        else:
            gate_str = ""
        history.append({"iter":it,"avg_gap":avg_g,"avg_reward":avg_r})
        print(f"  ppo {it+1}/{global_total}: reward={avg_r:.3f} gap={avg_g:.3f} gated={len(ppo_rewards)} | "
              f"fire={attn_fired} buf={len(gap_buffer)} cand={total_candidates}(+{n_buffer}b+{n_random}r) sf={soft_factor:.1f} | {gate_str}", flush=True)
        if (it+1)%25==0:
            out = resolve(args.output_dir)/"checkpoint.pt"; out.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model":{k:v.cpu() for k,v in model.state_dict().items()},"critic":{k:v.cpu() for k,v in critic.state_dict().items()},"log_std":[s.item() for s in model.log_std],"history":history,"heads":args.heads,"bottleneck":args.bottleneck,"hidden":args.hidden},out)

    # Final save (includes critic+log_std so a wrapper can --resume seamlessly)
    out = resolve(args.output_dir)/"checkpoint.pt"; out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":{k:v.cpu() for k,v in model.state_dict().items()},"critic":{k:v.cpu() for k,v in critic.state_dict().items()},"log_std":[s.item() for s in model.log_std],"history":history,"heads":args.heads,"bottleneck":args.bottleneck,"hidden":args.hidden},out)
    torch.save({"model":{k:v.cpu() for k,v in model.state_dict().items()},"history":history,"heads":args.heads,"bottleneck":args.bottleneck,"hidden":args.hidden}, resolve(args.output_dir)/"best.pt")
    for handle in hook_handles:
        handle.remove()
    evaluator.close()
    print(f"Saved {out} | final avg_gap={np.mean([h['avg_gap'] for h in history[-10:]]):.3f}")
    return 0

if __name__=="__main__": raise SystemExit(main())
