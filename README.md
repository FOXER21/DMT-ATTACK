# DMT

Reference implementation for **DMT: Diffusion-Prior Adversarial Texture Search for Closed-Loop VLA Patch Attacks**.

DMT searches over the entry latent of a frozen Stable Diffusion sampler, turns
the latent into a natural texture, places that texture in a LIBERO scene, and
measures the resulting closed-loop behavior of a frozen Pi0/Pi0.5 policy. This
repository contains the reproducible mainline only. Ablation studies, Table A
experiments, multi-model tables, and historical research scripts are excluded.

## Repository Layout

```text
dmt/
├── src/dmt/
│   ├── training/       # generator distillation and PPO optimization
│   ├── diffusion/      # frozen sampler, naturalness metrics, rewards
│   ├── inference/      # checkpoint -> texture inference
│   ├── scene/          # LIBERO scene materialization and texture replacement
│   ├── rollout/        # policy evaluation and paired rollout data
│   ├── runtime/        # small Pi0/Pi0.5 + LIBERO adapters
│   ├── environment/    # host and dependency checks
│   └── paths.py        # repository and external-runtime paths
├── bin/                # thin command-line wrappers
├── config/             # runtime variables, scene template, protocol file
├── data/               # external data contract; no large files are bundled
├── docs/               # method and reproduction notes
├── models/             # external model layout documentation
├── requirements/       # dependency groups
└── outputs/            # ignored local outputs
```

## Installation

```bash
cd dmt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/inference.txt
pip install -e .
cp config/runtime.env.example config/runtime.env
# edit config/runtime.env for model, LIBERO, LeRobot, and CUDA paths
```

Install `requirements/train.txt` for generator training and
`requirements/evaluation.txt` for LIBERO/MuJoCo rollouts. Model weights,
LIBERO assets, and CUDA runtimes are external and retain their own licenses.

## Commands

```bash
bin/check.sh
bin/prepare_scene.sh

# Offline distillation smoke test
bin/train_generator.sh --phase1-epochs 1 --phase2-iters 0 --device cpu

# Closed-loop PPO optimization
bin/train_ppo.sh --checkpoint outputs/training/checkpoint.pt

# Generate a texture from a trained checkpoint
bin/infer_texture.sh \
  --checkpoint outputs/training/best.pt \
  --prompt "photorealistic blue and white gingham cotton tablecloth fabric texture" \
  --output outputs/texture.png

# Paired clean or textured policy evaluation
bin/evaluate_policy.sh --model pi05 --task-suite libero_spatial \
  --output outputs/pi05_clean.json
```

After `pip install -e .`, the equivalent console commands are
`dmt-train`, `dmt-train-ppo`, `dmt-infer`, `dmt-evaluate`,
`dmt-prepare-scene`, and `dmt-check`.

## Mainline Modules

| Responsibility | Module |
| --- | --- |
| MoE generator training | `dmt.training.train_generator` |
| PPO actor-critic | `dmt.training.train_ppo` |
| Natural texture pool | `dmt.diffusion.natural_pool` |
| Frozen diffusion sampling and gates | `dmt.diffusion.sampler` |
| Terminal rollout reward | `dmt.diffusion.reward` |
| Texture inference | `dmt.inference.texture` |
| Scene construction | `dmt.scene.prepare`, `dmt.scene.tablecloth` |
| Policy rollout evaluation | `dmt.rollout.evaluate_policy` |

See [data/README.md](data/README.md) for the external data contract.
