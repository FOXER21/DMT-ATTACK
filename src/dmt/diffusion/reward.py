#!/usr/bin/env python3
"""Terminal black-box reward from real rollouts.

Reward is computed ONLY from new rollouts (never from history/critic). Layer 1:
multi-seed SR gap. Layer 2: dense failure proxies (steps / eef shortfall /
paired action divergence). Hard gates (natural + siglip + latent radius) floor
the reward when violated.

The optimizer minimizes the negative reward; success labels still come from
the configured rollout success rate.
"""
from __future__ import annotations

import numpy as np

FLOOR = -10.0  # hard-gate violation reward (worst possible)


def _mean_steps_ratio(episodes: list[dict], max_steps: int = 280) -> float:
    """Layer-2 w1: among SUCCESSFUL episodes, slower success = closer to failure.
    Returns mean(steps/max_steps) over successes; 1.0 if none succeeded."""
    s = [ep["steps"] / max_steps for ep in episodes if ep.get("success")]
    return float(np.mean(s)) if s else 1.0


def _eef_shortfall(episodes_c: list[dict], episodes_b: list[dict]) -> float:
    """Layer-2 w2: how much further the end-effector ends from where the benign
    rollout ended, paired by seed. Larger = adversarial pushed eef off-course."""
    b_final = {ep["seed"]: np.asarray(ep["eef_positions"][-1])
               for ep in episodes_b if ep.get("eef_positions")}
    diffs = []
    for ep in episodes_c:
        pos = ep.get("eef_positions")
        if pos and ep["seed"] in b_final:
            diffs.append(float(np.linalg.norm(np.asarray(pos[-1]) - b_final[ep["seed"]])))
    return float(np.mean(diffs)) if diffs else 0.0


def _paired_action_divergence(episodes_c: list[dict], episodes_b: list[dict]) -> float:
    """Layer-2 w3: paired same-seed mean ||a_B(t) - a_C(t)||^2 over the common
    prefix. Same initial state, so divergence is attributable to the texture."""
    b_act = {ep["seed"]: ep.get("action_trace") for ep in episodes_b}
    divs = []
    for ep in episodes_c:
        ac = ep.get("action_trace")
        ab = b_act.get(ep["seed"])
        if not ac or not ab:
            continue
        n = min(len(ac), len(ab))
        if n == 0:
            continue
        a = np.asarray(ac[:n], dtype=np.float64)
        b = np.asarray(ab[:n], dtype=np.float64)
        divs.append(float(np.mean(np.sum((a - b) ** 2, axis=1))))
    return float(np.mean(divs)) if divs else 0.0


def terminal_reward(
    result_c: dict,
    result_b: dict,
    *,
    gate_pass: bool,
    siglip_ok: bool,
    radius_ok: bool,
    w1: float = 0.3,
    w2: float = 0.5,
    w3: float = 0.1,
    max_steps: int = 280,
) -> dict:
    """Compute terminal reward. Hard gates floor the reward.

    result_c/result_b: Pi05Evaluator.eval(detail=True) outputs for adversarial
    (C) and same-prompt benign anchor (B), evaluated on the SAME seeds.
    """
    if not (gate_pass and siglip_ok and radius_ok):
        return {"reward": FLOOR, "floored": True,
                "gate_pass": gate_pass, "siglip_ok": siglip_ok, "radius_ok": radius_ok}

    sr_c = float(result_c["success_rate"])
    sr_b = float(result_b["success_rate"])
    gap = sr_b - sr_c
    eps_c = result_c["episodes"]
    eps_b = result_b["episodes"]

    w1_term = w1 * _mean_steps_ratio(eps_c, max_steps)
    w2_term = w2 * _eef_shortfall(eps_c, eps_b)
    w3_term = w3 * _paired_action_divergence(eps_c, eps_b)
    reward = gap + w1_term + w2_term + w3_term
    return {
        "reward": float(reward),
        "floored": False,
        "sr_c": sr_c, "sr_b": sr_b, "gap": gap,
        "w1_steps": w1_term, "w2_eef": w2_term, "w3_action": w3_term,
    }
