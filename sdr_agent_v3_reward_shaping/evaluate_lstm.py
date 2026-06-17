"""
Greedy evaluation for LSTM policy.
Maintains (h, c) state across steps within each episode.
Resets at episode start.

Returns rich statistics: mean SR, std, 95% CI (Wald), per-holdout breakdown, GR.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_here = Path(__file__).parent
sys.path.insert(0, str(_here))
sys.path.insert(1, str(_here.parent / "sdr_agent_v3_transfer_action"))

from env import GridWorld, all_combos, HOLDOUT   # noqa: E402


def _ci95(p: float, n: int) -> float:
    """Wald 95% CI half-width for a Bernoulli proportion."""
    if n == 0:
        return 0.0
    return 1.96 * math.sqrt(p * (1.0 - p) / n)


def evaluate_lstm(policy, n_eval_episodes: int = 200, seed: int = 9999,
                  online_train_sr: float = None) -> dict:
    """Greedy evaluation of LSTM policy on all 18 combos.

    Returns:
        train_sr, train_std, train_ci95,
        holdout_sr, holdout_std, holdout_ci95,
        greedy_online_gap,
        generalization_ratio (holdout_sr / train_sr),
        per_combo          {combo_str: sr},
        per_holdout        {combo_str: {sr, std, ci95, n}},
    """
    train_combos   = all_combos(holdout=True)
    holdout_combos = all_combos(holdout=False)
    all_c          = train_combos + holdout_combos

    env = GridWorld(all_c, n_distractors=2, seed=seed, reward_mode="sparse")

    policy.eval()
    rows = []
    with torch.no_grad():
        for combo in all_c:
            is_h = combo in HOLDOUT
            for _ in range(n_eval_episodes):
                obs, _ = env.reset(goal_combo=combo)
                h, c   = policy.init_hidden()
                success = False
                while not env.done:
                    obs_t  = torch.from_numpy(obs).unsqueeze(0)
                    goal_t = policy.encode_goal([combo[0]], [combo[1]], [combo[2]])
                    action, _, _, h, c = policy.act(obs_t, goal_t, h, c)
                    obs, _, _, info = env.step(int(action.item()))
                    if info["success"]:
                        success = True
                rows.append({
                    "combo":      f"{combo[0]}_{combo[1]}_{combo[2]}",
                    "is_holdout": is_h,
                    "success":    int(success),
                })

    df         = pd.DataFrame(rows)
    train_df   = df[~df.is_holdout]
    holdout_df = df[ df.is_holdout]

    # ── Aggregate stats ────────────────────────────────────────────────────────
    train_sr   = float(train_df["success"].mean())
    holdout_sr = float(holdout_df["success"].mean())

    # std of the sample mean = sqrt(p*(1-p)/n)
    n_train   = len(train_df)
    n_holdout = len(holdout_df)
    train_std   = math.sqrt(train_sr   * (1 - train_sr)   / n_train)   if n_train   else 0.0
    holdout_std = math.sqrt(holdout_sr * (1 - holdout_sr) / n_holdout) if n_holdout else 0.0
    train_ci95   = 1.96 * train_std
    holdout_ci95 = 1.96 * holdout_std

    # ── Per-combo ──────────────────────────────────────────────────────────────
    per_combo = df.groupby("combo")["success"].mean().to_dict()

    # ── Per-holdout detailed breakdown ────────────────────────────────────────
    per_holdout = {}
    for combo_str, grp in holdout_df.groupby("combo"):
        p = float(grp["success"].mean())
        n = len(grp)
        per_holdout[combo_str] = {
            "sr":    p,
            "std":   math.sqrt(p * (1 - p) / n) if n else 0.0,
            "ci95":  _ci95(p, n),
            "n":     n,
        }

    # ── Greedy/online gap ──────────────────────────────────────────────────────
    gap = abs(train_sr - online_train_sr) if online_train_sr is not None else None

    # ── Generalization ratio ──────────────────────────────────────────────────
    gr = holdout_sr / train_sr if train_sr > 0 else None

    return {
        "train_sr":          train_sr,
        "train_std":         train_std,
        "train_ci95":        train_ci95,
        "holdout_sr":        holdout_sr,
        "holdout_std":       holdout_std,
        "holdout_ci95":      holdout_ci95,
        "greedy_online_gap": gap,
        "generalization_ratio": gr,
        "per_combo":         per_combo,
        "per_holdout":       per_holdout,
    }


def evaluate_all_lstm(policies: dict, n_eval: int = 200, verbose: bool = True) -> dict:
    results = {}
    for label, (policy, online_sr) in policies.items():
        if verbose:
            print(f"  Evaluating [{label}] ...")
        r = evaluate_lstm(policy, n_eval_episodes=n_eval, online_train_sr=online_sr)
        results[label] = r
        if verbose:
            gap_str = f"  gap={r['greedy_online_gap']:.3f}" if r["greedy_online_gap"] is not None else ""
            print(f"    train_SR={r['train_sr']:.4f}±{r['train_ci95']:.4f}  "
                  f"holdout_SR={r['holdout_sr']:.4f}±{r['holdout_ci95']:.4f}{gap_str}  "
                  f"GR={r['generalization_ratio']:.3f}")
    return results
