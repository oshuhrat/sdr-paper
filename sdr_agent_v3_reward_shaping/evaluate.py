"""
Оценка обученного oracle-агента.

Всегда использует SPARSE reward и NO curriculum для честного сравнения.
Оценивается greedy политика (argmax) на всех 18 комбо.

Метрики:
  train_sr   — SR на 15 training комбо
  holdout_sr — SR на 3 holdout комбо (zero-shot)
  greedy_online_gap — |train_sr_greedy - train_sr_online|; хотим < 0.10
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_here = Path(__file__).parent
sys.path.insert(0, str(_here))
sys.path.insert(1, str(_here.parent / "sdr_agent_v3_transfer_action"))

from env import GridWorld, all_combos, HOLDOUT   # noqa: E402


def evaluate(
    policy,
    n_eval_episodes: int = 200,
    seed:            int = 9999,
    online_train_sr: float = None,   # from training log, to compute gap
) -> dict:
    """
    Greedy evaluation on train + holdout combos.
    Env uses sparse reward (no shaping) and no curriculum.
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
                success = False
                while not env.done:
                    obs_t  = torch.from_numpy(obs).unsqueeze(0)
                    goal_t = policy.encode_goal([combo[0]], [combo[1]], [combo[2]])
                    logits, _ = policy(obs_t, goal_t)
                    action = int(logits.argmax(dim=-1).item())
                    obs, _, _, info = env.step(action)
                    if info["success"]:
                        success = True
                rows.append({
                    "combo":      f"{combo[0]}_{combo[1]}_{combo[2]}",
                    "is_holdout": is_h,
                    "success":    success,
                })

    df         = pd.DataFrame(rows)
    train_sr   = float(df[~df.is_holdout]["success"].mean())
    holdout_sr = float(df[ df.is_holdout]["success"].mean())
    per_combo  = df.groupby("combo")["success"].mean().to_dict()

    gap = abs(train_sr - online_train_sr) if online_train_sr is not None else None

    return {
        "train_sr":           train_sr,
        "holdout_sr":         holdout_sr,
        "greedy_online_gap":  gap,
        "per_combo":          per_combo,
    }


def evaluate_all(policies: dict, n_eval: int = 200, verbose: bool = True) -> dict:
    results = {}
    for label, (policy, online_sr) in policies.items():
        if verbose:
            print(f"  Evaluating [{label}] ...")
        r = evaluate(policy, n_eval_episodes=n_eval, online_train_sr=online_sr)
        results[label] = r
        if verbose:
            gap_str = f"  gap={r['greedy_online_gap']:.3f}" if r["greedy_online_gap"] is not None else ""
            print(f"    train_SR={r['train_sr']:.4f}  "
                  f"holdout_SR={r['holdout_sr']:.4f}{gap_str}")
    return results
