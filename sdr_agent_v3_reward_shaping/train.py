"""
Обучение Oracle-агента с тремя режимами:
  "sparse"           — стандартная sparse reward (как в v3)
  "dense"            — dense reward shaping без curriculum
  "dense_curriculum" — dense reward + curriculum (4 фазы по расстоянию)

Возвращает (policy, history_df).
history_df содержит: rollout, total_steps, success_rate, mean_ep_len,
                      pg_loss, vf_loss, ent_loss, curriculum_phase, curriculum_dist.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).parent
sys.path.insert(0, str(_here))
sys.path.insert(1, str(_here.parent / "sdr_agent_v3_transfer_action"))

from env            import GridWorld, all_combos   # noqa: E402
from oracle_encoder import OracleGoalEncoder        # noqa: E402
from curriculum     import CurriculumScheduler      # noqa: E402
from policy         import Policy                   # noqa: E402
from ppo            import PPOTrainer               # noqa: E402


def train_staged(
    stages:    list  = None,   # [(n_distractors, n_rollouts), ...] default: [(0,100),(1,150),(2,250)]
    n_steps:   int   = 1024,
    lr:        float = 3e-4,
    ent_coef:  float = 0.001,
    alpha:     float = 0.10,
    beta:      float = 1.00,
    gamma:     float = 0.005,
    seed:      int   = 42,
    verbose:   bool  = True,
) -> tuple:
    """
    Staged distractor curriculum: train on 0 distractors first (pickup only),
    then 1, then 2 (full task). Reuses the same policy across stages.
    """
    if stages is None:
        stages = [(0, 100), (1, 150), (2, 250)]   # default: 500 rollouts total

    allowed = all_combos(holdout=True)
    oracle  = OracleGoalEncoder()
    policy  = Policy(oracle)
    trainer = PPOTrainer(policy, lr=lr, clip_eps=0.2, ent_coef=ent_coef,
                         gamma=0.99, n_steps=n_steps)

    history = []
    accum   = []
    t0      = time.perf_counter()
    total_rollouts = sum(r for _, r in stages)
    n_par   = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    if verbose:
        print(f"  [staged]  params={n_par:,}  stages={stages}  "
              f"n_steps={n_steps}  ent_coef={ent_coef}")

    global_idx = 0
    for stage_n, (n_dist, n_roll) in enumerate(stages):
        env = GridWorld(allowed, n_distractors=n_dist, seed=seed + stage_n,
                        reward_mode="dense", alpha=alpha, beta=beta, gamma=gamma)
        obs, info = env.reset()
        goal_attrs = info["goal_combo"]

        if verbose:
            print(f"  >>> Stage {stage_n}: n_distractors={n_dist}, rollouts={n_roll}")

        for local_idx in range(1, n_roll + 1):
            global_idx += 1
            obs, goal_attrs, ep_stats = trainer.collect_rollout(env, obs, goal_attrs)
            accum.extend(ep_stats)
            losses = trainer.update()

            if local_idx % 10 == 0 or local_idx == n_roll:
                recent = accum[-50:] if accum else []
                sr   = float(np.mean([e["success"] for e in recent])) if recent else 0.0
                elen = float(np.mean([e["ep_len"]  for e in recent])) if recent else 0.0
                history.append({
                    "label": "staged", "rollout": global_idx,
                    "total_steps": global_idx * n_steps, "stage": stage_n,
                    "n_distractors": n_dist, "success_rate": sr, "mean_ep_len": elen,
                    "curriculum_phase": stage_n, "curriculum_dist": None, **losses,
                })
                if verbose:
                    print(f"  [staged s{stage_n} d={n_dist}] "
                          f"rollout {local_idx:>3}/{n_roll}  SR={sr:.3f}  len={elen:.1f}  "
                          f"pg={losses['pg_loss']:.3f}  [{time.perf_counter()-t0:.0f}s]")

    return policy, pd.DataFrame(history)


def train(
    reward_mode:    str   = "dense",   # "sparse" | "dense"
    use_curriculum: bool  = False,
    n_rollouts:     int   = 500,
    n_steps:        int   = 1024,
    lr:             float = 3e-4,
    ent_coef:       float = 0.001,     # minimal entropy to prevent collapse; penalize wrong pickup instead
    alpha:          float = 0.10,
    beta:           float = 1.00,
    gamma:          float = 0.005,
    seed:           int   = 42,
    verbose:        bool  = True,
) -> tuple:
    allowed = all_combos(holdout=True)
    env = GridWorld(
        allowed, n_distractors=2, seed=seed,
        reward_mode=reward_mode,
        alpha=alpha, beta=beta, gamma=gamma,
    )

    oracle  = OracleGoalEncoder()
    policy  = Policy(oracle)
    trainer = PPOTrainer(
        policy, lr=lr, clip_eps=0.2, ent_coef=ent_coef,
        gamma=0.99, n_steps=n_steps,
    )

    curric = None
    if use_curriculum:
        curric = CurriculumScheduler(env, advance_threshold=0.80, window=5)

    obs, info = env.reset()
    goal_attrs = info["goal_combo"]

    history, accum = [], []
    t0     = time.perf_counter()
    label  = reward_mode + ("_curric" if use_curriculum else "")
    n_par  = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    if verbose:
        print(f"  [{label}]  params={n_par:,}  "
              f"rollouts={n_rollouts}  n_steps={n_steps}  ent_coef={ent_coef}")

    for idx in range(1, n_rollouts + 1):
        obs, goal_attrs, ep_stats = trainer.collect_rollout(env, obs, goal_attrs)
        accum.extend(ep_stats)
        losses = trainer.update()

        if idx % 10 == 0 or idx == n_rollouts:
            recent = accum[-50:] if accum else []
            sr     = float(np.mean([e["success"] for e in recent])) if recent else 0.0
            elen   = float(np.mean([e["ep_len"]  for e in recent])) if recent else 0.0
            c_phase = curric.phase    if curric else -1
            c_dist  = curric.max_dist if curric else None

            history.append({
                "label":            label,
                "rollout":          idx,
                "total_steps":      idx * n_steps,
                "success_rate":     sr,
                "mean_ep_len":      elen,
                "curriculum_phase": c_phase,
                "curriculum_dist":  str(c_dist),
                **losses,
            })

            if verbose:
                c_str = f"  [{curric.status()}]" if curric else ""
                print(f"  [{label}] rollout {idx:>4}/{n_rollouts}  "
                      f"SR={sr:.3f}  len={elen:.1f}  "
                      f"pg={losses['pg_loss']:.3f}"
                      f"{c_str}  [{time.perf_counter()-t0:.0f}s]")

            if curric:
                advanced = curric.update(sr)
                if advanced and verbose:
                    print(f"  >>> CURRICULUM ADVANCE → {curric.status()}")

    return policy, pd.DataFrame(history)
