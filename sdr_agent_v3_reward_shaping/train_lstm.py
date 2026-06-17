"""
Train LSTM Oracle agent.
Staged distractor curriculum: 0 → 1 → 2 distractors.
Uses LSTMPolicy + LSTMPPOTrainer from local files.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).parent
sys.path.insert(0, str(_here))
sys.path.insert(1, str(_here.parent / "sdr_agent_v3_transfer_action"))

from env            import GridWorld, all_combos           # noqa: E402
from oracle_encoder import OracleGoalEncoder               # noqa: E402
from policy_lstm    import LSTMPolicy, LSTMPolicySim       # noqa: E402
from ppo_lstm       import LSTMPPOTrainer                  # noqa: E402


def train_lstm(
    stages:     list  = None,   # [(n_distractors, n_rollouts), ...]
    n_steps:    int   = 1024,
    lr:         float = 3e-4,
    ent_coef:   float = 0.001,
    alpha:      float = 0.10,
    beta:       float = 1.00,
    gamma_env:  float = 0.005,
    seed:         int   = 42,
    verbose:      bool  = True,
    encoder       = None,   # if None → OracleGoalEncoder; else use provided encoder
    policy_class  = None,   # if None → LSTMPolicy; pass LSTMPolicySim for binding test
) -> tuple:
    """
    Returns (lstm_policy, history_df).
    Default stages: 0-distractor bootstrap (20%), 1-distractor (30%), 2-distractor (50%).
    If encoder is provided, it is used as goal_encoder (e.g. ConceptComposer for SDR test).
    """
    if stages is None:
        stages = None   # resolved below based on total rollouts

    allowed      = all_combos(holdout=True)
    goal_encoder = encoder       if encoder      is not None else OracleGoalEncoder()
    PolicyClass  = policy_class  if policy_class is not None else LSTMPolicy
    policy       = PolicyClass(goal_encoder)
    trainer = LSTMPPOTrainer(
        policy, lr=lr, clip_eps=0.2, ent_coef=ent_coef,
        gamma=0.99, n_steps=n_steps,
    )

    enc_name  = getattr(goal_encoder, "name", type(goal_encoder).__name__)
    pol_name  = "lstm_sim" if PolicyClass is LSTMPolicySim else "lstm"
    n_par     = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if verbose:
        print(f"  [{pol_name}/{enc_name}]  params={n_par:,}  stages={stages}  "
              f"n_steps={n_steps}  ent_coef={ent_coef}")

    history    = []
    accum      = []
    t0         = time.perf_counter()
    global_idx = 0

    for stage_n, (n_dist, n_roll) in enumerate(stages):
        env = GridWorld(
            allowed, n_distractors=n_dist, seed=seed + stage_n,
            reward_mode="dense", alpha=alpha, beta=beta, gamma=gamma_env,
        )
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
                    "label": f"{pol_name}/{enc_name}", "rollout": global_idx,
                    "total_steps": global_idx * n_steps,
                    "stage": stage_n, "n_distractors": n_dist,
                    "success_rate": sr, "mean_ep_len": elen,
                    "curriculum_phase": stage_n, "curriculum_dist": None,
                    **losses,
                })
                if verbose:
                    print(f"  [lstm s{stage_n} d={n_dist}] "
                          f"rollout {local_idx:>3}/{n_roll}  SR={sr:.3f}  len={elen:.1f}  "
                          f"pg={losses['pg_loss']:.3f}  [{time.perf_counter()-t0:.0f}s]")

    return policy, pd.DataFrame(history)
