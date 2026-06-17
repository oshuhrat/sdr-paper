"""
SDR-Agent v3 Reward Shaping — main entry point.

Modes:
  sparse           — sparse reward (baseline)
  dense            — dense reward shaping
  dense_curriculum — dense + distance curriculum
  staged           — MLP + staged distractor curriculum (0→1→2)
  lstm             — LSTM + Oracle goal encoder (frozen random projection)
  sdr              — LSTM + ConceptComposer (trainable compositional embeddings)
  cg               — run both lstm + sdr and report Compositional Gap

Usage:
  python -X utf8 main.py --fast --modes lstm
  python -X utf8 main.py --modes lstm
  python -X utf8 main.py --modes cg
  python -X utf8 main.py --fast --modes cg
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import pandas as pd
import numpy as np

_here = Path(__file__).parent
sys.path.insert(0, str(_here))
sys.path.insert(1, str(_here.parent / "sdr_agent_v3_transfer_action"))
sys.path.insert(2, str(_here.parent))

from config import set_seed, DEFAULT_SEED  # noqa: E402


def banner(text: str):
    print(f"\n{'='*64}\n  {text}\n{'='*64}")


_SUCCESS_TRAIN   = 0.80
_SUCCESS_HOLDOUT = 0.70
_NEVER_PICKUP    = 0.10
_GAP_THRESHOLD   = 0.15   # LSTM should close this significantly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",     action="store_true")
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--n-eval",   type=int, default=None)
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    parser.add_argument("--modes",    nargs="+",
                        default=["lstm"],
                        choices=["sparse", "dense", "dense_curriculum",
                                 "staged", "lstm", "sdr", "random", "cg",
                                 "lstm_attn", "sdr_attn", "random_attn", "cg_attn"])
    args = parser.parse_args()
    set_seed(args.seed)

    n_rollouts = args.rollouts or (100 if args.fast else 500)
    n_steps    = 512 if args.fast else 1024
    n_eval     = args.n_eval or (50 if args.fast else 200)

    print(f"[reward_shaping]  modes={args.modes}")
    print(f"  {n_rollouts} rollouts × {n_steps} steps  |  {n_eval} eval eps/combo")

    t0_global = time.perf_counter()
    trained   = {}
    histories = {}
    is_lstm   = {}   # track which policies are LSTM

    # "cg" / "cg_attn" expand to their constituent modes (sequential)
    expanded_modes = []
    for m in args.modes:
        if m == "cg":
            expanded_modes.extend(["lstm", "sdr", "random"])
        elif m == "cg_attn":
            expanded_modes.extend(["lstm_attn", "sdr_attn", "random_attn"])
        else:
            expanded_modes.append(m)
    run_modes = list(dict.fromkeys(expanded_modes))   # deduplicate preserving order

    for mode in run_modes:
        banner(f"Training: {mode.upper()}")

        if mode in ("lstm", "sdr", "random", "lstm_attn", "sdr_attn", "random_attn"):
            from train_lstm  import train_lstm
            from policy_lstm import LSTMPolicy, LSTMPolicySim
            r0 = max(1, n_rollouts // 5)
            r1 = max(1, n_rollouts * 3 // 10)
            r2 = n_rollouts - r0 - r1
            encoder      = None   # defaults to OracleGoalEncoder
            policy_class = None   # defaults to LSTMPolicy
            base = mode.replace("_attn", "")   # "lstm_attn" → "lstm", etc.
            if base == "sdr":
                from concept_encoder import ConceptComposer
                encoder = ConceptComposer()
            elif base == "random":
                from oracle_encoder import RandomGoalEncoder
                encoder = RandomGoalEncoder()
            if mode.endswith("_attn"):
                policy_class = LSTMPolicySim
            policy, hist = train_lstm(
                stages       = [(0, r0), (1, r1), (2, r2)],
                n_steps      = n_steps,
                verbose      = True,
                encoder      = encoder,
                policy_class = policy_class,
                seed         = args.seed,
            )
            is_lstm[mode] = True

        elif mode == "staged":
            from train import train_staged
            r0 = max(1, n_rollouts // 5)
            r1 = max(1, n_rollouts * 3 // 10)
            r2 = n_rollouts - r0 - r1
            policy, hist = train_staged(
                stages  = [(0, r0), (1, r1), (2, r2)],
                n_steps = n_steps,
                verbose = True,
            )
            is_lstm[mode] = False

        else:
            from evaluate import evaluate_all  # import BEFORE train to avoid v3 shadow
            from train    import train
            use_c    = (mode == "dense_curriculum")
            rwd_mode = "sparse" if mode == "sparse" else "dense"
            policy, hist = train(
                reward_mode    = rwd_mode,
                use_curriculum = use_c,
                n_rollouts     = n_rollouts,
                n_steps        = n_steps,
                verbose        = True,
            )
            is_lstm[mode] = False

        torch.save(policy.state_dict(), f"model_{mode}.pt")
        hist.to_csv(f"history_{mode}.csv", index=False)

        recent_sr = float(hist["success_rate"].iloc[-5:].mean()) if len(hist) >= 5 \
                    else float(hist["success_rate"].mean())
        trained[mode]   = (policy, recent_sr)
        histories[mode] = hist

    # ── Evaluation ────────────────────────────────────────────────────────────
    banner("Evaluation: Greedy Policy on ALL 18 combos")

    eval_results = {}
    for mode, (policy, online_sr) in trained.items():
        print(f"  Evaluating [{mode}] ...")
        if is_lstm.get(mode, False):
            from evaluate_lstm import evaluate_lstm
            r = evaluate_lstm(policy, n_eval_episodes=n_eval, online_train_sr=online_sr)
        else:
            from evaluate import evaluate
            r = evaluate(policy, n_eval_episodes=n_eval, online_train_sr=online_sr)
        eval_results[mode] = r
        gap_str = f"  gap={r['greedy_online_gap']:.3f}" if r["greedy_online_gap"] is not None else ""
        ci_str  = f"±{r['holdout_ci95']:.4f}" if "holdout_ci95" in r else ""
        gr_str  = f"  GR={r['generalization_ratio']:.3f}" if r.get("generalization_ratio") else ""
        print(f"    train_SR={r['train_sr']:.4f}  holdout_SR={r['holdout_sr']:.4f}{ci_str}{gap_str}{gr_str}")

    pd.DataFrame([{"mode": m, **{k: v for k, v in r.items() if k != "per_combo"}}
                  for m, r in eval_results.items()]).to_csv("results.csv", index=False)
    print("  Saved results.csv")

    # ── Report ────────────────────────────────────────────────────────────────
    banner("Report")
    elapsed = time.perf_counter() - t0_global

    verdict_any = False
    print(f"\n  {'Mode':<22} {'Online SR':>10} {'Train SR':>10} {'±95CI':>7} "
          f"{'Holdout SR':>12} {'±95CI':>7} {'GR':>6} {'Gap':>7}")
    print(f"  {'-'*85}")
    for mode in run_modes:
        _, online_sr = trained[mode]
        r = eval_results[mode]
        gap_str = f"{r['greedy_online_gap']:.3f}" if r["greedy_online_gap"] is not None else "  —  "
        t_ci    = f"{r.get('train_ci95',   0.0):.4f}"
        h_ci    = f"{r.get('holdout_ci95', 0.0):.4f}"
        gr_str  = f"{r.get('generalization_ratio', 0.0):.3f}" if r.get("generalization_ratio") else "  —  "
        ok_train   = r["train_sr"]   >= _SUCCESS_TRAIN
        ok_holdout = r["holdout_sr"] >= _SUCCESS_HOLDOUT
        ok_gap     = r["greedy_online_gap"] is not None and r["greedy_online_gap"] <= _GAP_THRESHOLD
        mark = "✓" if (ok_train and ok_holdout and ok_gap) else "✗"
        if ok_train and ok_holdout and ok_gap:
            verdict_any = True
        print(f"  {mode:<22} {online_sr:>10.4f} {r['train_sr']:>10.4f} {t_ci:>7} "
              f"{r['holdout_sr']:>12.4f} {h_ci:>7} {gr_str:>6} {gap_str:>7}  {mark}")

    print()
    print(f"  Criteria: train_SR>{_SUCCESS_TRAIN:.0%}  "
          f"holdout_SR>{_SUCCESS_HOLDOUT:.0%}  "
          f"greedy_online_gap<{_GAP_THRESHOLD:.0%}")

    # ── Per-holdout combo breakdown ────────────────────────────────────────────
    cg_modes = [m for m in run_modes if m in eval_results and "per_holdout" in eval_results[m]]
    if cg_modes:
        print(f"\n  ── Per-holdout combo SR ──────────────────────────────────")
        all_holdout_combos = sorted(set().union(*[eval_results[m]["per_holdout"].keys() for m in cg_modes]))
        header = f"  {'Combo':<30}" + "".join(f"{m:>16}" for m in cg_modes)
        print(header)
        print(f"  {'-'*(30 + 16*len(cg_modes))}")
        for combo_str in all_holdout_combos:
            row = f"  {combo_str:<30}"
            for m in cg_modes:
                ph = eval_results[m]["per_holdout"].get(combo_str, {})
                if ph:
                    row += f"  {ph['sr']:.3f}±{ph['ci95']:.3f}"
                else:
                    row += f"{'—':>16}"
            print(row)

    # ── CG summary ────────────────────────────────────────────────────────────
    oracle_key = "lstm_attn" if "lstm_attn" in eval_results else ("lstm" if "lstm" in eval_results else None)
    sdr_key    = "sdr_attn"  if "sdr_attn"  in eval_results else ("sdr"  if "sdr"  in eval_results else None)
    rand_key   = "random_attn" if "random_attn" in eval_results else ("random" if "random" in eval_results else None)

    if oracle_key and sdr_key:
        print(f"\n  ── SDR-L4 Compositional Gap ──────────────────────────────")
        oracle_h   = eval_results[oracle_key]["holdout_sr"]
        oracle_ci  = eval_results[oracle_key].get("holdout_ci95", 0.0)
        oracle_gr  = eval_results[oracle_key].get("generalization_ratio", None)
        sdr_h      = eval_results[sdr_key]["holdout_sr"]
        sdr_ci     = eval_results[sdr_key].get("holdout_ci95", 0.0)
        sdr_gr     = eval_results[sdr_key].get("generalization_ratio", None)
        cg         = oracle_h - sdr_h
        ci_overlap = abs(cg) < (oracle_ci + sdr_ci)

        _fmt = lambda label, h, ci, gr: (
            f"  {label:<22}: {h:.4f} ± {ci:.4f}  GR={gr:.3f}" if gr else
            f"  {label:<22}: {h:.4f} ± {ci:.4f}"
        )
        print(_fmt(f"Oracle [{oracle_key}]", oracle_h, oracle_ci, oracle_gr))
        print(_fmt(f"SDR    [{sdr_key}]",    sdr_h,    sdr_ci,    sdr_gr))
        if rand_key:
            rand_h  = eval_results[rand_key]["holdout_sr"]
            rand_ci = eval_results[rand_key].get("holdout_ci95", 0.0)
            rand_gr = eval_results[rand_key].get("generalization_ratio", None)
            print(_fmt(f"Random [{rand_key}]",  rand_h,   rand_ci,   rand_gr))
        print(f"  CG = Oracle − SDR : {cg:+.4f}")
        print(f"  CI overlap        : {'YES (не значимо)' if ci_overlap else 'NO (значимо)'}")

        if cg <= 0.05 and ci_overlap:
            verdict_cg = "SDR ≈ Oracle (CG ≤ 5%, CI перекрываются) → SDR-L4 ПОДТВЕРЖДЁН"
        elif cg <= 0.05:
            verdict_cg = "SDR ≈ Oracle (CG ≤ 5%, CI не перекрываются) → требует проверки"
        elif cg <= 0.15:
            verdict_cg = "Умеренный gap (CG 5–15%) → SDR частично работает"
        elif cg <= 0.30:
            verdict_cg = "Значимый gap (CG 15–30%) → SDR encoder требует улучшений"
        else:
            verdict_cg = "Большой gap (CG > 30%) → SDR не справляется"
        print(f"  VERDICT: {verdict_cg}")
        print()

    if verdict_any:
        print("  ✓ ORACLE SR достаточен.")
    else:
        best_mode = max(run_modes, key=lambda m: eval_results[m]["holdout_sr"])
        best_sr   = eval_results[best_mode]["holdout_sr"]
        print(f"  ✗ Лучший holdout SR = {best_sr:.4f} [{best_mode}].")
        if best_sr >= 0.50:
            print("  Прогресс. Попробовать: больше роллаутов или adjusting stages.")
        else:
            print("  LSTM тоже не помог — проблема глубже (среда или постановка задачи).")

    print(f"\n  Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
