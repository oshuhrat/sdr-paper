"""
SDR Structural Retention Test (v2)  —  entry point.

Usage:
    python -X utf8 main.py           # full run  (~15-30 min)
    python -X utf8 main.py --fast    # smoke-test (50 instances, <30 sec)
    python -X utf8 main.py --jobs N  # set CPU count
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import set_seed, DEFAULT_SEED  # noqa: E402

from experiment import (
    run_main_experiment,
    run_mixed_experiment,
    run_translation_test,
)
from analysis import (
    statistical_tests,
    survival_vs_metrics,
    plot_metric_profiles,
    plot_tau200_heatmap,
    plot_glider_vs_nmi_comparison,
    plot_translation_test,
    plot_survival_correlation,
    plot_invariant_drift,
    plot_sdr_score_v2,
    plot_field_evolution,
    plot_class_separation,
)
from report import generate_report


def banner(text: str):
    print(f"\n{'='*62}")
    print(f"  {text}")
    print(f"{'='*62}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    set_seed(args.seed)

    if args.fast:
        import experiment as _e
        _e.N_INSTANCES = 50
        _e.N_MIXED     = 50
        print("[fast mode] 50 instances per type")

    Path("plots").mkdir(exist_ok=True)
    t0 = time.perf_counter()

    banner("SDR Structural Retention Test  v2")
    print("Hypothesis SDR-L1: stable forms preserve structural invariants.")
    print("Key fix: shift-invariant MI via FFT — Glider should score HIGH.")

    # ── Experiment 1 ──────────────────────────────────────────────────────────
    banner("Experiment 1/3  —  Main")
    df = run_main_experiment(n_jobs=args.jobs)
    Path("results.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"  Saved results.csv  ({len(df):,} rows)")

    # ── Experiment 2 ──────────────────────────────────────────────────────────
    banner("Experiment 2/3  —  Mixed (noise + structure)")
    df_mixed = run_mixed_experiment(n_jobs=args.jobs)
    Path("results_mixed.csv").write_text(df_mixed.to_csv(index=False), encoding="utf-8")
    print(f"  Saved results_mixed.csv  ({len(df_mixed):,} rows)")

    # ── Experiment 3 ──────────────────────────────────────────────────────────
    banner("Experiment 3/3  —  Glider Translation Test")
    import experiment as _e
    n_trans = 50 if args.fast else 200
    df_trans = run_translation_test(n_trials=n_trans)
    Path("results_translation.csv").write_text(
        df_trans.to_csv(index=False), encoding="utf-8"
    )
    print(f"  Saved results_translation.csv  ({len(df_trans):,} rows)")

    # ── Analysis ──────────────────────────────────────────────────────────────
    banner("Statistical Analysis")
    stats    = statistical_tests(df)
    corr_df, agg_surv = survival_vs_metrics(df_mixed)
    Path("stats.csv").write_text(stats.to_csv(index=False), encoding="utf-8")

    sig = stats.significant.sum()
    print(f"  Significant (p<0.01): {sig}/{len(stats)}")
    best_m = corr_df.loc[corr_df.spearman_rho.abs().idxmax()]
    print(f"  Best survival predictor: {best_m['metric']}  ρ={best_m['spearman_rho']:.3f}")

    # quick console preview at tau=200
    d200 = df[df.tau == 200]
    print(f"\n  SR_shift at tau=200:")
    for obj in ["random", "block", "glider", "lwss", "pulsar"]:
        sub = d200[d200.object_type == obj]
        if len(sub):
            print(f"    {obj:<10} NMI_old={sub['nmi_old'].mean():.4f}  "
                  f"SR_shift={sub['sr_shift'].mean():.4f}  "
                  f"SR_shape={sub['sr_shape'].mean():.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    banner("Generating Plots")
    plot_field_evolution();              print("  plots/field_evolution.png")
    plot_metric_profiles(df);           print("  plots/metric_profiles.png")
    plot_tau200_heatmap(df);            print("  plots/tau200_heatmap.png")
    plot_glider_vs_nmi_comparison(df);  print("  plots/glider_nmi_comparison.png")
    plot_translation_test(df_trans);    print("  plots/translation_test.png")
    plot_invariant_drift(df);           print("  plots/invariant_drift.png")
    plot_class_separation(df);          print("  plots/class_separation.png")
    plot_survival_correlation(agg_surv, corr_df)
    print("  plots/survival_correlation.png")
    sdr_combined = plot_sdr_score_v2(df, df_mixed)
    print("  plots/sdr_score_v2.png")

    # ── Report ────────────────────────────────────────────────────────────────
    banner("Generating Report")
    report_text = generate_report(df, df_mixed, df_trans, stats, corr_df, sdr_combined)
    Path("report.md").write_text(report_text, encoding="utf-8")
    print("  Saved report.md")

    elapsed = time.perf_counter() - t0
    banner(f"Done in {elapsed:.1f}s")
    print("  results.csv             – main metrics (long format)")
    print("  results_mixed.csv       – noise+structure survival")
    print("  results_translation.csv – Glider Translation Test")
    print("  stats.csv               – statistical tests")
    print("  plots/                  – 9 figures")
    print("  report.md               – auto-generated conclusions")


if __name__ == "__main__":
    main()
