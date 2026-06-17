"""
SDR-Agent v2 — Concept Discovery  entry point.

Hypothesis SDR-L3:
  'New concepts can be constructed as compositions of already stable concepts
   without additional training.'

Test:
  Withhold 3 combos (red_large_circle, green_large_square, blue_large_triangle).
  Train on remaining 15. Compose embeddings. Measure CCS.

Usage:
  python -X utf8 main.py              # full run (~25-35 min CPU)
  python -X utf8 main.py --fast       # smoke-test (5 epochs, 100/combo, ~3 min)
  python -X utf8 main.py --no-tsne    # skip t-SNE
  python -X utf8 main.py --epochs 30  # override epoch count
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import set_seed, DEFAULT_SEED  # noqa: E402


def banner(text: str):
    print(f"\n{'='*64}")
    print(f"  {text}")
    print(f"{'='*64}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",    action="store_true", help="5 epochs, 100/combo")
    parser.add_argument("--no-tsne", action="store_true", help="skip t-SNE (slow)")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--embed",   type=int,   default=128)
    parser.add_argument("--batch",   type=int,   default=128)
    parser.add_argument("--seed",    type=int,   default=DEFAULT_SEED)
    args = parser.parse_args()
    set_seed(args.seed)

    n_per_combo = 100  if args.fast else 300
    n_epochs    = (args.epochs or 5) if args.fast else (args.epochs or 30)
    skip_tsne   = args.no_tsne or args.fast
    device      = "cpu"

    if args.fast:
        print(f"[fast mode] {n_epochs} epochs, {n_per_combo} images/combo, t-SNE skipped")

    Path("plots").mkdir(exist_ok=True)
    t0 = time.perf_counter()

    # ── imports ────────────────────────────────────────────────────────────────
    from train         import train
    from dataset       import ShapeDataset
    from evaluate      import (build_concept_graph, fit_learned_linear,
                               evaluate_ccs, concept_separation)
    from composition   import WeightedMeanComposer, MeanComposer
    from analysis      import (plot_training_curve, plot_tsne_embeddings,
                               plot_concept_graph, plot_ccs_distribution,
                               plot_composition_comparison, plot_zero_shot_examples)
    from report        import generate_report
    import pandas as pd

    # ── Step 1: Train ──────────────────────────────────────────────────────────
    banner("Step 1/5  —  Contrastive Training (15 combos, 3 withheld)")
    print(f"  Holdout (zero-shot test): red_large_circle, "
          f"green_large_square, blue_large_triangle")
    model, history = train(
        n_per_combo  = n_per_combo,
        n_epochs     = n_epochs,
        batch_size   = args.batch,
        embed_dim    = args.embed,
        device       = device,
        verbose      = True,
    )
    torch.save(model.state_dict(), "model.pt")
    history.to_csv("training_history.csv", index=False)
    print(f"  Saved model.pt + training_history.csv")

    # ── Step 2: Build concept graph ────────────────────────────────────────────
    banner("Step 2/5  —  Building Concept Graph")
    train_ds = ShapeDataset(n_per_combo=n_per_combo, holdout=True, seed=0)
    print(f"  Extracting embeddings for {len(train_ds):,} training images …")
    graph, Z_train, colors, sizes, shapes, combos = build_concept_graph(
        model, train_ds, device=device,
    )
    print(graph.summary())
    sep = concept_separation(graph)
    print(f"\n  Concept separation (inter-concept distance):")
    for attr, s in sep.items():
        print(f"    {attr:<8} {s:.4f}")

    # ── Step 3: Composition ────────────────────────────────────────────────────
    banner("Step 3/5  —  Composition & CCS Evaluation")

    # Fit learned linear on training combos
    ll_composer = fit_learned_linear(graph, Z_train, colors, sizes, shapes, combos)
    print(f"  Fitted LearnedLinearComposer on {ll_composer.n_train} combos")

    ccs_data = evaluate_ccs(
        model, graph, ll_composer,
        device=device,
        n_per_holdout=min(n_per_combo, 100),
    )
    ccs_rows = ccs_data["rows"]

    # Print CCS summary
    import numpy as np
    df_ccs = pd.DataFrame(ccs_rows)
    print()
    for method in ["mean", "weighted_mean", "learned_linear", "random_baseline"]:
        sub = df_ccs[df_ccs.method == method]["ccs"]
        if len(sub):
            print(f"  {method:<20}  mean CCS = {sub.mean():.4f} ± {sub.std():.4f}")

    # ── Step 4: Plots ──────────────────────────────────────────────────────────
    banner("Step 4/5  —  Generating Plots")

    plot_training_curve(history)
    print("  plots/training_curve.png")

    if not skip_tsne:
        plot_tsne_embeddings(Z_train, colors, sizes, shapes, combos)
        print("  plots/tsne_embeddings.png")
    else:
        print("  [skipped] tsne_embeddings.png")

    plot_concept_graph(graph)
    print("  plots/concept_graph.png")

    plot_ccs_distribution(ccs_rows)
    print("  plots/ccs_distribution.png")

    plot_composition_comparison(ccs_rows)
    print("  plots/composition_comparison.png")

    # Build composed z for each holdout combo (using best method)
    composed_per_combo = {}
    for combo in sorted(set(r["combo"] for r in ccs_rows)):
        color, size, shape = combo.split("_")
        composed_per_combo[combo] = graph.find_composition(
            color, size, shape, method="mean",
        )

    holdout_ds = ShapeDataset(n_per_combo=100, holdout=False, seed=99999)
    plot_zero_shot_examples(
        holdout_ds,
        ccs_data["Z_holdout"], ccs_data["combos_holdout"],
        Z_train, combos,
        composed_per_combo,
    )
    print("  plots/zero_shot_examples.png")

    # ── Step 5: Report ─────────────────────────────────────────────────────────
    banner("Step 5/5  —  Report")
    report = generate_report(history, ccs_rows, graph, sep, fast_mode=args.fast)
    Path("report.md").write_text(report, encoding="utf-8")
    print("  Saved report.md")

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    banner(f"Done in {elapsed:.1f}s")
    df_ccs2 = pd.DataFrame(ccs_rows)
    for method in ["mean", "weighted_mean", "learned_linear"]:
        sub = df_ccs2[df_ccs2.method == method]["ccs"]
        if len(sub):
            tag = (" ✓ >0.90" if sub.mean() > 0.90
                   else " ✓ >0.75" if sub.mean() > 0.75
                   else " ✗ <0.75")
            print(f"  CCS ({method:<16}): {sub.mean():.4f}{tag}")
    rand_sub = df_ccs2[df_ccs2.method == "random_baseline"]["ccs"]
    print(f"  CCS (random baseline   ): {rand_sub.mean():.4f}")
    print()
    print("  Files:")
    print("    model.pt                    – encoder weights")
    print("    training_history.csv        – loss/SR per epoch")
    print("    plots/training_curve.png")
    print("    plots/concept_graph.png")
    print("    plots/ccs_distribution.png")
    print("    plots/composition_comparison.png")
    print("    plots/zero_shot_examples.png")
    if not skip_tsne:
        print("    plots/tsne_embeddings.png")
    print("    report.md                   – SDR-L3 verdict")


if __name__ == "__main__":
    main()
