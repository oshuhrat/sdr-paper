"""
Visualisations for SDR-Agent v2.

Plots:
  1. tsne_embeddings.png      — 3 t-SNE panels: colour by shape / color / size
  2. concept_graph.png        — concept prototype positions in 2D (PCA)
  3. ccs_distribution.png     — CCS histograms: mean vs weighted vs linear vs random
  4. zero_shot_examples.png   — example images from holdout + their nearest neighbours
  5. composition_comparison.png — bar chart of CCS per method per holdout combo
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

PLOTS = Path("plots")

# Consistent palettes
COLOR_MAP = {"red": "#e05c5c", "green": "#5cb85c", "blue": "#5c7ae0"}
SHAPE_MAP = {"circle": "o",   "square": "s",       "triangle": "^"}
SIZE_MAP  = {"small":  "#aaaaaa", "large": "#333333"}


def plot_training_curve(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["loss"], lw=2, color="steelblue")
    axes[0].set(xlabel="Epoch", ylabel="NT-Xent Loss", title="Training Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["pos_sr"], lw=2, color="green",  label="Positive SR")
    axes[1].plot(history["epoch"], history["neg_sr"], lw=2, color="tomato", label="Negative SR", ls="--")
    axes[1].set(xlabel="Epoch", ylabel="Cosine Similarity", title="SR+ vs SR−")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "training_curve.png", dpi=150)
    plt.close(fig)


def plot_tsne_embeddings(
    Z:      np.ndarray,
    colors: List[str],
    sizes:  List[str],
    shapes: List[str],
    combos: List[str],
    n_max:  int = 3000,
):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Z), size=min(n_max, len(Z)), replace=False)
    Z_s = Z[idx]
    cols_s  = [colors[i] for i in idx]
    szs_s   = [sizes[i]  for i in idx]
    shs_s   = [shapes[i] for i in idx]
    cbs_s   = [combos[i] for i in idx]

    print("    t-SNE …")
    emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(Z_s)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: colour by shape
    for sh in ["circle", "square", "triangle"]:
        mask = np.array([s == sh for s in shs_s])
        axes[0].scatter(emb[mask, 0], emb[mask, 1],
                        marker=SHAPE_MAP[sh], s=20, alpha=0.6, label=sh)
    axes[0].set_title("By shape"); axes[0].legend(markerscale=2)

    # Panel 2: colour by color
    for col in ["red", "green", "blue"]:
        mask = np.array([c == col for c in cols_s])
        axes[1].scatter(emb[mask, 0], emb[mask, 1],
                        c=COLOR_MAP[col], s=20, alpha=0.6, label=col)
    axes[1].set_title("By color"); axes[1].legend(markerscale=2)

    # Panel 3: colour by size
    for sz in ["small", "large"]:
        mask = np.array([s == sz for s in szs_s])
        axes[2].scatter(emb[mask, 0], emb[mask, 1],
                        c=SIZE_MAP[sz], s=20, alpha=0.6, label=sz)
    axes[2].set_title("By size"); axes[2].legend(markerscale=2)

    for ax in axes:
        ax.axis("off")
    fig.suptitle("t-SNE of Training Embeddings", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "tsne_embeddings.png", dpi=150)
    plt.close(fig)
    return emb


def plot_concept_graph(graph):
    """PCA-2D projection of all concept prototypes."""
    from concept_graph import ConceptNode
    nodes   = graph._node_list
    embeds  = np.stack([n.embedding for n in nodes])

    pca = PCA(n_components=2).fit(embeds)
    pts = pca.transform(embeds)

    fig, ax = plt.subplots(figsize=(9, 7))
    attr_colors = {"color": "#e88", "size": "#8b8", "shape": "#88e"}

    for node, (x, y) in zip(nodes, pts):
        c = attr_colors[node.attribute]
        ax.scatter(x, y, s=300, color=c, edgecolors="black", lw=1.5, zorder=3)
        ax.text(x + 0.01, y + 0.01, node.label, fontsize=10, fontweight="bold", zorder=4)

    # Draw neighbor links
    for node in nodes:
        x0, y0 = pts[node.id]
        for nid in node.neighbors:
            if nid < len(pts):
                x1, y1 = pts[nid]
                ax.plot([x0, x1], [y0, y1], "k-", alpha=0.2, lw=0.8, zorder=1)

    patches = [mpatches.Patch(color=c, label=a)
               for a, c in attr_colors.items()]
    ax.legend(handles=patches, title="Attribute")
    ax.set_title("ConceptGraph: PCA-2D projection of concept prototypes", fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(PLOTS / "concept_graph.png", dpi=150)
    plt.close(fig)


def plot_ccs_distribution(rows: list):
    import pandas as pd
    df = pd.DataFrame(rows)[["method", "ccs"]]

    methods = ["mean", "weighted_mean", "learned_linear", "random_baseline"]
    method_labels = {
        "mean":             "Mean",
        "weighted_mean":    "Weighted Mean",
        "learned_linear":   "Learned Linear",
        "random_baseline":  "Random Baseline",
    }
    colors = ["steelblue", "mediumseagreen", "darkorange", "tomato"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for method, col in zip(methods, colors):
        vals = df[df.method == method]["ccs"].values
        if len(vals):
            ax.hist(vals, bins=15, label=f"{method_labels.get(method, method)} (μ={vals.mean():.3f})",
                    color=col, alpha=0.7, density=True)

    ax.axvline(0.75, color="black", ls="--", lw=1.5, label="0.75 threshold")
    ax.axvline(0.90, color="black", ls=":",  lw=1.5, label="0.90 threshold")
    ax.set(xlabel="CCS (Concept Composition Score)", ylabel="Density",
           title="CCS Distribution: SDR composition vs random baseline")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "ccs_distribution.png", dpi=150)
    plt.close(fig)


def plot_composition_comparison(rows: list):
    """Bar chart of CCS per method per holdout combo."""
    import pandas as pd
    df = pd.DataFrame(rows)
    sdr_df = df[df.method != "random_baseline"]

    combos  = sorted(sdr_df["combo"].unique())
    methods = ["mean", "weighted_mean", "learned_linear"]
    method_labels = {"mean": "Mean", "weighted_mean": "Weighted", "learned_linear": "Learned"}
    colors  = ["steelblue", "mediumseagreen", "darkorange"]

    x  = np.arange(len(combos))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (method, col) in enumerate(zip(methods, colors)):
        vals = [sdr_df[(sdr_df.combo == c) & (sdr_df.method == method)]["ccs"].mean()
                for c in combos]
        bars = ax.bar(x + i * w, vals, w, label=method_labels[method],
                      color=col, edgecolor="black", lw=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + w/2, bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", fontsize=8)

    # Random baseline mean
    rand_mean = df[df.method == "random_baseline"]["ccs"].mean()
    ax.axhline(rand_mean, color="tomato", ls="--", lw=1.5,
               label=f"Random baseline μ={rand_mean:.3f}")
    ax.axhline(0.75, color="gray", ls=":", lw=1.5, label="0.75 target")

    ax.set_xticks(x + w)
    ax.set_xticklabels([c.replace("_", "\n") for c in combos])
    ax.set(ylabel="CCS", title="CCS per Holdout Combo × Composition Method",
           ylim=(0, 1.05))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "composition_comparison.png", dpi=150)
    plt.close(fig)


def plot_zero_shot_examples(
    holdout_ds,
    Z_holdout:  np.ndarray,
    cbs_holdout: list,
    Z_train:    np.ndarray,
    cbs_train:  list,
    composed_z_per_combo: dict,    # {combo_str: z_composed_mean}
    n_examples: int = 4,
):
    """
    For each holdout combo:
      Top row:    real holdout images
      Arrow:      similarity to composed z_pred
      Bottom text: top-5 nearest training combos to composed z
    """
    import torchvision.transforms.functional as TF

    holdout_combos = sorted(set(cbs_holdout))
    n_rows = len(holdout_combos)

    fig, axes = plt.subplots(n_rows, n_examples + 1, figsize=((n_examples + 1) * 1.8, n_rows * 2.2))
    if n_rows == 1:
        axes = [axes]

    Z_tr_n = Z_train / (np.linalg.norm(Z_train, axis=1, keepdims=True) + 1e-9)

    for row, combo in enumerate(holdout_combos):
        # Find indices of this combo in holdout set
        combo_idx = [i for i, c in enumerate(cbs_holdout) if c == combo][:n_examples]

        # Show real images
        for col, idx in enumerate(combo_idx):
            img, *_ = holdout_ds[idx]
            axes[row][col].imshow(img.permute(1, 2, 0).numpy())
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_ylabel(combo.replace("_", "\n"), fontsize=8, rotation=0,
                                           labelpad=70, va="center")

        # Last column: CCS info
        ax_info = axes[row][-1]
        z_comp  = composed_z_per_combo.get(combo)
        if z_comp is not None:
            z_comp_n = z_comp / (np.linalg.norm(z_comp) + 1e-9)
            sims = Z_tr_n @ z_comp_n
            top5_idx  = np.argsort(sims)[-5:][::-1]
            top5_cbs  = [cbs_train[i] for i in top5_idx]
            top5_sims = [float(sims[i]) for i in top5_idx]
            text  = "Nearest (composed):\n" + "\n".join(
                f"{c.replace('_',' ')[:18]}  {s:.3f}" for c, s in zip(top5_cbs, top5_sims)
            )
            ax_info.text(0.05, 0.5, text, transform=ax_info.transAxes,
                         va="center", fontsize=7, family="monospace")
        ax_info.axis("off")

        # Fill empty columns
        for col in range(len(combo_idx), n_examples):
            axes[row][col].axis("off")

    fig.suptitle("Zero-shot Holdout Objects + Nearest Neighbours (from composed embedding)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "zero_shot_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
