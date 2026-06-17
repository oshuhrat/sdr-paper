"""
Statistical analysis and visualisations for the SDR Structural Retention Test.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, ttest_ind, spearmanr

from life_engine import run as life_run
from patterns import PATTERNS, PATTERN_NAMES, place_on_empty, random_config, CATEGORY

PLOTS_DIR = Path("plots")
TAUS      = [10, 20, 50, 100, 200]
ALL_TYPES = ["random"] + PATTERN_NAMES
SR_COLS   = ["nmi_old", "sr_shift", "sr_shape", "sr_center", "sr_comp"]
SR_LABELS = {
    "nmi_old":   "NMI (v1, per-cell)",
    "sr_shift":  "SR_shift (FFT align)",
    "sr_shape":  "SR_shape (topology)",
    "sr_center": "SR_center (COM align)",
    "sr_comp":   "SR_comp (complexity)",
}

PALETTE = {
    "random":  "#888888",
    "block":   "#E63946",
    "beehive": "#F4A261",
    "loaf":    "#2A9D8F",
    "boat":    "#457B9D",
    "blinker": "#A8DADC",
    "toad":    "#8338EC",
    "glider":  "#FB5607",
    "lwss":    "#FFBE0B",
    "pulsar":  "#3A86FF",
}


def _ensure():
    PLOTS_DIR.mkdir(exist_ok=True)


def _safe_spearman(x, y):
    if x.nunique() < 2 or y.nunique() < 2:
        return 0.0, 1.0
    rho, p = spearmanr(x, y)
    return (float(rho) if rho == rho else 0.0,
            float(p)   if p   == p   else 1.0)


# ── Statistics ─────────────────────────────────────────────────────────────────

def statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U + Welch t for each metric × type vs random at tau=200."""
    d200 = df[df.tau == 200]
    noise = d200[d200.object_type == "random"]
    rows = []
    for obj in PATTERN_NAMES:
        grp = d200[d200.object_type == obj]
        for metric in SR_COLS:
            u, p_mw = mannwhitneyu(grp[metric], noise[metric], alternative="greater")
            t, p_t  = ttest_ind(grp[metric], noise[metric], equal_var=False, alternative="greater")
            rows.append({
                "object_type": obj,
                "metric":      metric,
                "mean_val":    grp[metric].mean(),
                "std_val":     grp[metric].std(),
                "mean_noise":  noise[metric].mean(),
                "p_mw":        p_mw,
                "p_t":         p_t,
                "significant": bool(p_mw < 0.01 and p_t < 0.01),
            })
    return pd.DataFrame(rows)


def survival_vs_metrics(df_mixed: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ between survival probability and each SR metric."""
    agg = df_mixed.groupby("object_type").agg(
        survival=("survived", "mean"),
        nmi_old=("nmi_old", "mean"),
        sr_shift=("sr_shift", "mean"),
        sr_shape=("sr_shape", "mean"),
        sr_center=("sr_center", "mean"),
        sr_comp=("sr_comp", "mean"),
        sr_mean=("sr_mean", "mean"),
    ).reset_index()
    rows = []
    for metric in SR_COLS + ["sr_mean"]:
        rho, p = _safe_spearman(agg[metric], agg["survival"])
        rows.append({"metric": metric, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows), agg


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_metric_profiles(df: pd.DataFrame):
    """5-panel line plot: each SR metric vs tau for all object types."""
    _ensure()
    fig, axes = plt.subplots(1, len(SR_COLS), figsize=(22, 5), sharey=False)
    for ax, metric in zip(axes, SR_COLS):
        for obj in ALL_TYPES:
            sub = df[df.object_type == obj]
            means = [sub[sub.tau == tau][metric].mean() for tau in TAUS]
            ax.plot(TAUS, means, "o-", label=obj, color=PALETTE[obj], linewidth=2, markersize=5)
        ax.set_title(SR_LABELS[metric], fontsize=10)
        ax.set_xlabel("τ  (steps)")
        ax.set_ylabel("Score")
        ax.set_xticks(TAUS)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Structural Retention Metrics  SR_τ  vs  τ\n"
                 "Key: SR_shift captures Glider/LWSS; NMI_old misses them.",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metric_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_tau200_heatmap(df: pd.DataFrame):
    """Heatmap: all metrics × all types at τ=200."""
    _ensure()
    d200 = df[df.tau == 200]
    pivot = pd.DataFrame({
        metric: d200.groupby("object_type")[metric].mean()
        for metric in SR_COLS
    }).reindex(ALL_TYPES)
    pivot.columns = [SR_LABELS[m].split("(")[0].strip() for m in SR_COLS]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                cbar_kws={"label": "Mean score at τ=200"})
    ax.set_title("All SR Metrics at τ=200  (rows = object types, cols = metrics)\n"
                 "Green = high structural retention, Red = low", fontsize=11)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Object type")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tau200_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_glider_vs_nmi_comparison(df: pd.DataFrame):
    """
    Side-by-side bar: NMI_old vs SR_shift for Glider and Block.
    The key figure showing that NMI_old 'misses' the Glider.
    """
    _ensure()
    d200 = df[df.tau == 200]
    focus = ["block", "glider", "lwss", "random"]
    agg = d200[d200.object_type.isin(focus)].groupby("object_type")[SR_COLS].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(focus))
    w = 0.35
    for ax, (m1, m2, title) in [
        (axes[0], ("nmi_old",  "sr_shift",  "NMI_old vs SR_shift")),
        (axes[1], ("sr_shape", "sr_center", "SR_shape vs SR_center")),
    ]:
        bars1 = ax.bar(x - w/2, [agg.loc[o, m1] if o in agg.index else 0 for o in focus],
                       w, label=SR_LABELS[m1], color="#E63946", alpha=0.85)
        bars2 = ax.bar(x + w/2, [agg.loc[o, m2] if o in agg.index else 0 for o in focus],
                       w, label=SR_LABELS[m2], color="#2A9D8F", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(focus)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score at τ=200")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(1.0, color="k", linewidth=0.5, linestyle="--")

    fig.suptitle("Key Result: NMI_old misses Glider and LWSS\n"
                 "SR_shift (FFT alignment) correctly scores them as ≈1.0",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "glider_nmi_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_translation_test(df_trans: pd.DataFrame):
    """
    Glider Translation Test results:
    Show NMI vs SR_shift for case A (pure shift) and case B (evolved).
    """
    _ensure()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_show = ["nmi_old", "sr_shift", "sr_shape"]
    labels_show  = [SR_LABELS[m].split("(")[0].strip() for m in metrics_show]

    for ax, obj in zip(axes, ["glider", "block", "random"]):
        sub = df_trans[df_trans.object_type == obj]
        caseA = sub[sub.case == "A_shift_only"]
        caseB = sub[sub.case == "B_evolved_200"]

        x = np.arange(len(metrics_show))
        w = 0.3
        ax.bar(x - w/2, [caseA[m].mean() for m in metrics_show],
               w, label=f"Artificial shift (+{20})", color="#2A9D8F", alpha=0.9)
        ax.bar(x + w/2, [caseB[m].mean() for m in metrics_show],
               w, label="GoL evolved (τ=200)", color="#E63946", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_show, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{obj.capitalize()}", fontsize=12)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(1.0, color="k", linewidth=0.5, linestyle="--")

    fig.suptitle(
        "Glider Translation Test (validation of SR_shift)\n"
        "Case A (green): artificial shift (+20,+20). Case B (red): evolved 200 steps.\n"
        "SR_shift should be high in both cases; NMI_old should be near 0 in Case A.",
        fontsize=10, y=1.05,
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "translation_test.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_survival_correlation(agg_surv: pd.DataFrame, corr_df: pd.DataFrame):
    """Scatter: each metric vs survival probability."""
    _ensure()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for ax, metric in zip(axes, SR_COLS + ["sr_mean"]):
        ax.scatter(
            agg_surv[metric], agg_surv["survival"],
            s=120, c=[PALETTE.get(o, "#999") for o in agg_surv.object_type],
            zorder=3, edgecolors="k", linewidth=0.5,
        )
        for _, row in agg_surv.iterrows():
            ax.annotate(row.object_type, (row[metric], row.survival),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        rho, p = _safe_spearman(agg_surv[metric], agg_surv["survival"])
        ax.set_xlabel(SR_LABELS.get(metric, metric), fontsize=9)
        ax.set_ylabel("Survival Probability")
        ax.set_title(f"ρ={rho:.3f}  (p={p:.3f})", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.1)

    fig.suptitle(
        "Survival Probability vs Each Structural Retention Metric\n"
        "(Noise + Structure experiment, N=500 trials/type)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_invariant_drift(df: pd.DataFrame):
    """Line: invariant cosine distance from t=0 for each type vs tau."""
    _ensure()
    fig, ax = plt.subplots(figsize=(12, 6))
    for obj in ALL_TYPES:
        sub = df[df.object_type == obj]
        means = [sub[sub.tau == tau]["inv_cos"].mean() for tau in TAUS]
        ax.plot(TAUS, means, "o-", label=obj, color=PALETTE.get(obj, "#999"),
                linewidth=2, markersize=5)
    ax.set_xlabel("τ  (steps)")
    ax.set_ylabel("Cosine distance of invariant vectors (lower = more stable)")
    ax.set_title("Invariant Vector Drift from t=0\n"
                 "(0 = identical invariants, 1 = completely different)")
    ax.set_xticks(TAUS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "invariant_drift.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sdr_score_v2(df: pd.DataFrame, df_mixed: pd.DataFrame):
    """
    SDR_SCORE_V2 = SR_mean × Persistence / EnergyCost
    """
    _ensure()
    # Persistence from mixed experiment
    pers = df_mixed.groupby("object_type")["survived"].mean().rename("persistence")
    # SR_mean and energy at tau=200
    d200 = df[df.tau == 200].groupby("object_type").agg(
        sr_mean=("sr_mean", "mean"),
        energy=("energy_cost", "mean"),
    )
    combined = d200.join(pers)
    combined["persistence"] = combined["persistence"].fillna(0.5)   # random has no mixed trial
    combined["energy_safe"] = combined["energy"].clip(lower=1.0)
    combined["sdr_v2"] = combined["sr_mean"] * combined["persistence"] / combined["energy_safe"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    combined["sdr_v2"].reindex(ALL_TYPES).plot.bar(
        ax=axes[0],
        color=[PALETTE.get(o, "#999") for o in combined.reindex(ALL_TYPES).index],
        edgecolor="k",
    )
    axes[0].set_title("SDR_SCORE_V2 = SR_mean × Persistence / EnergyCost")
    axes[0].set_xlabel("Object type")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(True, axis="y", alpha=0.3)

    combined["sr_mean"].reindex(ALL_TYPES).plot.bar(
        ax=axes[1],
        color=[PALETTE.get(o, "#999") for o in combined.reindex(ALL_TYPES).index],
        edgecolor="k",
    )
    axes[1].set_title("Mean SR (avg of 4 metrics) at τ=200")
    axes[1].set_xlabel("Object type")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, axis="y", alpha=0.3)

    combined["energy_safe"].reindex(ALL_TYPES).plot.bar(
        ax=axes[2],
        color=[PALETTE.get(o, "#999") for o in combined.reindex(ALL_TYPES).index],
        edgecolor="k",
    )
    axes[2].set_title("Mean Energy Cost (changed cells/step)")
    axes[2].set_xlabel("Object type")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("SDR_SCORE_V2 Decomposition", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sdr_score_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    return combined


def plot_field_evolution():
    """Example evolution grids for key objects."""
    _ensure()
    from life_engine import run as lrun
    SNAP   = [0, 10, 50, 200]
    SHOWCASES = ["random", "block", "glider", "lwss", "pulsar"]

    fig, axes = plt.subplots(len(SHOWCASES), len(SNAP), figsize=(16, len(SHOWCASES) * 3.2))
    for i, obj in enumerate(SHOWCASES):
        rng = np.random.default_rng(77 + i)
        if obj == "random":
            grid = random_config(100, 0.5, rng)
        else:
            grid = place_on_empty(100, PATTERNS[obj], rng)
        states, _ = lrun(grid, 200, tuple(SNAP))
        for j, t in enumerate(SNAP):
            ax = axes[i][j]
            ax.imshow(states[t], cmap="binary_r", interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(f"{obj}  t={t}", fontsize=9)
            ax.axis("off")

    plt.suptitle("Field Evolution Examples (100×100 torus)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "field_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_separation(df: pd.DataFrame):
    """
    Box plots: separate structured classes from random at τ=200.
    Shows whether all stable structures cluster above random.
    """
    _ensure()
    d200 = df[df.tau == 200].copy()
    d200["category"] = d200["object_type"].map(CATEGORY).fillna("random")

    cat_order = ["still_life", "oscillator_p2", "oscillator_p3", "spaceship", "random"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in [(axes[0], "sr_shift"), (axes[1], "sr_mean")]:
        sns.boxplot(
            data=d200, x="category", y=metric,
            order=[c for c in cat_order if c in d200.category.values],
            hue="category",
            palette={"still_life": "#E63946", "oscillator_p2": "#8338EC",
                     "oscillator_p3": "#3A86FF", "spaceship": "#FB5607", "random": "#888"},
            legend=False, ax=ax,
        )
        ax.set_title(SR_LABELS.get(metric, metric) + " at τ=200 by structural class")
        ax.set_xlabel("Structural category")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Class Separation Test: do all stable structures score above random?",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_separation.png", dpi=150, bbox_inches="tight")
    plt.close()
