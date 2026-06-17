"""
Evaluation suite for SDR-Agent v2 — Concept Composition.

Tests:
  1. Concept quality:  how tight and pure are the 8 concept clusters?
  2. CCS (main):       cosine_sim(z_real, z_composed) for 3 holdout objects
  3. CCS baseline:     random composition (shuffled attribute concepts)
  4. Zero-shot retrieval: use z_composed to find holdout images in embedding space
  5. Composition method comparison: mean vs weighted_mean vs learned_linear
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from dataset import ShapeDataset, COLORS, SIZES, SHAPES, HOLDOUT
from concept_graph import ConceptGraph
from composition import (
    MeanComposer, WeightedMeanComposer, LearnedLinearComposer, cosine_similarity
)


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract(
    model,
    dataset: ShapeDataset,
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[str], List[str], List[str], List[str]]:
    """
    Returns (Z, colors, sizes, shapes, combos)
    Z shape: (N, embed_dim)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    Zs, cols, szs, shs, cbs = [], [], [], [], []
    for img, color, size, shape, combo in loader:
        z = model.encode(img.to(device)).cpu().numpy()
        Zs.append(z)
        cols.extend(list(color)); szs.extend(list(size))
        shs.extend(list(shape));  cbs.extend(list(combo))
    return np.concatenate(Zs), cols, szs, shs, cbs


# ── Concept quality ────────────────────────────────────────────────────────────

def concept_separation(graph: ConceptGraph) -> dict:
    """
    For each attribute, measure how separable the concept embeddings are.
    Returns mean inter-concept cosine distance (higher = more separable).
    """
    result = {}
    for attr in ["color", "size", "shape"]:
        nodes  = list(graph.concepts[attr].values())
        embeds = np.stack([n.embedding for n in nodes])
        # pairwise cosine
        sims = embeds @ embeds.T
        n = len(nodes)
        if n < 2:
            result[attr] = 0.0
            continue
        mask = ~np.eye(n, dtype=bool)
        result[attr] = float(1.0 - sims[mask].mean())   # distance = 1 - similarity
    return result


# ── Build concept graph ────────────────────────────────────────────────────────

def build_concept_graph(
    model,
    train_ds: ShapeDataset,
    device:   str = "cpu",
) -> Tuple[ConceptGraph, np.ndarray, List[str], List[str], List[str], List[str]]:
    """Extract training embeddings and build concept graph."""
    Z, colors, sizes, shapes, combos = extract(model, train_ds, device=device)
    graph = ConceptGraph()
    graph.build_from_embeddings(Z, colors, sizes, shapes)
    return graph, Z, colors, sizes, shapes, combos


# ── Fit learned linear composer ───────────────────────────────────────────────

def fit_learned_linear(
    graph:  ConceptGraph,
    Z:      np.ndarray,
    colors: List[str],
    sizes:  List[str],
    shapes: List[str],
    combos: List[str],
) -> LearnedLinearComposer:
    """
    For each of the 15 training combos, compute:
      - concept embeddings for its color, size, shape
      - mean embedding of all images with that combo
    Then fit Ridge regression.
    """
    unique_combos = sorted(set(combos))
    c_colors, c_sizes, c_shapes, targets = [], [], [], []

    for combo in unique_combos:
        parts = combo.split("_")
        color, size, shape = parts[0], parts[1], parts[2]
        c_colors.append(graph.concepts["color"][color].embedding)
        c_sizes.append( graph.concepts["size"][size].embedding)
        c_shapes.append(graph.concepts["shape"][shape].embedding)
        mask   = np.array([c == combo for c in combos])
        z_mean = Z[mask].mean(axis=0)
        z_mean = z_mean / (np.linalg.norm(z_mean) + 1e-9)
        targets.append(z_mean)

    composer = LearnedLinearComposer()
    composer.fit(
        np.stack(c_colors),
        np.stack(c_sizes),
        np.stack(c_shapes),
        np.stack(targets),
    )
    return composer


# ── Main CCS evaluation ────────────────────────────────────────────────────────

def evaluate_ccs(
    model,
    graph:            ConceptGraph,
    learned_composer: LearnedLinearComposer,
    device:           str = "cpu",
    n_per_holdout:    int = 100,
) -> dict:
    """
    For each holdout combo:
      1. Extract z_real from encoder (images the model never saw during training)
      2. Compose z_pred using mean / weighted_mean / learned_linear
      3. CCS = cosine_sim(z_real, z_pred)
      4. Random baseline: random shuffle of concept assignments
    """
    holdout_ds = ShapeDataset(n_per_combo=n_per_holdout, holdout=False, seed=99999)
    Z_h, cols_h, szs_h, shs_h, cbs_h = extract(model, holdout_ds, device=device)

    composers = [
        MeanComposer(),
        WeightedMeanComposer(),
        learned_composer,
    ]

    rows = []
    for combo_key in sorted(set(cbs_h)):
        color, size, shape = combo_key.split("_")
        mask  = np.array([c == combo_key for c in cbs_h])
        z_real = Z_h[mask].mean(axis=0)
        z_real = z_real / (np.linalg.norm(z_real) + 1e-9)

        v_color = graph.concepts["color"][color].variance
        v_size  = graph.concepts["size"][size].variance
        v_shape = graph.concepts["shape"][shape].variance

        c_color = graph.concepts["color"][color].embedding
        c_size  = graph.concepts["size"][size].embedding
        c_shape = graph.concepts["shape"][shape].embedding

        for comp in composers:
            if isinstance(comp, WeightedMeanComposer):
                z_pred = comp.predict(c_color, c_size, c_shape,
                                      v_color=v_color, v_size=v_size, v_shape=v_shape)
            else:
                z_pred = comp.predict(c_color, c_size, c_shape)
            ccs = cosine_similarity(z_real, z_pred)
            rows.append({
                "combo":  combo_key,
                "method": comp.name,
                "ccs":    ccs,
                "z_real": z_real,
                "z_pred": z_pred,
            })

        # Random baseline: shuffle attribute assignments
        rng = np.random.default_rng(0)
        for _ in range(20):
            rc = rng.choice(list(graph.concepts["color"].keys()))
            rs = rng.choice(list(graph.concepts["size"].keys()))
            rsh= rng.choice(list(graph.concepts["shape"].keys()))
            # Ensure at least one wrong assignment
            if rc == color and rs == size and rsh == shape:
                rsh = [s for s in graph.concepts["shape"] if s != shape][0]
            z_rand = (graph.concepts["color"][rc].embedding +
                      graph.concepts["size"][rs].embedding +
                      graph.concepts["shape"][rsh].embedding) / 3.0
            z_rand = z_rand / (np.linalg.norm(z_rand) + 1e-9)
            rows.append({
                "combo": combo_key, "method": "random_baseline",
                "ccs":   cosine_similarity(z_real, z_rand),
                "z_real": z_real, "z_pred": z_rand,
            })

    return {"rows": rows, "Z_holdout": Z_h, "combos_holdout": cbs_h}


# ── Zero-shot retrieval ────────────────────────────────────────────────────────

def zero_shot_retrieval(
    z_composed:   np.ndarray,     # composed embedding for holdout object
    Z_all:        np.ndarray,     # all test embeddings (including holdout)
    combos_all:   List[str],      # combo label for each row in Z_all
    target_combo: str,
    k_list:       List[int] = (1, 5, 10),
) -> dict:
    """
    Use composed embedding as a query.
    Search in Z_all (which includes real holdout images).
    Compute Recall@k: are holdout images in top-k?
    """
    z_q = z_composed / (np.linalg.norm(z_composed) + 1e-9)
    Z_n = Z_all / (np.linalg.norm(Z_all, axis=1, keepdims=True) + 1e-9)
    sims = Z_n @ z_q
    order = np.argsort(sims)[::-1]

    result = {}
    for k in k_list:
        top_k_combos = [combos_all[i] for i in order[:k]]
        hits = sum(1 for c in top_k_combos if c == target_combo)
        total_target = sum(1 for c in combos_all if c == target_combo)
        result[f"recall@{k}"] = hits / max(total_target, 1)
        result[f"precision@{k}"] = hits / k
    result["top5_combos"] = [combos_all[i] for i in order[:5]]
    return result
