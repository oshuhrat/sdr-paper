"""
ConceptNode and ConceptGraph for SDR-Agent v2.

Concept extraction strategy:
  Run K-Means separately for each attribute (shape K=3, color K=3, size K=2).
  Assign labels to clusters by majority vote using known training labels.
  This gives us 8 concept prototypes (embeddings in R^128).

Also supports supervised concept extraction:
  C[attr][val] = mean(z) for all training images with attribute=val
  This is used as the ground-truth reference concept.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


# ── ConceptNode ────────────────────────────────────────────────────────────────

@dataclass
class ConceptNode:
    id:          int
    label:       str              # e.g. "red", "circle", "large"
    attribute:   str              # "color", "shape", "size"
    embedding:   np.ndarray       # centroid in R^128 (L2-normalised)
    count:       int              # number of training examples
    persistence: float            # count / total in that attribute group
    variance:    float            # mean squared distance from centroid
    neighbors:   List[int] = field(default_factory=list)  # ids of nearest nodes


# ── ConceptGraph ───────────────────────────────────────────────────────────────

class ConceptGraph:
    """
    Stores concept prototypes for each attribute dimension.

    After construction:
      self.concepts["color"]["red"]    → ConceptNode
      self.concepts["shape"]["circle"] → ConceptNode
      self.concepts["size"]["large"]   → ConceptNode
    """

    def __init__(self):
        # {attr: {value: ConceptNode}}
        self.concepts: Dict[str, Dict[str, ConceptNode]] = {}
        self._node_list: List[ConceptNode] = []

    # ── Supervised extraction (preferred) ─────────────────────────────────────

    def build_from_embeddings(
        self,
        Z:      np.ndarray,    # (N, d)  L2-normalised embeddings
        colors: List[str],
        sizes:  List[str],
        shapes: List[str],
    ) -> None:
        """
        Build concept prototypes by averaging embeddings per attribute value.
        No K-Means required — uses known training labels.
        """
        self.concepts = {"color": {}, "size": {}, "shape": {}}
        node_id = 0

        for attr, values_list in [
            ("color", colors),
            ("size",  sizes),
            ("shape", shapes),
        ]:
            unique_vals = sorted(set(values_list))
            for val in unique_vals:
                mask = np.array([v == val for v in values_list])
                z_subset = Z[mask]
                centroid = z_subset.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                variance = float(((z_subset - centroid) ** 2).sum(axis=1).mean())
                node = ConceptNode(
                    id=node_id, label=val, attribute=attr,
                    embedding=centroid,
                    count=int(mask.sum()),
                    persistence=float(mask.sum()) / max(len(Z), 1),
                    variance=variance,
                )
                self.concepts[attr][val] = node
                self._node_list.append(node)
                node_id += 1

        self._build_neighbor_links()

    # ── K-Means extraction (unsupervised, for validation) ──────────────────────

    def cluster_embeddings(
        self,
        Z:      np.ndarray,
        colors: List[str],
        sizes:  List[str],
        shapes: List[str],
        seed:   int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Cluster per attribute using K-Means. Assign labels by majority vote.
        Returns {attr: cluster_labels_array} for purity analysis.
        """
        attr_config = [
            ("color", colors, sorted(set(colors))),
            ("size",  sizes,  sorted(set(sizes))),
            ("shape", shapes, sorted(set(shapes))),
        ]
        km_labels = {}
        for attr, attr_vals, unique_vals in attr_config:
            k  = len(unique_vals)
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            cl = km.fit_predict(Z)
            km_labels[attr] = cl

            # Label each cluster by majority vote
            cluster_label_map = {}
            for c in range(k):
                mask    = cl == c
                members = [attr_vals[i] for i, m in enumerate(mask) if m]
                if members:
                    vals, cnts = np.unique(members, return_counts=True)
                    cluster_label_map[c] = vals[cnts.argmax()]
                else:
                    cluster_label_map[c] = "?"

        return km_labels

    # ── Query interface ────────────────────────────────────────────────────────

    def add_observation(self, z: np.ndarray, label: str = "") -> None:
        """Placeholder for streaming observation (used in interactive mode)."""
        pass   # Full implementation would update running mean per concept

    def nearest_concepts(
        self,
        z:     np.ndarray,
        k:     int = 3,
        attr:  Optional[str] = None,
    ) -> List[Tuple[ConceptNode, float]]:
        """
        Find k nearest ConceptNodes to query z (by cosine similarity).
        Optionally restrict to a single attribute group.
        """
        z = z / (np.linalg.norm(z) + 1e-9)
        candidates = (
            self._node_list if attr is None
            else list(self.concepts[attr].values())
        )
        sims = [(n, float(n.embedding @ z)) for n in candidates]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def find_composition(
        self,
        color: str,
        size:  str,
        shape: str,
        method: str = "mean",
        weights: Optional[np.ndarray] = None,
        linear_composer=None,
    ) -> np.ndarray:
        """
        Compose concept embeddings into a predicted object embedding.

        method = "mean"          → simple average of 3 concept prototypes
        method = "weighted_mean" → weighted by 1/variance (tighter = more weight)
        method = "learned_linear"→ uses fitted LearnedLinearComposer
        """
        c_color = self.concepts["color"][color].embedding
        c_size  = self.concepts["size"][size].embedding
        c_shape = self.concepts["shape"][shape].embedding

        if method == "mean":
            z_pred = (c_color + c_size + c_shape) / 3.0

        elif method == "weighted_mean":
            v_color = self.concepts["color"][color].variance
            v_size  = self.concepts["size"][size].variance
            v_shape = self.concepts["shape"][shape].variance
            w = 1.0 / (np.array([v_color, v_size, v_shape]) + 1e-6)
            w = w / w.sum()
            z_pred = w[0] * c_color + w[1] * c_size + w[2] * c_shape

        elif method == "learned_linear":
            assert linear_composer is not None, "Pass a fitted LearnedLinearComposer"
            z_pred = linear_composer.predict(c_color, c_size, c_shape)

        else:
            raise ValueError(f"Unknown method: {method}")

        # L2-normalise the composed embedding
        norm = np.linalg.norm(z_pred)
        return z_pred / norm if norm > 1e-9 else z_pred

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_neighbor_links(self, k: int = 3):
        """Connect each node to its k nearest other nodes."""
        if len(self._node_list) < 2:
            return
        C = np.stack([n.embedding for n in self._node_list])
        sim = C @ C.T
        for i, node in enumerate(self._node_list):
            row = sim[i].copy()
            row[i] = -1e9                           # exclude self
            top_k = np.argsort(row)[-k:][::-1]
            node.neighbors = [int(j) for j in top_k]

    def summary(self) -> str:
        lines = ["ConceptGraph summary:", ""]
        for attr in ["color", "size", "shape"]:
            lines.append(f"  {attr}:")
            for val, node in sorted(self.concepts.get(attr, {}).items()):
                lines.append(
                    f"    {val:<10}  count={node.count:>4}  "
                    f"persist={node.persistence:.3f}  var={node.variance:.4f}"
                )
        return "\n".join(lines)
