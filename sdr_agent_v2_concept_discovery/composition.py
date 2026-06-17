"""
Composition methods for predicting embeddings of unseen concept combinations.

Three methods with increasing expressive power:

1. MeanComposer
   z_pred = (C_color + C_size + C_shape) / 3
   Assumption: embedding space is approximately linear in attribute dimensions.
   No parameters. Pure unsupervised.

2. WeightedMeanComposer
   z_pred = Σ w_i * C_i    where w_i = 1/variance_i (normalised)
   Concept with tight cluster (low variance) gets higher weight.
   No parameters. Unsupervised.

3. LearnedLinearComposer
   z_pred = W @ [C_color, C_size, C_shape] + b   (Ridge regression)
   W: R^(3d) → R^d  trained on the 15 training combos.
   NOT seen: the 3 holdout combos.
   Most expressive — finds the best linear combination.
   If this fails (CCS < 0.5), the embedding space has no linear structure at all.
"""
import numpy as np
from sklearn.linear_model import Ridge
from typing import Optional


class MeanComposer:
    name = "mean"

    def predict(
        self,
        c_color: np.ndarray,
        c_size:  np.ndarray,
        c_shape: np.ndarray,
    ) -> np.ndarray:
        z = (c_color + c_size + c_shape) / 3.0
        n = np.linalg.norm(z)
        return z / n if n > 1e-9 else z


class WeightedMeanComposer:
    name = "weighted_mean"

    def __init__(self):
        self._var_color: Optional[float] = None
        self._var_size:  Optional[float] = None
        self._var_shape: Optional[float] = None

    def set_variances(self, v_color: float, v_size: float, v_shape: float):
        self._var_color = v_color
        self._var_size  = v_size
        self._var_shape = v_shape

    def predict(
        self,
        c_color: np.ndarray,
        c_size:  np.ndarray,
        c_shape: np.ndarray,
        v_color: Optional[float] = None,
        v_size:  Optional[float] = None,
        v_shape: Optional[float] = None,
    ) -> np.ndarray:
        vc = v_color if v_color is not None else self._var_color or 1.0
        vs = v_size  if v_size  is not None else self._var_size  or 1.0
        vsh= v_shape if v_shape is not None else self._var_shape or 1.0
        w  = 1.0 / (np.array([vc, vs, vsh]) + 1e-6)
        w  = w / w.sum()
        z  = w[0] * c_color + w[1] * c_size + w[2] * c_shape
        n  = np.linalg.norm(z)
        return z / n if n > 1e-9 else z


class LearnedLinearComposer:
    """
    Ridge regression: [C_color || C_size || C_shape] → z_object

    Training: all 15 non-holdout combos
    Test:     3 holdout combos
    """
    name = "learned_linear"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.reg: Optional[Ridge] = None
        self.n_train = 0

    def fit(
        self,
        color_embeddings: np.ndarray,  # (n_combos, d)
        size_embeddings:  np.ndarray,  # (n_combos, d)
        shape_embeddings: np.ndarray,  # (n_combos, d)
        target_z:         np.ndarray,  # (n_combos, d)  — mean z per combo
    ):
        """
        Train on training combos only.
        X = concat([C_color, C_size, C_shape]) per combo.
        y = mean embedding of that combo in the training set.
        """
        X = np.concatenate([color_embeddings, size_embeddings, shape_embeddings], axis=1)
        self.reg = Ridge(alpha=self.alpha, fit_intercept=True)
        self.reg.fit(X, target_z)
        self.n_train = len(X)

    def predict(
        self,
        c_color: np.ndarray,
        c_size:  np.ndarray,
        c_shape: np.ndarray,
    ) -> np.ndarray:
        assert self.reg is not None, "Call .fit() first"
        x = np.concatenate([c_color, c_size, c_shape]).reshape(1, -1)
        z = self.reg.predict(x)[0]
        n = np.linalg.norm(z)
        return z / n if n > 1e-9 else z


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """CCS — Concept Composition Score."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))
