"""
Four shift/structure-invariant Structural Retention metrics (SR_τ).

Why per-cell NMI fails for moving objects:
  NMI(X_0, X_τ) measures how much knowing "is cell (r,c) alive at t=0?"
  predicts "is cell (r,c) alive at t=τ?".
  For a Glider that has moved, the alive cells are at *different positions*,
  so this mutual information is near zero — a false null.

The four metrics fix this via different invariance assumptions:

  SR_shift  — invariant to arbitrary translation (FFT cross-correlation)
  SR_shape  — invariant to translation AND position (topology/moments)
  SR_center — invariant to translation (center-of-mass alignment)
  SR_comp   — invariant to spatial structure (Kolmogorov complexity proxy)
"""
import gzip
import numpy as np
from scipy.ndimage import label, center_of_mass
from sklearn.metrics import normalized_mutual_info_score


# ── helpers ───────────────────────────────────────────────────────────────────

def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    return float(normalized_mutual_info_score(
        a.ravel(), b.ravel(), average_method="geometric"
    ))


def _shape_feature_vector(grid: np.ndarray) -> np.ndarray:
    """
    15-dimensional feature vector:
      [n_components, n_alive,
       sorted_top5_component_areas (padded with 0),
       mu20/n, mu02/n, mu11/n, aspect_ratio]
    All normalised so vectors from different densities are comparable.
    """
    labeled, n_comp = label(grid)
    n_alive = int(grid.sum())

    areas = sorted(
        [int((labeled == i).sum()) for i in range(1, n_comp + 1)],
        reverse=True,
    )[:5]
    areas += [0] * (5 - len(areas))          # pad to length 5

    if n_alive > 0:
        r, c = np.nonzero(grid)
        cr, cc = r.mean(), c.mean()
        mu20 = float(((r - cr) ** 2).mean())
        mu02 = float(((c - cc) ** 2).mean())
        mu11 = float(((r - cr) * (c - cc)).mean())
        # Normalize by alive count so scale-independent
        norm = float(n_alive)
        c_range = int(c.max() - c.min())
        r_range = int(r.max() - r.min())
        asp = float(r_range + 1) / float(c_range + 1) if c_range > 0 else 1.0
    else:
        mu20 = mu02 = mu11 = asp = 0.0
        norm = 1.0

    N = float(grid.size)   # 10000 for 100×100

    # Normalise all features to roughly [0, 1] so no single feature dominates
    # the cosine direction.  n_alive and areas become densities (fraction of N).
    # Moments are already per-cell; aspect ratio is already dimensionless.
    feat = np.array(
        [n_comp / N,           # component density
         n_alive / N,          # alive density
         *[a / N for a in areas],
         mu20 / norm,          # per-cell 2nd moment (translation-invariant)
         mu02 / norm,
         mu11 / norm,
         asp],
        dtype=np.float64,
    )
    return feat


# ── METRIC 1: Shift-Invariant MI (SR_shift) ──────────────────────────────────

def sr_shift(
    x0: np.ndarray,
    xt: np.ndarray,
) -> tuple[float, int, int]:
    """
    Find the cyclic shift (dy, dx) that maximises cross-correlation,
    align xt with x0, then compute NMI.

    Algorithm:
      1. FFT cross-correlation in O(N^2 log N) → peak gives best shift
      2. Apply shift to xt (one np.roll each axis)
      3. NMI on aligned pair

    Returns: (NMI_at_best_shift, dy, dx)

    For Glider (moved 50 cells diagonally after 200 steps):
      FFT finds dy=50, dx=50 → aligned → NMI ≈ 1.0
    For Random:
      FFT finds random peak, but NMI of two uncorrelated arrays ≈ 0 even after shift
    """
    if x0.sum() == 0 or xt.sum() == 0:
        return 0.0, 0, 0

    F0 = np.fft.rfft2(x0.astype(np.float32))
    Ft = np.fft.rfft2(xt.astype(np.float32))
    cross = np.fft.irfft2(np.conj(F0) * Ft, s=x0.shape)

    flat_idx = int(cross.argmax())
    dy, dx = divmod(flat_idx, x0.shape[1])

    # Align xt → x0 by shifting xt by (-dy, -dx)
    xt_aligned = np.roll(np.roll(xt, -dy, axis=0), -dx, axis=1)
    nmi_val = _nmi(x0, xt_aligned)
    return nmi_val, int(dy), int(dx)


# ── METRIC 2: Connected-Component Shape (SR_shape) ───────────────────────────

def sr_shape(x0: np.ndarray, xt: np.ndarray) -> float:
    """
    Cosine similarity of shape-feature vectors.

    Features capture: component count, component sizes, central moments,
    aspect ratio.  All are translation-invariant.

    For Glider (period 4, same 5-cell shape at any time):
      features_t ≈ features_{t+τ} → cosine similarity ≈ 1.0
    For Pulsar (period 3, same topology in all phases):
      topological features stable → similarity > NMI
    For Random (complex mix of still-lifes after 200 steps vs initial noise):
      features drift → similarity < structured objects
    """
    f0 = _shape_feature_vector(x0)
    ft = _shape_feature_vector(xt)
    denom = np.linalg.norm(f0) * np.linalg.norm(ft)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(f0, ft) / denom)


# ── METRIC 3: Center-Aligned NMI (SR_center) ─────────────────────────────────

def sr_center(x0: np.ndarray, xt: np.ndarray) -> float:
    """
    Translate each grid so its center of mass is at the grid centre, then NMI.

    Fixes translation for moving objects (Glider, LWSS, Spaceship).
    Does not fix phase changes for oscillators (Pulsar).

    NOTE: for multi-cluster random fields, the center of mass is the centroid
    of all alive cells, which can be near grid-centre anyway — so this metric
    is less discriminative for random fields than sr_shift.
    """
    def _align(g: np.ndarray) -> np.ndarray:
        n_alive = int(g.sum())
        if n_alive == 0:
            return g
        r, c = np.nonzero(g)
        cr = int(round(r.mean()))
        cc = int(round(c.mean()))
        target_r, target_c = g.shape[0] // 2, g.shape[1] // 2
        return np.roll(np.roll(g, target_r - cr, axis=0), target_c - cc, axis=1)

    return _nmi(_align(x0), _align(xt))


# ── METRIC 4: Compression Retention (SR_comp) ────────────────────────────────

def sr_comp(x0: np.ndarray, xt: np.ndarray) -> float:
    """
    Kolmogorov complexity proxy via gzip.

    SR_comp = 1 - |K(X_0) - K(X_τ)| / max(K(X_0), K(X_τ))

    Interpretation:
      Structured objects maintain similar compressibility across time.
      Still life: K(X_0) = K(X_τ) → SR_comp = 1.0
      Glider: same 5-cell pattern → K(X_0) = K(X_τ) → 1.0
      Pulsar: same bit density across phases → K stable → ~1.0
      Random decay: K(X_0) >> K(X_τ) (noise → ordered still lifes) → SR_comp < 1

    Limitation: gzip operates on raw bytes; for sparse matrices with few ones,
    the compression ratio is dominated by the run-length of zeros.
    Still, the ratio K(X_0)/K(X_τ) is a valid complexity ratio.
    """
    def _k(g: np.ndarray) -> int:
        return len(gzip.compress(g.astype(np.uint8).tobytes(), compresslevel=6))

    k0 = _k(x0)
    kt = _k(xt)
    kmax = max(k0, kt, 1)
    return float(1.0 - abs(k0 - kt) / kmax)


# ── All metrics together ──────────────────────────────────────────────────────

def compute_sr_all(
    x0: np.ndarray,
    xt: np.ndarray,
) -> dict:
    """Compute all four SR metrics + derived quantities."""
    shift_val, dy, dx = sr_shift(x0, xt)
    shape_val          = sr_shape(x0, xt)
    center_val         = sr_center(x0, xt)
    comp_val           = sr_comp(x0, xt)
    nmi_old            = _nmi(x0, xt)          # baseline from v1

    sr_mean = float(np.mean([shift_val, shape_val, center_val, comp_val]))

    return {
        "nmi_old":   nmi_old,
        "sr_shift":  shift_val,
        "sr_shift_dy": dy,
        "sr_shift_dx": dx,
        "sr_shape":  shape_val,
        "sr_center": center_val,
        "sr_comp":   comp_val,
        "sr_mean":   sr_mean,
    }
