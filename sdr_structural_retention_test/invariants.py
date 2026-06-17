"""
Shape invariants for binary GoL grids.

An invariant I(F) is a quantity that is preserved under the symmetry group
of the object.  Here we focus on spatial-translation invariants, since that
is the primary symmetry of moving patterns (Glider, LWSS).

Invariants computed:
  1. n_alive           – number of alive cells (conserved by still lifes exactly)
  2. n_components      – topology (connected components)
  3. component_sizes   – sorted list of component areas
  4. central_moments   – mu20, mu02, mu11 (shape descriptors, translation-invariant)
  5. fft_power_top20   – top-20 magnitudes of |FFT|^2 (shift-invariant by Parseval)
  6. periodicity_score – detected period of the object (0 if unknown)

The invariant DISTANCE between t=0 and t=τ measures structural drift.
"""
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import normalized_mutual_info_score


# ── Extraction ─────────────────────────────────────────────────────────────────

def extract(grid: np.ndarray) -> dict:
    labeled, n_comp = label(grid)
    n_alive = int(grid.sum())

    sizes = sorted(
        [int((labeled == i).sum()) for i in range(1, n_comp + 1)],
        reverse=True,
    )

    if n_alive > 0:
        r, c = np.nonzero(grid)
        cr, cc = float(r.mean()), float(c.mean())
        mu20 = float(((r - cr) ** 2).mean())
        mu02 = float(((c - cc) ** 2).mean())
        mu11 = float(((r - cr) * (c - cc)).mean())
        span_r = int(r.max() - r.min())
        span_c = int(c.max() - c.min())
    else:
        mu20 = mu02 = mu11 = 0.0
        span_r = span_c = 0

    # Shift-invariant Fourier power spectrum
    fft_mag2 = (np.abs(np.fft.rfft2(grid.astype(np.float32))) ** 2).ravel()
    top20_idx = np.argpartition(fft_mag2, -20)[-20:]
    fft_top20 = float(fft_mag2[top20_idx].mean())    # summary scalar

    return {
        "n_alive":      n_alive,
        "n_components": n_comp,
        "top5_sizes":   sizes[:5] + [0] * max(0, 5 - len(sizes)),
        "mu20":         mu20,
        "mu02":         mu02,
        "mu11":         mu11,
        "span_r":       span_r,
        "span_c":       span_c,
        "fft_power":    fft_top20,
    }


def to_vector(inv: dict, grid_size: int = 100) -> np.ndarray:
    """
    Flatten invariant dict to a normalised float64 vector for cosine distance.

    All features are scaled to roughly [0, 1] so that no single feature
    (especially fft_power, which spans many orders of magnitude) dominates
    the cosine direction.
    """
    N = float(grid_size ** 2)
    return np.array([
        inv["n_alive"]      / N,          # alive density
        inv["n_components"] / N,          # component density
        *[s / N for s in inv["top5_sizes"]],
        np.log1p(inv["mu20"]) / np.log1p(N),   # log-normalised moments
        np.log1p(inv["mu02"]) / np.log1p(N),
        np.log1p(abs(inv["mu11"])) / np.log1p(N),
        inv["span_r"] / grid_size,
        inv["span_c"] / grid_size,
        np.log1p(inv["fft_power"]) / np.log1p(N * N),  # log-normalised power
    ], dtype=np.float64)


# ── Distance ──────────────────────────────────────────────────────────────────

def cosine_distance(inv0: dict, invt: dict) -> float:
    """1 - cosine_similarity(vector(inv0), vector(invt))."""
    v0 = to_vector(inv0)
    vt = to_vector(invt)
    d0 = np.linalg.norm(v0)
    dt = np.linalg.norm(vt)
    if d0 < 1e-12 or dt < 1e-12:
        return 1.0
    return float(1.0 - np.dot(v0, vt) / (d0 * dt))


def l1_relative(inv0: dict, invt: dict) -> float:
    """Relative L1 distance (sum of |Δfeature| / range), in [0,1]."""
    v0 = to_vector(inv0)
    vt = to_vector(invt)
    denom = np.abs(v0) + np.abs(vt) + 1e-9
    return float((np.abs(v0 - vt) / denom).mean())


# ── Periodicity detection ─────────────────────────────────────────────────────

def detect_period(
    grid: np.ndarray,
    max_period: int = 20,
    n_steps: int = None,
) -> int:
    """
    Run GoL up to max_period steps and check if grid returns to original.
    Returns detected period (1 for still life, 0 if not periodic within max_period).
    """
    from life_engine import step as life_step
    if n_steps is None:
        n_steps = max_period
    cur = grid.copy()
    for p in range(1, n_steps + 1):
        cur = life_step(cur)
        if np.array_equal(cur, grid):
            return p
    return 0


# ── Summary across time ───────────────────────────────────────────────────────

def compute_invariant_profile(
    states: dict[int, np.ndarray],
    taus: tuple[int, ...],
) -> dict:
    """
    For each tau: extract invariants from states[tau] and compute distance
    from states[0].  Returns flat dict suitable for DataFrame row.
    """
    inv0 = extract(states[0])
    result = {}
    for tau in taus:
        if tau == 0:
            continue
        invt = extract(states[tau])
        result[f"inv_cos_{tau}"]   = cosine_distance(inv0, invt)
        result[f"inv_l1_{tau}"]    = l1_relative(inv0, invt)
        result[f"n_alive_{tau}"]   = invt["n_alive"]
        result[f"n_comp_{tau}"]    = invt["n_components"]
        result[f"fft_power_{tau}"] = invt["fft_power"]
    return result
