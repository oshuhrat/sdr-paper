"""
Experiments for the SDR Structural Retention Test (v2).

Experiment 1 – Main:
    1000 instances × 10 types (random + 9 patterns).
    Compute old NMI + 4 new SR metrics + invariant profile at each tau.

Experiment 2 – Mixed:
    N_MIXED instances of noise + structure.
    Track survival to T=200. Used for SDR_SCORE_V2.

Experiment 3 – Glider Translation Test:
    Explicit validation of SR_shift correctness.
    Shows NMI drops to 0 for artificial shift, SR_shift stays ≈ 1.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from life_engine import run
from patterns import PATTERNS, PATTERN_NAMES, place_on_empty, place_on_noise, random_config
from metrics_structural import compute_sr_all, _nmi
from invariants import compute_invariant_profile

# ── Config ────────────────────────────────────────────────────────────────────

GRID_SIZE     = 100
T             = 200
SNAPSHOTS     = (0, 10, 20, 50, 100, 200)
TAUS          = (10, 20, 50, 100, 200)
N_INSTANCES   = 1000
N_MIXED       = 500
NOISE_DENSITY = 0.15  # 0.30 destroys all small patterns (Block needs 8 dead neighbours)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _structure_survival(mixed_t: np.ndarray, struct_t: np.ndarray) -> tuple[float, int]:
    """
    Fraction of the STRUCTURE's alive cells (at their evolved positions in struct_t)
    that are also alive in the mixed field.

    Dice(mixed, struct) fails here because background noise creates many alive cells in
    mixed_t, bloating the denominator.  Instead we ask:
      'Are the specific cells the structure WOULD occupy actually alive in the mixed run?'

    Returns (overlap_fraction, survived_flag with threshold 0.5).
    """
    struct_alive = struct_t.astype(bool)
    n_struct = int(struct_alive.sum())
    if n_struct == 0:
        return 1.0, 1   # both empty = trivially survived
    n_overlap = int((mixed_t.astype(bool) & struct_alive).sum())
    frac = n_overlap / n_struct
    return frac, int(frac >= 0.5)


# ── Experiment 1: main ────────────────────────────────────────────────────────

def _run_one_main(obj_type: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)

    if obj_type == "random":
        grid = random_config(GRID_SIZE, 0.5, rng)
    else:
        grid = place_on_empty(GRID_SIZE, PATTERNS[obj_type], rng)

    states, energy_cost = run(grid, T, SNAPSHOTS)
    inv_profile = compute_invariant_profile(states, TAUS)

    rows = []
    x0 = states[0]
    for tau in TAUS:
        xt = states[tau]
        sr = compute_sr_all(x0, xt)

        # SDR_SCORE_V2 requires Persistence (computed in mixed experiment),
        # so we store energy_cost here and assemble the score later.
        ec_safe = max(energy_cost, 1e-6)
        sdr_v2_partial = sr["sr_mean"] / ec_safe   # × Persistence added post-hoc

        rows.append({
            "object_type": obj_type,
            "trial_id":    seed,
            "tau":         tau,
            # old baseline
            "nmi_old":     sr["nmi_old"],
            # four new metrics
            "sr_shift":    sr["sr_shift"],
            "sr_shift_dy": sr["sr_shift_dy"],
            "sr_shift_dx": sr["sr_shift_dx"],
            "sr_shape":    sr["sr_shape"],
            "sr_center":   sr["sr_center"],
            "sr_comp":     sr["sr_comp"],
            "sr_mean":     sr["sr_mean"],
            # invariant distances
            "inv_cos":     inv_profile.get(f"inv_cos_{tau}", 1.0),
            "inv_l1":      inv_profile.get(f"inv_l1_{tau}", 1.0),
            "n_alive_tau": inv_profile.get(f"n_alive_{tau}", 0),
            "n_comp_tau":  inv_profile.get(f"n_comp_{tau}", 0),
            "fft_power":   inv_profile.get(f"fft_power_{tau}", 0.0),
            # energy
            "energy_cost": energy_cost,
            "sdr_v2_partial": sdr_v2_partial,
        })
    return rows


def run_main_experiment(n_jobs: int = -1) -> pd.DataFrame:
    all_types = ["random"] + PATTERN_NAMES
    tasks = [
        (obj, i * 10_000 + trial)
        for i, obj in enumerate(all_types)
        for trial in range(N_INSTANCES)
    ]
    print(f"  Main: {len(tasks)} simulations × {T} steps …")
    nested = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_one_main)(obj, seed) for obj, seed in tasks
    )
    return pd.DataFrame([r for sub in nested for r in sub])


# ── Experiment 2: mixed ───────────────────────────────────────────────────────

def _run_one_mixed(obj_type: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pattern = PATTERNS[obj_type]

    mixed_grid, pat_only_grid = place_on_noise(GRID_SIZE, pattern, NOISE_DENSITY, rng)
    mixed_states,  _ = run(mixed_grid,    T, (0, T))
    struct_states, _ = run(pat_only_grid, T, (0, T))

    overlap, survived = _structure_survival(mixed_states[T], struct_states[T])

    # compute SR_shift on mixed (for correlation with survival)
    sr = compute_sr_all(mixed_states[0], mixed_states[T])
    return {
        "object_type": obj_type,
        "trial_id":    seed,
        "survived":    survived,
        "overlap_frac": overlap,
        "sr_shift":    sr["sr_shift"],
        "sr_shape":    sr["sr_shape"],
        "sr_center":   sr["sr_center"],
        "sr_comp":     sr["sr_comp"],
        "sr_mean":     sr["sr_mean"],
        "nmi_old":     sr["nmi_old"],
    }


def run_mixed_experiment(n_jobs: int = -1) -> pd.DataFrame:
    tasks = [
        (obj, 500_000 + i * 10_000 + trial)
        for i, obj in enumerate(PATTERN_NAMES)
        for trial in range(N_MIXED)
    ]
    print(f"  Mixed: {len(tasks)} simulations × {T} steps …")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_one_mixed)(obj, seed) for obj, seed in tasks
    )
    return pd.DataFrame(results)


# ── Experiment 3: Glider Translation Test ────────────────────────────────────

def run_translation_test(n_trials: int = 200) -> pd.DataFrame:
    """
    Validation test: create a Glider, artificially shift it by (20, 20),
    compare old NMI vs SR_shift.

    Expected:
      NMI(X, shift(X, 20, 20))  ≈ 0   (different cell positions)
      SR_shift(X, shift(X, 20, 20)) ≈ 1.0  (FFT finds (20,20), aligns perfectly)

    Also tests on random fields as negative control:
      SR_shift(random, shift(random, 20, 20)) should also be ≈ 1.0
      BUT SR_shift(random_0, random_evolved_200) ≈ 0  (not just a shift)

    This validates that SR_shift does NOT simply return 1.0 for everything —
    it correctly identifies when the relationship is a pure translation.
    """
    SHIFT = 20
    rows = []

    for obj_type in ["glider", "block", "random"]:
        for trial in range(n_trials):
            rng = np.random.default_rng(900_000 + trial)

            if obj_type == "random":
                x0 = random_config(GRID_SIZE, 0.5, rng)
            else:
                x0 = place_on_empty(GRID_SIZE, PATTERNS[obj_type], rng)

            # Case A: X vs artificially shifted X (pure translation)
            x_shifted = np.roll(np.roll(x0, SHIFT, axis=0), SHIFT, axis=1)
            sr_A = compute_sr_all(x0, x_shifted)

            # Case B: X vs evolved X (200 steps of GoL)
            states, _ = run(x0, T, (0, T))
            x_evolved = states[T]
            sr_B = compute_sr_all(x0, x_evolved)

            rows.append({
                "object_type":    obj_type,
                "trial_id":       trial,
                "case":           "A_shift_only",
                "nmi_old":        sr_A["nmi_old"],
                "sr_shift":       sr_A["sr_shift"],
                "sr_shape":       sr_A["sr_shape"],
                "sr_center":      sr_A["sr_center"],
                "sr_comp":        sr_A["sr_comp"],
                "detected_dy":    sr_A["sr_shift_dy"],
                "detected_dx":    sr_A["sr_shift_dx"],
                "expected_shift": SHIFT,
            })
            rows.append({
                "object_type":    obj_type,
                "trial_id":       trial,
                "case":           "B_evolved_200",
                "nmi_old":        sr_B["nmi_old"],
                "sr_shift":       sr_B["sr_shift"],
                "sr_shape":       sr_B["sr_shape"],
                "sr_center":      sr_B["sr_center"],
                "sr_comp":        sr_B["sr_comp"],
                "detected_dy":    sr_B["sr_shift_dy"],
                "detected_dx":    sr_B["sr_shift_dx"],
                "expected_shift": -1,   # unknown / not applicable
            })

    return pd.DataFrame(rows)
