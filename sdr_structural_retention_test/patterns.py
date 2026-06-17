"""
GoL patterns: still lifes, oscillators, spaceships.
New in v2: LWSS (moves 2 right every 4 steps) and Pulsar (period 3).
"""
import numpy as np

PATTERNS: dict[str, np.ndarray] = {
    "block": np.array([[1, 1],
                       [1, 1]], dtype=np.uint8),

    "beehive": np.array([[0, 1, 1, 0],
                         [1, 0, 0, 1],
                         [0, 1, 1, 0]], dtype=np.uint8),

    "loaf": np.array([[0, 1, 1, 0],
                      [1, 0, 0, 1],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]], dtype=np.uint8),

    "boat": np.array([[1, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]], dtype=np.uint8),

    "blinker": np.array([[1, 1, 1]], dtype=np.uint8),

    "toad": np.array([[0, 1, 1, 1],
                      [1, 1, 1, 0]], dtype=np.uint8),

    "glider": np.array([[0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 1]], dtype=np.uint8),

    # LWSS: period 4, moves 2 cells right per period (c/2 eastward)
    # On 100x100 torus: after 200 steps it moves 100 cells = returns to start!
    "lwss": np.array([[0, 1, 0, 0, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0]], dtype=np.uint8),

    # Pulsar: period 3, 48 alive cells, 13×13 bounding box
    # At τ = 10,20,50,100,200 (all ≢ 0 mod 3): different phase than t=0
    # Great stress-test for metrics that claim to see "structural" retention
    "pulsar": np.array([
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    ], dtype=np.uint8),
}

PATTERN_NAMES = list(PATTERNS.keys())


# ── Structural categories (for report) ────────────────────────────────────────
CATEGORY: dict[str, str] = {
    "block":   "still_life",
    "beehive": "still_life",
    "loaf":    "still_life",
    "boat":    "still_life",
    "blinker": "oscillator_p2",
    "toad":    "oscillator_p2",
    "pulsar":  "oscillator_p3",
    "glider":  "spaceship",
    "lwss":    "spaceship",
    "random":  "random",
}


def place_on_empty(
    grid_size: int,
    pattern: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    ph, pw = pattern.shape
    r = int(rng.integers(0, grid_size))
    c = int(rng.integers(0, grid_size))
    for dr in range(ph):
        for dc in range(pw):
            grid[(r + dr) % grid_size, (c + dc) % grid_size] = pattern[dr, dc]
    return grid


def place_on_noise(
    grid_size: int,
    pattern: np.ndarray,
    noise_density: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    noise = (rng.random((grid_size, grid_size)) < noise_density).astype(np.uint8)
    pat_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    ph, pw = pattern.shape
    r = int(rng.integers(0, grid_size))
    c = int(rng.integers(0, grid_size))
    for dr in range(ph):
        for dc in range(pw):
            pat_grid[(r + dr) % grid_size, (c + dc) % grid_size] = pattern[dr, dc]
    mixed = np.clip(noise.astype(np.int16) + pat_grid, 0, 1).astype(np.uint8)
    return mixed, pat_grid


def random_config(
    grid_size: int,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return (rng.random((grid_size, grid_size)) < density).astype(np.uint8)
