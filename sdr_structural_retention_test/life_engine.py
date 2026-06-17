"""Conway's Game of Life on a torus (cyclic boundary)."""
import numpy as np


def step(grid: np.ndarray) -> np.ndarray:
    p = np.pad(grid, 1, mode="wrap")
    n = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2]               + p[1:-1, 2:] +
        p[2:,  :-2] + p[2:,  1:-1] + p[2:,  2:]
    )
    return np.uint8((n == 3) | (grid.astype(bool) & (n == 2)))


def run(
    grid: np.ndarray,
    steps: int,
    snapshots: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], float]:
    snap_set = set(snapshots)
    states: dict[int, np.ndarray] = {}
    cur = grid.copy()
    total_diff = 0

    if 0 in snap_set:
        states[0] = cur.copy()

    for t in range(1, steps + 1):
        nxt = step(cur)
        total_diff += int(np.count_nonzero(cur ^ nxt))
        cur = nxt
        if t in snap_set:
            states[t] = cur.copy()

    return states, total_diff / steps
