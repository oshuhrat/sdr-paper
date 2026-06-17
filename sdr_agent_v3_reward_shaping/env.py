"""
GridWorld с dense reward shaping и поддержкой curriculum.

reward_mode="sparse"  → +1 только при pickup (как в v3)
reward_mode="dense"   → alpha*Δdist + beta*pickup - gamma*time_step

max_init_dist         → ограничение начального расстояния до цели (curriculum).
                        Устанавливается извне через env.max_init_dist = N.
                        None = без ограничений.
"""
import numpy as np

SHAPES = ["circle", "square", "triangle"]
COLORS = ["red",    "green",  "blue"]
SIZES  = ["small",  "large"]

SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}
SIZE_TO_IDX  = {s: i for i, s in enumerate(SIZES)}

HOLDOUT = frozenset([
    ("red",   "large", "circle"),
    ("green", "large", "square"),
    ("blue",  "large", "triangle"),
])


def all_combos(holdout: bool = True):
    result = []
    for color in COLORS:
        for size in SIZES:
            for shape in SHAPES:
                combo = (color, size, shape)
                is_h  = combo in HOLDOUT
                if holdout and not is_h:
                    result.append(combo)
                elif not holdout and is_h:
                    result.append(combo)
    return result


GRID_SIZE  = 3      # VIEW_SIZE=5, half=2 → from any 3x3 cell, sees entire grid (100%)
VIEW_SIZE  = 5
N_FEATURES = 1 + 3 + 2 + 3      # is_agent + color + size + shape
OBS_DIM    = VIEW_SIZE * VIEW_SIZE * N_FEATURES   # 225
N_ACTIONS  = 5
MAX_STEPS  = 100

_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
_PICKUP = 4


class GridWorld:
    def __init__(
        self,
        allowed_combos: list,
        n_distractors:  int   = 2,
        max_steps:      int   = MAX_STEPS,
        seed:           int   = None,
        reward_mode:    str   = "sparse",   # "sparse" | "dense"
        alpha:          float = 0.10,       # distance reward per unit
        beta:           float = 1.00,       # pickup bonus
        gamma:          float = 0.005,      # time penalty per step
        max_init_dist:  int   = None,       # curriculum: max start dist to goal
    ):
        self.allowed_combos = allowed_combos
        self.n_distractors  = n_distractors
        self.max_steps      = max_steps
        self.rng            = np.random.default_rng(seed)
        self.reward_mode    = reward_mode
        self.alpha          = alpha
        self.beta           = beta
        self.gamma          = gamma
        self.max_init_dist  = max_init_dist   # mutable: curriculum updates this

        self.agent_pos  = None
        self.goal_pos   = None
        self.goal_combo = None
        self.objects    = {}
        self.steps      = 0
        self.done       = False
        self._prev_dist = 0

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _random_empty(self, occupied):
        while True:
            pos = (int(self.rng.integers(0, GRID_SIZE)),
                   int(self.rng.integers(0, GRID_SIZE)))
            if pos not in occupied:
                return pos

    def _within_dist(self, center, max_dist, occupied):
        """Random empty cell within manhattan distance max_dist of center."""
        r, c = center
        cands = [
            (r + dr, c + dc)
            for dr in range(-max_dist, max_dist + 1)
            for dc in range(-max_dist, max_dist + 1)
            if 1 <= abs(dr) + abs(dc) <= max_dist
            and 0 <= r + dr < GRID_SIZE
            and 0 <= c + dc < GRID_SIZE
            and (r + dr, c + dc) not in occupied
        ]
        if cands:
            return cands[int(self.rng.integers(0, len(cands)))]
        return self._random_empty(occupied)   # fallback

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(self, goal_combo=None):
        if goal_combo is None:
            self.goal_combo = self.allowed_combos[
                int(self.rng.integers(0, len(self.allowed_combos)))
            ]
        else:
            self.goal_combo = goal_combo

        occupied = set()

        self.agent_pos = self._random_empty(occupied)
        occupied.add(self.agent_pos)

        if self.max_init_dist is not None:
            self.goal_pos = self._within_dist(self.agent_pos, self.max_init_dist, occupied)
        else:
            self.goal_pos = self._random_empty(occupied)
        occupied.add(self.goal_pos)

        self.objects = {
            self.goal_pos: {
                "color": self.goal_combo[0], "size": self.goal_combo[1],
                "shape": self.goal_combo[2], "is_goal": True,
            }
        }
        for _ in range(self.n_distractors):
            pos = self._random_empty(occupied)
            occupied.add(pos)
            while True:
                color = COLORS[int(self.rng.integers(0, 3))]
                size  = SIZES[ int(self.rng.integers(0, 2))]
                shape = SHAPES[int(self.rng.integers(0, 3))]
                if (color, size, shape) != self.goal_combo:
                    break
            self.objects[pos] = {"color": color, "size": size,
                                  "shape": shape, "is_goal": False}

        self.steps = 0
        self.done  = False
        self._prev_dist = self._manhattan(self.agent_pos, self.goal_pos)

        return self._obs(), {
            "goal_combo":       self.goal_combo,
            "optimal_distance": self._prev_dist,
        }

    def step(self, action: int):
        assert not self.done
        self.steps += 1
        success       = False
        wrong_pickup  = False

        if action in _DELTAS:
            dr, dc = _DELTAS[action]
            nr = max(0, min(GRID_SIZE - 1, self.agent_pos[0] + dr))
            nc = max(0, min(GRID_SIZE - 1, self.agent_pos[1] + dc))
            self.agent_pos = (nr, nc)
        elif action == _PICKUP:
            obj = self.objects.get(self.agent_pos)
            if obj and obj["is_goal"]:
                success   = True
                self.done = True
            elif obj and not obj["is_goal"]:
                wrong_pickup = True
                self.done    = True   # wrong pickup = episode over (forces goal-conditioned behavior)

        if self.steps >= self.max_steps:
            self.done = True

        curr_dist       = self._manhattan(self.agent_pos, self.goal_pos)
        delta           = self._prev_dist - curr_dist   # positive = closer
        self._prev_dist = curr_dist

        if self.reward_mode == "dense":
            reward = self.alpha * delta - self.gamma
            if success:
                reward += self.beta
            if wrong_pickup:
                reward -= 0.2   # penalty: discourages wrong-pickup loops (< +1.0 success reward)
        else:
            reward = 1.0 if success else 0.0

        return self._obs(), reward, self.done, {"success": success, "steps": self.steps}

    # ── Observation ────────────────────────────────────────────────────────────

    def _encode_cell(self, r: int, c: int) -> np.ndarray:
        feat = np.zeros(N_FEATURES, dtype=np.float32)
        if (r, c) == self.agent_pos:
            feat[0] = 1.0
        obj = self.objects.get((r, c))
        if obj:
            feat[1 + COLOR_TO_IDX[obj["color"]]] = 1.0
            feat[4 + SIZE_TO_IDX[obj["size"]]]   = 1.0
            feat[6 + SHAPE_TO_IDX[obj["shape"]]] = 1.0
        return feat

    def _obs(self) -> np.ndarray:
        ar, ac = self.agent_pos
        half   = VIEW_SIZE // 2
        obs    = np.zeros((VIEW_SIZE, VIEW_SIZE, N_FEATURES), dtype=np.float32)
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                r, c = ar + dr, ac + dc
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    obs[dr + half, dc + half] = self._encode_cell(r, c)
        return obs.flatten()
