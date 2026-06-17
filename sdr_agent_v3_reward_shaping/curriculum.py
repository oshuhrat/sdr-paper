"""
CurriculumScheduler — постепенное увеличение сложности задачи навигации.

Фазы (max_init_dist между агентом и целью, для GRID_SIZE=4):
  0: dist ≤ 1   — цель рядом (1 шаг), агент учит pickup.
  1: dist ≤ 2   — короткая навигация.
  2: dist ≤ 4   — средняя навигация.
  3: dist = None — полный 4×4 мир (max dist = 6). Финальная задача.

Переход: когда online SR за последние `window` роллаутов ≥ threshold.
"""
import numpy as np


class CurriculumScheduler:
    PHASES = [1, 2, None]   # max_init_dist per phase; None = unlimited (max dist in 3x3 = 4)

    def __init__(
        self,
        env,
        advance_threshold: float = 0.80,
        window:            int   = 5,
    ):
        self.env       = env
        self.threshold = advance_threshold
        self.window    = window
        self.phase_idx = 0
        self._sr_buf   = []

        env.max_init_dist = self.PHASES[0]

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def phase(self) -> int:
        return self.phase_idx

    @property
    def max_dist(self):
        return self.PHASES[self.phase_idx]

    @property
    def finished(self) -> bool:
        return self.PHASES[self.phase_idx] is None

    # ── Update ─────────────────────────────────────────────────────────────────

    def update(self, online_sr: float) -> bool:
        """
        Call after each rollout with that rollout's online SR.
        Returns True if curriculum advanced to next phase.
        """
        self._sr_buf.append(online_sr)

        if self.finished or len(self._sr_buf) < self.window:
            return False

        recent = float(np.mean(self._sr_buf[-self.window:]))
        if recent >= self.threshold:
            self.phase_idx        += 1
            self.env.max_init_dist = self.PHASES[self.phase_idx]
            self._sr_buf           = []
            return True

        return False

    def status(self) -> str:
        d = self.max_dist
        return f"phase={self.phase_idx}  max_dist={'∞' if d is None else d}"
