"""OracleGoalEncoder — копия из sdr_agent_v3_diagnostics для автономности модуля."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try current env first (reward_shaping has own env.py with these constants).
# Fall back to v3 only if not already importable.
try:
    from env import COLOR_TO_IDX, SIZE_TO_IDX, SHAPE_TO_IDX
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "sdr_agent_v3_transfer_action"))
    from env import COLOR_TO_IDX, SIZE_TO_IDX, SHAPE_TO_IDX

GOAL_DIM = 32
N_ATTR   = 8   # 3 colors + 2 sizes + 3 shapes


def attrs_to_onehot(colors, sizes, shapes, device=None) -> torch.Tensor:
    B    = len(colors)
    feat = torch.zeros(B, N_ATTR)
    for i, (c, sz, sh) in enumerate(zip(colors, sizes, shapes)):
        feat[i, COLOR_TO_IDX[c]]      = 1.0
        feat[i, 3 + SIZE_TO_IDX[sz]]  = 1.0
        feat[i, 5 + SHAPE_TO_IDX[sh]] = 1.0
    return feat if device is None else feat.to(device)


class OracleGoalEncoder(nn.Module):
    """
    Frozen random projection: 8-dim attribute one-hot → 32-dim goal_z.
    Работает для всех 18 комбо (включая holdout). Не обучается.
    """
    name = "oracle"

    def __init__(self, goal_dim: int = GOAL_DIM, seed: int = 42):
        super().__init__()
        gen  = torch.Generator().manual_seed(seed)
        proj = torch.randn(N_ATTR, goal_dim, generator=gen) / (N_ATTR ** 0.5)
        self.register_buffer("proj", proj)

    def forward(self, colors, sizes, shapes) -> torch.Tensor:
        feat = attrs_to_onehot(colors, sizes, shapes, device=self.proj.device)
        return F.normalize(feat @ self.proj, dim=-1)


class RandomGoalEncoder(nn.Module):
    """
    Pure random noise goal_z on every call — no useful goal information.
    Baseline: Oracle ≈ SDR >> Random confirms both provide genuine goal signal,
    not just "something to condition on."
    """
    name = "random"

    def __init__(self, goal_dim: int = GOAL_DIM):
        super().__init__()
        self.goal_dim = goal_dim

    def forward(self, colors, sizes, shapes) -> torch.Tensor:
        B = len(colors)
        return F.normalize(torch.randn(B, self.goal_dim), dim=-1)
