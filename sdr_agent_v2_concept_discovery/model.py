"""
SDR-Agent encoder adapted for 64×64 RGB input.

Architecture identical to v1 but with:
  - in_channels = 3 (RGB vs grayscale)
  - extra MaxPool2d to handle 64px (v1 was 28px)

64×64 → MaxPool → 32×32 → MaxPool → 16×16 → MaxPool → 8×8 → AvgPool → 128-dim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """(B, 3, 64, 64) → (B, embed_dim)  L2-normalised."""

    def __init__(self, embed_dim: int = 128, in_channels: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = nn.Sequential(
            _Block(in_channels, 32),
            nn.MaxPool2d(2),          # 32×32
            _Block(32, 64),
            nn.MaxPool2d(2),          # 16×16
            _Block(64, 128),
            nn.MaxPool2d(2),          # 8×8
            _Block(128, 128),
            nn.AdaptiveAvgPool2d(1),  # 1×1
            nn.Flatten(),             # 128
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)


class ProjectionHead(nn.Module):
    """Training-only projection head. 128 → 64, L2-normalised."""

    def __init__(self, embed_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=1)


class SDRAgent(nn.Module):
    def __init__(self, embed_dim: int = 128, proj_dim: int = 64, in_channels: int = 3):
        super().__init__()
        self.encoder   = Encoder(embed_dim, in_channels)
        self.proj_head = ProjectionHead(embed_dim, proj_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_head(self.encoder(x))
