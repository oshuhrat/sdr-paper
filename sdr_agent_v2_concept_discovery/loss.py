"""NT-Xent contrastive loss — identical to sdr_agent v1."""
import torch
import torch.nn.functional as F
from typing import Tuple


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    N  = z1.shape[0]
    z  = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    diag = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim  = sim.masked_fill(diag, -1e9)
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N,     device=z.device),
    ])
    loss = F.cross_entropy(sim, labels)
    with torch.no_grad():
        pos_sr = (z1 * z2).sum(dim=1).mean().item()
        cross  = torch.mm(z1, z2.T)
        mask   = torch.eye(N, device=z.device, dtype=torch.bool)
        neg_sr = cross.masked_fill(mask, 0).sum().item() / max(N * (N - 1), 1)
    return loss, {"pos_sr": pos_sr, "neg_sr": neg_sr}
