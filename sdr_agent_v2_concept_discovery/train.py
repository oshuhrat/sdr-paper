"""
Contrastive training loop for SDR-Agent v2.

Training uses ONLY the 15 non-holdout combos.
Zero-shot test combos (red_large_circle, etc.) are never seen.

Positive pair = same image with two independent augmentations.
Negative pairs = all other images in the batch.
"""
import time
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model import SDRAgent
from augmentations import augment_pair
from loss import nt_xent_loss
from dataset import ShapeDataset


class _PairDataset(torch.utils.data.Dataset):
    """Wraps ShapeDataset; returns (view1, view2, combo_label)."""
    def __init__(self, base: ShapeDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, color, size, shape, combo = self.base[idx]
        v1, v2 = augment_pair(img)
        return v1, v2, combo


def train(
    n_per_combo:       int   = 300,
    n_epochs:          int   = 30,
    batch_size:        int   = 128,
    embed_dim:         int   = 128,
    proj_dim:          int   = 64,
    temperature:       float = 0.5,
    lr:                float = 3e-4,
    weight_decay:      float = 1e-4,
    device:            str   = "cpu",
    verbose:           bool  = True,
    seed:              int   = 42,
    checkpoint_epochs: list  = None,  # e.g. [5, 30] → saves checkpoint_ep5_seed42.pth
) -> tuple:
    """Train and return (model, history_df)."""
    base = ShapeDataset(n_per_combo=n_per_combo, holdout=True, seed=0)
    ds   = _PairDataset(base)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    model    = SDRAgent(embed_dim=embed_dim, proj_dim=proj_dim, in_channels=3).to(device)
    optim    = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched    = CosineAnnealingLR(optim, T_max=n_epochs, eta_min=lr * 0.01)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_train  = len(base)
    n_combos = len(set(m["combo"] for m in base._meta))
    if verbose:
        print(f"  SDRAgent params={n_params:,}  embed={embed_dim}  "
              f"τ={temperature}  bs={batch_size}")
        print(f"  Training: {n_combos} combos × {n_per_combo} images "
              f"= {n_train:,}  |  {len(loader)} batches/epoch")

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot_loss = tot_pos = tot_neg = 0.0
        n_bat = 0
        for v1, v2, _ in loader:
            v1, v2 = v1.to(device), v2.to(device)
            p1 = model.project(v1)
            p2 = model.project(v2)
            loss, stats = nt_xent_loss(p1, p2, temperature)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot_loss += loss.item(); tot_pos += stats["pos_sr"]
            tot_neg  += stats["neg_sr"]; n_bat += 1
        sched.step()
        el = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "loss":   tot_loss / n_bat,
            "pos_sr": tot_pos  / n_bat,
            "neg_sr": tot_neg  / n_bat,
            "lr":     sched.get_last_lr()[0],
            "time_s": el,
        }
        history.append(row)
        if verbose:
            print(f"  epoch {epoch:3d}/{n_epochs}  "
                  f"loss={row['loss']:.4f}  SR+={row['pos_sr']:.4f}  "
                  f"SR-={row['neg_sr']:.4f}  [{el:.1f}s]")
        if checkpoint_epochs and epoch in checkpoint_epochs:
            path = f"checkpoint_ep{epoch}_seed{seed}.pth"
            torch.save(model.state_dict(), path)
            if verbose:
                print(f"  [checkpoint] saved {path}")
    return model, pd.DataFrame(history)
