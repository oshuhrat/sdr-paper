"""
Linear probing: do embeddings at epoch 5 vs epoch 30 encode attributes?

For each checkpoint:
  - Extract z = model.encode(img) for all 18 combos (train + holdout)
  - Train LogisticRegression to predict: shape / color / size / combo
  - Report test accuracy (stratified 80/20 split within each combo)

Expected pattern (overfitting hypothesis):
  Epoch 5:  shape/color/size accuracy ~moderate, combo accuracy ~low
  Epoch 30: shape/color/size accuracy drops, combo accuracy rises
  → model trades attribute disentanglement for per-combo memorisation
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))

from model   import SDRAgent
from dataset import ShapeDataset


def extract_embeddings(model: SDRAgent, n_per_combo: int = 200,
                       device: str = "cpu") -> dict:
    """Extract z_real for all 18 combos (train + holdout)."""
    model.eval()

    all_z, all_color, all_size, all_shape, all_combo = [], [], [], [], []

    for holdout_flag in (True, False):  # True = train combos, False = holdout combos
        ds = ShapeDataset(n_per_combo=n_per_combo, holdout=holdout_flag, seed=0)
        with torch.no_grad():
            for img, color, size, shape, combo in ds:
                img_t = img.unsqueeze(0).to(device)
                z = model.encode(img_t).squeeze(0).cpu().numpy()
                all_z.append(z)
                all_color.append(color)
                all_size.append(size)
                all_shape.append(shape)
                all_combo.append(combo)

    return {
        "Z":     np.array(all_z),
        "color": all_color,
        "size":  all_size,
        "shape": all_shape,
        "combo": all_combo,
    }


def probe_attribute(Z: np.ndarray, labels: list, label_name: str) -> float:
    """Train LogisticRegression probe; return test accuracy."""
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        Z, y, test_size=0.2, random_state=42, stratify=y,
    )
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    n_classes = len(le.classes_)
    chance = 1.0 / n_classes
    print(f"    {label_name:<8} acc={acc:.4f}  chance={chance:.4f}  "
          f"Δ={acc - chance:+.4f}  classes={list(le.classes_)}")
    return acc


def run_probing(checkpoint_path: str, embed_dim: int = 128,
                n_per_combo: int = 200, device: str = "cpu") -> dict:
    print(f"\n  Loading {checkpoint_path} ...")
    model = SDRAgent(embed_dim=embed_dim, proj_dim=64, in_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    data = extract_embeddings(model, n_per_combo=n_per_combo, device=device)
    Z    = data["Z"]
    print(f"  Embeddings: {Z.shape}  norm_mean={np.linalg.norm(Z, axis=1).mean():.4f}")

    results = {}
    for attr in ("shape", "color", "size", "combo"):
        results[attr] = probe_attribute(Z, data[attr], attr)
    return results


def main():
    seed = 42
    checkpoints = {
        "epoch_5":  f"checkpoint_ep5_seed{seed}.pth",
        "epoch_30": f"checkpoint_ep30_seed{seed}.pth",
    }

    all_results = {}
    for label, path in checkpoints.items():
        if not Path(path).exists():
            print(f"  [MISSING] {path} — skipping")
            continue
        print(f"\n{'='*60}")
        print(f"  Probing: {label}  ({path})")
        print(f"{'='*60}")
        all_results[label] = run_probing(path)

    # ── Summary table ────────────────────────────────────────────────────────────
    if len(all_results) == 2:
        print(f"\n{'='*60}")
        print(f"  Linear Probing Summary (seed={seed})")
        print(f"{'='*60}")
        print(f"  {'Probe':<10} {'Epoch 5':>10} {'Epoch 30':>10} {'Δ(30-5)':>10}")
        print(f"  {'-'*44}")
        for attr in ("shape", "color", "size", "combo"):
            e5  = all_results["epoch_5"][attr]
            e30 = all_results["epoch_30"][attr]
            delta = e30 - e5
            mark = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "≈")
            print(f"  {attr:<10} {e5:>10.4f} {e30:>10.4f} {delta:>+10.4f}  {mark}")

        print()
        e5_attr  = np.mean([all_results["epoch_5"][a]  for a in ("shape", "color", "size")])
        e30_attr = np.mean([all_results["epoch_30"][a] for a in ("shape", "color", "size")])
        print(f"  Attr mean (shape+color+size): ep5={e5_attr:.4f}  ep30={e30_attr:.4f}")
        print(f"  Combo:                        ep5={all_results['epoch_5']['combo']:.4f}  "
              f"ep30={all_results['epoch_30']['combo']:.4f}")

        ep5_attr_up  = e5_attr  > e30_attr
        ep30_combo_up = all_results["epoch_30"]["combo"] > all_results["epoch_5"]["combo"]
        if ep5_attr_up and ep30_combo_up:
            print("\n  VERDICT: Overfitting confirmed — epoch 5 better for attributes,")
            print("           epoch 30 better for per-combo memorisation.")
            print("           SDR-L3 hypothesis: fast mode CCS > full-mode CCS is structural.")
        else:
            print("\n  VERDICT: Overfitting hypothesis NOT confirmed by probing.")
            print("           Investigate further.")


if __name__ == "__main__":
    main()
