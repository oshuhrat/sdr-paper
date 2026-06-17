"""
Synthetic 64×64 RGB shape dataset for SDR-L3 testing.

18 object types = 3 shapes × 3 colors × 2 sizes.

Zero-shot holdout (never seen during training):
  red_large_circle
  green_large_square
  blue_large_triangle

These 3 combos are used ONLY for the zero-shot composition test.
The remaining 15 combos form the training set.

Images are generated on-the-fly from seeds to keep memory usage low.
"""
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from typing import Optional

# ── Attribute definitions ──────────────────────────────────────────────────────

SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red":   (220, 60,  60),
    "green": (60,  200, 60),
    "blue":  (60,  60,  220),
}
SIZES  = {"small": 10, "large": 24}

IMG_SIZE = 64
BG_COLOR = (240, 240, 240)

# The 3 combos withheld from training — used only for zero-shot evaluation
HOLDOUT = frozenset([
    ("red",   "large", "circle"),
    ("green", "large", "square"),
    ("blue",  "large", "triangle"),
])


# ── Image generation ──────────────────────────────────────────────────────────

def _draw_shape(draw: ImageDraw.Draw, cx: int, cy: int, r: int,
                shape: str, color: tuple):
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        draw.polygon([
            (cx,     cy - r),
            (cx - r, cy + r),
            (cx + r, cy + r),
        ], fill=color)


def make_image(
    color: str,
    size:  str,
    shape: str,
    rng:   np.random.Generator,
) -> np.ndarray:
    """
    Generate a single 64×64 RGB image.

    Returns float32 array (H, W, 3) in [0, 1].
    Object is placed at a random position, fully inside the frame.
    """
    r = SIZES[size]
    margin = r + 4
    cx = int(rng.integers(margin, IMG_SIZE - margin))
    cy = int(rng.integers(margin, IMG_SIZE - margin))

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
    draw = ImageDraw.Draw(img)
    _draw_shape(draw, cx, cy, r, shape, COLORS[color])
    return np.array(img, dtype=np.float32) / 255.0


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShapeDataset(Dataset):
    """
    Builds metadata list (combo + seed) at init; generates images on __getitem__.

    holdout=True  → training set  (15 combos, excludes HOLDOUT)
    holdout=False → zero-shot set (3 holdout combos only)
    """

    def __init__(
        self,
        n_per_combo: int = 300,
        holdout:     bool = True,
        seed:        int  = 0,
        transform    = None,
    ):
        self.transform   = transform
        self._meta: list[dict] = []

        base_seed = seed
        idx = 0
        for color in COLORS:
            for size in SIZES:
                for shape in SHAPES:
                    combo = (color, size, shape)
                    is_holdout = combo in HOLDOUT
                    if holdout and is_holdout:
                        continue
                    if not holdout and not is_holdout:
                        continue
                    for k in range(n_per_combo):
                        self._meta.append({
                            "color": color,
                            "size":  size,
                            "shape": shape,
                            "combo": f"{color}_{size}_{shape}",
                            "seed":  base_seed + idx,
                        })
                        idx += 1

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(self, idx: int):
        m = self._meta[idx]
        rng = np.random.default_rng(m["seed"])
        img = make_image(m["color"], m["size"], m["shape"], rng)
        t   = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        if self.transform is not None:
            t = self.transform(t)
        return t, m["color"], m["size"], m["shape"], m["combo"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def all_combos(holdout: bool = True) -> list[tuple]:
    """Return list of (color, size, shape) combos for the given split."""
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
