"""
Augmentations for synthetic shape images.

Design constraint: preserve color, shape, size identity.
Vary:  position (random shift), brightness (±15%), horizontal flip.

NOT used: color jitter, rotation beyond small angles, strong scale change.
Reason: color and shape ARE the semantic attributes we want to encode —
destroying them would prevent learning discriminative concept embeddings.
"""
import random
import torch
import torchvision.transforms.functional as TF
from typing import Tuple


IMG_SIZE = 64
PAD      = 10    # max pixel shift in each direction


def augment(img: torch.Tensor) -> torch.Tensor:
    """
    Single augmentation pass for a 64×64 RGB tensor (3, H, W).
    Applies: random crop (position shift) + brightness jitter.
    """
    # Random position shift via pad → crop
    padded = TF.pad(img, PAD, fill=0)             # (3, 84, 84)
    i = random.randint(0, 2 * PAD)
    j = random.randint(0, 2 * PAD)
    img = TF.crop(padded, i, j, IMG_SIZE, IMG_SIZE)

    # Brightness jitter (uniform ×factor, preserves hue/saturation)
    factor = random.uniform(0.85, 1.15)
    img    = torch.clamp(img * factor, 0.0, 1.0)

    return img


def augment_pair(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two independently augmented views of the same image (positive pair)."""
    return augment(img), augment(img)
