"""
cxr_transforms.py

Defines torchvision transform pipelines for CXR images.

Two normalization modes, selected by the `backbone` argument:

  backbone="xrv"  (default — torchxrayvision DenseNet-121)
      Maps [0, 1] pixel values → [-1024, 1024] via (2·x - 1)·1024.
      Matches the Hounsfield-equivalent scale used during MIMIC-CXR pretraining
      in torchxrayvision. stats_path is not needed in this mode.

  backbone="imagenet"  (legacy — ImageNet ResNet-50)
      Applies per-channel Normalize with mean/std from stats_path (produced by
      fit_cxr_transforms.py). stats_path is required in this mode.

Usage:
    from scripts.cxr_transforms import get_cxr_transforms

    train_tfm = get_cxr_transforms("train", backbone="xrv")
    val_tfm   = get_cxr_transforms("val",   backbone="xrv")

    # In Dataset.__getitem__:
    img = Image.open(path).convert("RGB")
    tensor = train_tfm(img)   # shape (3, 224, 224), values in [-1024, 1024]

Transform pipelines:
    train : Resize(256) → RandomCrop(224) → RandomHorizontalFlip → ToTensor → Normalize
    val   : Resize(256) → CenterCrop(224) → ToTensor → Normalize
    test  : identical to val

Notes:
    - CXR images are grayscale but stored as / converted to RGB.
      The model receives a 3-channel tensor (all channels identical for grayscale).
      CXREncoder.forward averages the 3 channels to 1 before passing to the
      single-channel DenseNet backbone.
    - RandomHorizontalFlip is appropriate for CXR (left/right are anatomically
      equivalent for most conditions). Do NOT add RandomVerticalFlip — CXR
      orientation is meaningful (lungs at top, diaphragm at bottom).
    - All augmentations are minimal by design: contrastive loss is already
      hard to optimize with a noisy heuristic. Aggressive augmentation during
      training may prevent the encoder from learning the CXR structure at all.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import torch
import torchvision.transforms as T

# Standard dimensions
RESIZE_SIZE = 256
CROP_SIZE   = 224

Split    = Literal["train", "val", "test"]
Backbone = Literal["xrv", "imagenet"]


class XRVNormalize:
    """
    Maps a [0, 1] float tensor → [-1024, 1024].

    Formula: (2·x - 1)·1024
    Matches xrv.datasets.normalize(img, maxval=255) used during MIMIC-CXR
    pretraining. Top-level class (not lambda) so it is picklable by
    Windows DataLoader spawn workers.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (2.0 * x - 1.0) * 1024.0

    def __repr__(self) -> str:
        return "XRVNormalize()"


def get_cxr_transforms(
    split: Split,
    stats_path: Optional[str | Path] = None,
    backbone: Backbone = "xrv",
) -> T.Compose:
    """
    Build the torchvision Compose transform for a given split.

    Args:
        split:      "train", "val", or "test"
        stats_path: path to cxr_transform_stats.json — only needed when
                    backbone="imagenet". Ignored for backbone="xrv".
        backbone:   "xrv" (default) or "imagenet".

    Returns:
        torchvision.transforms.Compose
    """
    if backbone == "xrv":
        normalize = XRVNormalize()   
    elif backbone == "imagenet":
        if stats_path is None:
            raise ValueError(
                "stats_path is required when backbone='imagenet'.\n"
                "Generate cxr_transform_stats.json before using this option."
            )
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(
                f"CXR normalization stats not found: {stats_path}\n"
                "Fit CXR transform statistics before using this option."
            )
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        normalize = T.Normalize(mean=stats["mean"], std=stats["std"])
    else:
        raise ValueError(f"backbone must be 'xrv' or 'imagenet'. Got: {backbone!r}")

    if split == "train":
        return T.Compose([
            T.Resize(RESIZE_SIZE),           # resize shorter edge to 256
            T.RandomCrop(CROP_SIZE),         # random 224×224 crop
            T.RandomHorizontalFlip(p=0.5),   # safe for CXR
            T.ToTensor(),                    # PIL → float [0, 1], shape (3, H, W)
            normalize,                       # scale to target range
        ])
    elif split in ("val", "test"):
        return T.Compose([
            T.Resize(RESIZE_SIZE),
            T.CenterCrop(CROP_SIZE),
            T.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test'. Got: {split!r}")


def get_stats(stats_path: str | Path) -> dict:
    """Return the raw ImageNet stats dict (for inspection or logging)."""
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)
