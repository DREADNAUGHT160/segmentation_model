"""
RUGD Dataset — Robot Unstructured Ground Dataset.

The dataset contains 7,545 images across 25 semantic classes for off-road /
unstructured terrain.  Official site: http://rugd.vision/

Annotation format (HuggingFace mirror)
---------------------------------------
Annotation PNGs are **RGB colourmap images** — each pixel's (R, G, B) value
encodes its class according to ``RUGD_annotation-colormap.txt``.  This module
automatically converts RGB → class index using ``_rgb_to_class_index()``.

Expected directory layout:
    RUGD/
    ├── RUGD_frames-with-annotations/
    │   ├── creek/
    │   │   ├── creek_00001.png   ← RGB image
    │   │   └── ...
    │   └── ...
    └── RUGD_annotations/
        ├── creek/
        │   ├── creek_00001.png   ← RGB colourmap annotation
        │   └── ...
        └── ...

Download sample:
    python data/download_rugd.py --output data/RUGD

Set ``data.root_dir`` in configs/config.yaml to point here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

# 25 RUGD semantic classes in index order (0–24).
# Source: RUGD_annotation-colormap.txt (official dataset metadata).
RUGD_CLASSES: list[str] = [
    "void",          # 0  — unlabelled / ignore
    "dirt",          # 1
    "sand",          # 2
    "grass",         # 3
    "tree",          # 4
    "pole",          # 5
    "water",         # 6
    "sky",           # 7
    "vehicle",       # 8
    "container",     # 9
    "asphalt",       # 10
    "gravel",        # 11
    "building",      # 12
    "mulch",         # 13
    "rock-bed",      # 14
    "log",           # 15
    "bicycle",       # 16
    "person",        # 17
    "fence",         # 18
    "bush",          # 19
    "sign",          # 20
    "rock",          # 21
    "bridge",        # 22
    "concrete",      # 23
    "picnic-table",  # 24
]

# Class indices considered traversable ground surfaces.
# Useful for post-inference ground extraction without retraining.
GROUND_CLASS_INDICES: set[int] = {
    RUGD_CLASSES.index(c)
    for c in ("dirt", "sand", "grass", "asphalt", "gravel", "mulch", "rock-bed", "water")
}

# RGB colormap — exact values from RUGD_annotation-colormap.txt.
# Shape: (25, 3), dtype uint8.
RUGD_COLORMAP: np.ndarray = np.array(
    [
        [0, 0, 0],        # 0  void
        [108, 64, 20],    # 1  dirt
        [255, 229, 204],  # 2  sand
        [0, 102, 0],      # 3  grass
        [0, 255, 0],      # 4  tree
        [0, 153, 153],    # 5  pole
        [0, 128, 255],    # 6  water
        [0, 0, 255],      # 7  sky
        [255, 255, 0],    # 8  vehicle
        [255, 0, 127],    # 9  container
        [64, 64, 64],     # 10 asphalt
        [255, 128, 0],    # 11 gravel
        [255, 0, 0],      # 12 building
        [153, 76, 0],     # 13 mulch
        [102, 102, 0],    # 14 rock-bed
        [102, 0, 0],      # 15 log
        [0, 255, 128],    # 16 bicycle
        [204, 153, 255],  # 17 person
        [102, 0, 204],    # 18 fence
        [255, 153, 204],  # 19 bush
        [0, 102, 102],    # 20 sign
        [153, 204, 255],  # 21 rock
        [102, 255, 255],  # 22 bridge
        [101, 101, 11],   # 23 concrete
        [114, 85, 47],    # 24 picnic-table
    ],
    dtype=np.uint8,
)


def _rgb_to_class_index(rgb_mask: np.ndarray) -> np.ndarray:
    """Convert an RGB colourmap annotation to a class-index mask.

    Each pixel's (R, G, B) value is matched against ``RUGD_COLORMAP`` to
    find its class index.  Unknown colours are mapped to 0 (void).

    Args:
        rgb_mask: ``(H, W, 3)`` uint8 array.

    Returns:
        ``(H, W)`` uint8 array with class indices in ``[0, 24]``.
    """
    h, w = rgb_mask.shape[:2]
    flat = rgb_mask.reshape(-1, 3).astype(np.int32)          # (N, 3)
    cmap = RUGD_COLORMAP.astype(np.int32)                     # (C, 3)
    # Squared L2 distance: (N, C)
    dists = np.sum((flat[:, None, :] - cmap[None, :, :]) ** 2, axis=2)
    return dists.argmin(axis=1).reshape(h, w).astype(np.uint8)

# ---------------------------------------------------------------------------
# Official train / val / test sequence splits
# Sourced from: Wigness et al., IROS 2019 (RUGD paper)
# ---------------------------------------------------------------------------
RUGD_SPLITS: dict[str, list[str]] = {
    "train": [
        "creek",
        "park-1",
        "park-8",
        "trail",
        "trail-3",
        "trail-4",
        "trail-5",
        "trail-6",
        "trail-9",
        "trail-10",
        "trail-11",
        "trail-12",
        "trail-14",
        "trail-15",
    ],
    "val": ["park-2", "trail-7"],
    "test": ["trail-7", "trail-13"],
}

# Pixel value used to ignore a class during loss computation.
IGNORE_INDEX: int = 255


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class RUGDDataset(Dataset):
    """PyTorch Dataset for RUGD semantic segmentation.

    Args:
        root_dir:   Path to the RUGD root directory (contains
                    ``RUGD_frames-with-annotations/`` and
                    ``RUGD_annotations/``).
        split:      One of ``"train"``, ``"val"``, or ``"test"``.
                    Controls which sequences are loaded.
        transform:  An **albumentations** ``Compose`` transform that
                    accepts ``image=`` and ``mask=`` keyword arguments
                    and returns a dict with the same keys.  Applied to
                    both image and label mask together so spatial
                    augmentations stay consistent.
        image_size: ``(H, W)`` tuple.  If provided, images/masks are
                    resized to this size *before* the albumentations
                    transform.  If ``None`` the original resolution is
                    kept.
        sequences:  Override the default split sequences with a custom
                    list of sequence names (e.g. ``["creek", "trail"]``).
    """

    IMAGES_DIR = "RUGD_frames-with-annotations"
    LABELS_DIR = "RUGD_annotations"

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Optional[tuple[int, int]] = None,
        sequences: Optional[list[str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size  # (H, W) or None

        seqs = sequences if sequences is not None else RUGD_SPLITS[split]
        self.samples = self._collect_samples(seqs)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split='{split}' in {self.root_dir}. "
                "Verify the RUGD directory layout described in this file's docstring."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_samples(self, sequences: list[str]) -> list[tuple[Path, Path]]:
        """Return a sorted list of (image_path, label_path) pairs."""
        samples: list[tuple[Path, Path]] = []
        img_root = self.root_dir / self.IMAGES_DIR
        lbl_root = self.root_dir / self.LABELS_DIR

        missing: list[str] = []
        for seq in sequences:
            img_seq_dir = img_root / seq
            lbl_seq_dir = lbl_root / seq

            if not img_seq_dir.is_dir() or not lbl_seq_dir.is_dir():
                missing.append(seq)
                continue

            for img_path in sorted(img_seq_dir.glob("*.png")):
                lbl_path = lbl_seq_dir / img_path.name
                if lbl_path.exists():
                    samples.append((img_path, lbl_path))

        if missing:
            import warnings
            warnings.warn(
                f"Skipping {len(missing)} sequences not found on disk: {missing}\n"
                f"  Run: python data/download_rugd.py --sequences {' '.join(missing)}\n"
                f"  to download them.",
                stacklevel=3,
            )

        return samples

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a sample dict with keys ``"image"`` and ``"mask"``.

        ``image``: FloatTensor of shape ``(3, H, W)``, values in [0, 1].
        ``mask``:  LongTensor of shape ``(H, W)``, values in ``[0, 23]``.
                   Pixels with value 0 (``void``) are typically ignored
                   by setting ``ignore_index=0`` in the loss function,
                   or you can set them to ``IGNORE_INDEX`` (255) here.
        """
        img_path, lbl_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))   # (H, W, 3) uint8
        raw_lbl = np.array(Image.open(lbl_path).convert("RGB"))  # (H, W, 3) uint8
        # Annotations are RGB colourmap images → convert to class indices
        mask = _rgb_to_class_index(raw_lbl)                      # (H, W)    uint8

        # Resize before augmentation if requested
        if self.image_size is not None:
            h, w = self.image_size
            image = np.array(Image.fromarray(image).resize((w, h), Image.BILINEAR))
            mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))

        # Apply albumentations (handles both image + mask consistently)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # FloatTensor (3, H, W) after ToTensorV2
            mask = augmented["mask"]    # LongTensor  (H, W) or (1, H, W) in albu 2.x
            # albumentations 2.x ToTensorV2 adds a channel dim to single-channel masks
            if mask.ndim == 3:
                mask = mask.squeeze(0)  # (1, H, W) → (H, W)
        else:
            # Default: HWC uint8 → CHW float in [0, 1]
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask.astype(np.int64))

        return {"image": image, "mask": mask.long()}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def compute_class_weights(
        root_dir: str | Path,
        split: str = "train",
        num_classes: int = 24,
        ignore_index: int = 0,
    ) -> torch.Tensor:
        """Compute inverse-frequency class weights over the training split.

        Slow on first call — cache the result.  The weights are
        suitable for ``torch.nn.CrossEntropyLoss(weight=...)``.

        Args:
            root_dir:     RUGD root directory.
            split:        Which split to compute weights from.
            num_classes:  Number of classes (24 for RUGD).
            ignore_index: Class index to exclude from weight computation.
        """
        dataset = RUGDDataset(root_dir=root_dir, split=split)
        counts = np.zeros(num_classes, dtype=np.int64)
        for _, lbl_path in dataset.samples:
            # Annotations are RGB colourmap images — convert first
            mask = _rgb_to_class_index(
                np.array(Image.open(lbl_path).convert("RGB"))
            ).ravel()
            for cls_idx in range(num_classes):
                if cls_idx != ignore_index:
                    counts[cls_idx] += (mask == cls_idx).sum()

        # Inverse frequency; classes with 0 pixels get weight 0
        with np.errstate(divide="ignore", invalid="ignore"):
            freq = counts / counts.sum()
            weights = np.where(freq > 0, 1.0 / freq, 0.0)
            weights /= weights.sum()  # normalise to sum to 1

        return torch.tensor(weights, dtype=torch.float32)
