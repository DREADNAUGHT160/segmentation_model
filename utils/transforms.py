"""
Albumentations transform pipelines for RUGD training and validation.

Albumentations applies spatial augmentations identically to both the
image and the segmentation mask, preventing misalignment.

All pipelines end with ``ToTensorV2`` which converts:
  - image: HWC uint8 → CHW float32 (values kept in [0, 255] unless
    Normalize is applied before it — we do apply Normalize)
  - mask:  HW uint8  → HW int64
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics — used because the backbone was pretrained on ImageNet.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.451, 0.224, 0.225)


def get_train_transforms(image_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """Return training augmentation pipeline.

    Augmentations applied
    ---------------------
    - Random horizontal flip (p=0.5)
    - Random scale crop (scale 0.5–2.0, then crop to image_size)
    - Colour jitter (brightness, contrast, saturation, hue)
    - Gaussian blur (occasional, p=0.2)
    - ImageNet normalisation → ToTensorV2

    Args:
        image_size: ``(H, W)`` target size after random crop.
    """
    h, w = image_size
    return A.Compose(
        [
            # Spatial augmentations (applied equally to image + mask)
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(
                size=(h, w),     # albumentations 2.x API
                scale=(0.4, 1.0),  # fraction of image area to crop
                ratio=(0.75, 1.333),
                interpolation=1,  # cv2.INTER_LINEAR
            ),
            A.Affine(                    # albumentations 2.x (replaces ShiftScaleRotate)
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                p=0.3,
            ),
            # Photometric augmentations (image only — mask unaffected)
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
                p=0.6,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(p=0.1),
            # Normalise then convert to tensor
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """Return deterministic validation / test transform pipeline.

    Only resizes and normalises — no random augmentations.

    Args:
        image_size: ``(H, W)`` target size.
    """
    h, w = image_size
    return A.Compose(
        [
            A.Resize(height=h, width=w, interpolation=1),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def denormalize(
    tensor: "torch.Tensor",
    mean: tuple[float, ...] = _IMAGENET_MEAN,
    std: tuple[float, ...] = _IMAGENET_STD,
) -> "torch.Tensor":
    """Reverse ImageNet normalisation for visualisation.

    Args:
        tensor: Float tensor of shape ``(3, H, W)`` or ``(B, 3, H, W)``.

    Returns:
        Tensor with values approximately in ``[0, 1]``.
    """
    import torch

    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 4:  # batch
        mean_t = mean_t[None, :, None, None]
        std_t = std_t[None, :, None, None]
    else:  # single image
        mean_t = mean_t[:, None, None]
        std_t = std_t[:, None, None]

    return tensor * std_t + mean_t
