"""
Visualisation utilities for segmentation results.

All functions operate on numpy arrays or PyTorch tensors and return
numpy uint8 RGB images that can be saved with PIL or OpenCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from data.rugd_dataset import RUGD_COLORMAP, RUGD_CLASSES


def colorize_mask(
    mask: np.ndarray | torch.Tensor,
    colormap: np.ndarray = RUGD_COLORMAP,
) -> np.ndarray:
    """Convert a class-index mask to an RGB colour image.

    Args:
        mask:      Integer array / tensor of shape ``(H, W)`` with values
                   in ``[0, num_classes)``.
        colormap:  ``(C, 3)`` uint8 array mapping class index → RGB.

    Returns:
        RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.int32)
    rgb = colormap[np.clip(mask, 0, len(colormap) - 1)]
    return rgb.astype(np.uint8)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray | torch.Tensor,
    alpha: float = 0.5,
    colormap: np.ndarray = RUGD_COLORMAP,
) -> np.ndarray:
    """Blend a coloured segmentation mask over an RGB image.

    Args:
        image:    Original RGB image, shape ``(H, W, 3)``, dtype ``uint8``
                  or ``float32`` in ``[0, 1]``.
        mask:     Class-index mask, shape ``(H, W)``.
        alpha:    Blend factor.  ``0`` = original image only;
                  ``1`` = coloured mask only.
        colormap: Class index → RGB mapping.

    Returns:
        Blended RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
    """
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    coloured = colorize_mask(mask, colormap)
    blended = (image * (1 - alpha) + coloured * alpha).astype(np.uint8)
    return blended


def save_comparison(
    image: np.ndarray,
    gt_mask: np.ndarray | torch.Tensor,
    pred_mask: np.ndarray | torch.Tensor,
    save_path: str | Path,
    colormap: np.ndarray = RUGD_COLORMAP,
) -> None:
    """Save a side-by-side comparison: image | ground truth | prediction.

    Args:
        image:      RGB image ``(H, W, 3)`` uint8.
        gt_mask:    Ground-truth class-index mask ``(H, W)``.
        pred_mask:  Predicted class-index mask ``(H, W)``.
        save_path:  Where to write the PNG.
        colormap:   Class index → RGB mapping.
    """
    gt_rgb = colorize_mask(gt_mask, colormap)
    pred_rgb = colorize_mask(pred_mask, colormap)

    # Horizontal concat: image | GT | prediction
    panel = np.concatenate([image, gt_rgb, pred_rgb], axis=1)
    Image.fromarray(panel).save(save_path)


def save_legend(
    save_path: str | Path,
    class_names: list[str] = RUGD_CLASSES,
    colormap: np.ndarray = RUGD_COLORMAP,
    swatch_size: int = 30,
) -> None:
    """Save a PNG legend image showing each class colour and name.

    Useful for presentations or reports.

    Args:
        save_path:   Output PNG path.
        class_names: List of class name strings.
        colormap:    ``(C, 3)`` RGB colormap.
        swatch_size: Height/width in pixels of each colour swatch.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(4, len(class_names) * 0.35 + 0.5))
        ax.axis("off")
        patches = [
            mpatches.Patch(
                color=[c / 255 for c in colormap[i].tolist()],
                label=f"{i}: {name}",
            )
            for i, name in enumerate(class_names)
        ]
        ax.legend(handles=patches, loc="center", ncol=1, frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        # Fallback: simple swatch strip without matplotlib
        n = len(class_names)
        legend = np.zeros((n * swatch_size, swatch_size + 200, 3), dtype=np.uint8)
        for i, color in enumerate(colormap[:n]):
            row_start = i * swatch_size
            legend[row_start : row_start + swatch_size, :swatch_size] = color
        Image.fromarray(legend).save(save_path)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor back to a uint8 numpy array.

    Args:
        tensor: Float tensor ``(3, H, W)`` with values roughly in
                ``[-2, 2]`` (post ImageNet normalisation).

    Returns:
        ``(H, W, 3)`` uint8 numpy array.
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    # De-normalise (approximate — fine for visualisation)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)
