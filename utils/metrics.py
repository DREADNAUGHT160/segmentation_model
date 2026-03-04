"""
Segmentation evaluation metrics.

Implements a streaming confusion-matrix accumulator so metrics can be
computed incrementally over large validation sets without storing all
predictions in memory.

Metrics computed
----------------
- Per-class IoU (Intersection over Union)
- Mean IoU (mIoU) — average over classes present in the ground truth
- Pixel accuracy — fraction of correctly classified pixels
- Mean class accuracy — average per-class recall
"""

from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    """Streaming segmentation metric tracker using a confusion matrix.

    Usage::

        metrics = SegmentationMetrics(num_classes=24, ignore_index=0)

        for batch in val_loader:
            preds = model(batch["image"]).argmax(dim=1)  # (B, H, W)
            metrics.update(preds, batch["mask"])

        results = metrics.compute()
        print(f"mIoU: {results['mIoU']:.4f}")
        metrics.reset()

    Args:
        num_classes:  Total number of segmentation classes.
        ignore_index: Class index to exclude from all metric computations.
                      Typically 0 (``void``) for RUGD.
    """

    def __init__(self, num_classes: int, ignore_index: int = 0) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ------------------------------------------------------------------
    # Core accumulation
    # ------------------------------------------------------------------

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate predictions into the confusion matrix.

        Args:
            preds:   Predicted class indices, shape ``(B, H, W)`` or
                     ``(H, W)``.  Values in ``[0, num_classes)``.
            targets: Ground-truth class indices, same shape as ``preds``.
        """
        preds = preds.detach().cpu().numpy().ravel()
        targets = targets.detach().cpu().numpy().ravel()

        # Mask ignored pixels
        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]

        # Clip to valid range (safety guard)
        preds = np.clip(preds, 0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)

        # Fast confusion matrix accumulation
        np.add.at(
            self._confusion,
            (targets, preds),
            1,
        )

    def reset(self) -> None:
        """Zero the confusion matrix for a new epoch."""
        self._confusion[:] = 0

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute(self) -> dict[str, float | np.ndarray]:
        """Compute all metrics from the accumulated confusion matrix.

        Returns a dict with:
        - ``"mIoU"``              (float) — mean IoU over present classes
        - ``"pixel_accuracy"``    (float) — overall pixel accuracy
        - ``"mean_class_acc"``    (float) — mean per-class recall
        - ``"iou_per_class"``     (ndarray, shape (C,)) — per-class IoU
        - ``"acc_per_class"``     (ndarray, shape (C,)) — per-class accuracy
        """
        conf = self._confusion.astype(np.float64)

        # True positives on the diagonal
        tp = np.diag(conf)

        # Per-class IoU: TP / (TP + FP + FN)
        row_sum = conf.sum(axis=1)  # all predicted-as-class (col axis = pred)
        col_sum = conf.sum(axis=0)  # all ground-truth-class (row axis = gt)
        union = row_sum + col_sum - tp

        with np.errstate(divide="ignore", invalid="ignore"):
            iou_per_class = np.where(union > 0, tp / union, np.nan)

        # Only average over classes that appear in the ground truth
        present_mask = row_sum > 0
        subset = iou_per_class[present_mask]
        miou = float(np.nanmean(subset)) if subset.size > 0 else float("nan")

        # Pixel accuracy
        total_pixels = conf.sum()
        pixel_acc = float(tp.sum() / total_pixels) if total_pixels > 0 else 0.0

        # Per-class recall (class accuracy)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc_per_class = np.where(row_sum > 0, tp / row_sum, np.nan)
        subset_acc = acc_per_class[present_mask]
        mean_class_acc = float(np.nanmean(subset_acc)) if subset_acc.size > 0 else float("nan")

        return {
            "mIoU": miou,
            "pixel_accuracy": pixel_acc,
            "mean_class_acc": mean_class_acc,
            "iou_per_class": iou_per_class,
            "acc_per_class": acc_per_class,
        }

    def class_report(self, class_names: list[str]) -> str:
        """Return a human-readable per-class IoU table.

        Args:
            class_names: List of class name strings, length == num_classes.
        """
        results = self.compute()
        iou = results["iou_per_class"]
        lines = [f"{'Class':<20} {'IoU':>8}"]
        lines.append("-" * 30)
        for i, name in enumerate(class_names):
            val = f"{iou[i]:.4f}" if not np.isnan(iou[i]) else "  N/A"
            lines.append(f"{name:<20} {val:>8}")
        lines.append("-" * 30)
        lines.append(f"{'mIoU':<20} {results['mIoU']:>8.4f}")
        lines.append(f"{'Pixel Acc':<20} {results['pixel_accuracy']:>8.4f}")
        return "\n".join(lines)
