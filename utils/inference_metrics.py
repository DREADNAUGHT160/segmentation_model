"""
Metrics for unlabelled real-world inference (no ground truth required).

When you run the model on your own images/videos without annotation labels,
you cannot compute IoU or accuracy.  However you CAN measure:

  1. Confidence / Uncertainty
     - Shannon entropy per pixel (high = uncertain)
     - Mean max-softmax confidence (low = model is uncertain overall)
     - Low-confidence pixel fraction (fraction below a threshold)

  2. Class Coverage
     - Pixel fraction predicted as each class
     - Ground-class coverage (fraction of traversable area)

  3. Temporal Consistency  (video only)
     - Frame-to-frame prediction IoU (how stable are predictions)
     - Flicker rate (fraction of pixels that flip class between frames)

  4. Boundary Quality
     - Edge density (proportion of boundary pixels)
     - Thin-boundary ratio (sanity check for fragmented predictions)

  5. Throughput
     - Inference FPS

All functions return plain Python dicts / numpy arrays so they are easy
to log to W&B, save as JSON, or include in reports.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Confidence / Entropy
# ---------------------------------------------------------------------------


def compute_entropy_map(probs: np.ndarray) -> np.ndarray:
    """Pixel-wise Shannon entropy of the class probability distribution.

    High entropy = model is uncertain about that pixel.

    Args:
        probs: ``(H, W, C)`` float32 softmax probabilities.

    Returns:
        ``(H, W)`` float32 entropy in nats (range [0, ln C]).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(np.clip(probs, 1e-10, 1.0))
    return -np.sum(probs * log_p, axis=-1).astype(np.float32)


def compute_confidence_stats(
    probs: np.ndarray,
    low_conf_threshold: float = 0.5,
) -> dict[str, float]:
    """Scalar confidence statistics for a single image.

    Args:
        probs:               ``(H, W, C)`` softmax probabilities.
        low_conf_threshold:  Pixels with max-prob below this are "uncertain".

    Returns:
        Dict with:
        - ``mean_confidence``         — average max-softmax probability
        - ``min_confidence``          — lowest max-softmax probability pixel
        - ``low_conf_pixel_fraction`` — fraction with max-prob < threshold
        - ``mean_entropy``            — average Shannon entropy (nats)
        - ``max_entropy``             — peak entropy value in the image
        - ``entropy_std``             — std of entropy map
    """
    max_probs = probs.max(axis=-1)                  # (H, W)
    entropy   = compute_entropy_map(probs)           # (H, W)
    return {
        "mean_confidence":         float(max_probs.mean()),
        "min_confidence":          float(max_probs.min()),
        "low_conf_pixel_fraction": float((max_probs < low_conf_threshold).mean()),
        "mean_entropy":            float(entropy.mean()),
        "max_entropy":             float(entropy.max()),
        "entropy_std":             float(entropy.std()),
    }


# ---------------------------------------------------------------------------
# Class coverage
# ---------------------------------------------------------------------------


def compute_class_coverage(
    pred_mask: np.ndarray,
    num_classes: int,
    class_names: Optional[list[str]] = None,
    ground_class_indices: Optional[set[int]] = None,
) -> dict:
    """Fraction of pixels predicted as each class.

    Args:
        pred_mask:             ``(H, W)`` int array of class indices.
        num_classes:           Total number of classes.
        class_names:           Optional list of class name strings.
        ground_class_indices:  Set of class indices considered traversable.

    Returns:
        Dict with:
        - ``coverage_by_index``  — {class_index: fraction} for all classes
        - ``coverage_by_name``   — {class_name: fraction} (if names provided)
        - ``ground_coverage``    — total fraction of traversable ground
        - ``dominant_class``     — index of most-predicted class
    """
    total = pred_mask.size
    counts = np.bincount(pred_mask.ravel(), minlength=num_classes)
    fractions = counts / max(total, 1)

    result: dict = {
        "coverage_by_index": {int(i): float(fractions[i]) for i in range(num_classes)},
    }

    if class_names:
        result["coverage_by_name"] = {
            class_names[i]: float(fractions[i]) for i in range(min(num_classes, len(class_names)))
        }

    if ground_class_indices:
        result["ground_coverage"] = float(
            sum(fractions[i] for i in ground_class_indices if i < num_classes)
        )

    result["dominant_class"] = int(counts.argmax())
    return result


# ---------------------------------------------------------------------------
# Temporal consistency (video)
# ---------------------------------------------------------------------------


def compute_temporal_consistency(
    pred_sequence: list[np.ndarray],
    ignore_index: int = 0,
) -> dict[str, float]:
    """Measure frame-to-frame prediction stability for a video sequence.

    Runs over consecutive frame pairs.

    Args:
        pred_sequence:  List of ``(H, W)`` int class-index arrays.
        ignore_index:   Class to exclude from consistency measurement.

    Returns:
        Dict with:
        - ``mean_frame_iou``     — average IoU between consecutive frames
        - ``flicker_rate``       — fraction of pixels that change class
                                   between consecutive frames (excl. void)
        - ``stable_pixel_frac`` — complement of flicker_rate
    """
    if len(pred_sequence) < 2:
        return {"mean_frame_iou": float("nan"),
                "flicker_rate": float("nan"),
                "stable_pixel_frac": float("nan")}

    frame_ious: list[float] = []
    flicker_rates: list[float] = []

    for prev, curr in zip(pred_sequence[:-1], pred_sequence[1:]):
        # Flicker rate
        valid_mask = (prev != ignore_index) | (curr != ignore_index)
        if valid_mask.sum() == 0:
            continue
        changed = (prev[valid_mask] != curr[valid_mask])
        flicker_rates.append(float(changed.mean()))

        # Frame IoU — treat each frame as a set of class labels
        classes = set(prev[valid_mask].tolist()) | set(curr[valid_mask].tolist())
        classes.discard(ignore_index)
        if not classes:
            continue
        ious = []
        for c in classes:
            inter = ((prev == c) & (curr == c)).sum()
            union = ((prev == c) | (curr == c)).sum()
            if union > 0:
                ious.append(inter / union)
        if ious:
            frame_ious.append(float(np.mean(ious)))

    mean_iou  = float(np.mean(frame_ious))  if frame_ious   else float("nan")
    flicker   = float(np.mean(flicker_rates)) if flicker_rates else float("nan")
    return {
        "mean_frame_iou":    mean_iou,
        "flicker_rate":      flicker,
        "stable_pixel_frac": 1.0 - flicker if not np.isnan(flicker) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Boundary quality
# ---------------------------------------------------------------------------


def compute_boundary_stats(pred_mask: np.ndarray) -> dict[str, float]:
    """Analyse the density and quality of prediction boundaries.

    Uses horizontal and vertical pixel differences to find boundaries.

    Args:
        pred_mask: ``(H, W)`` int class-index array.

    Returns:
        Dict with:
        - ``edge_density``      — fraction of pixels on a class boundary
        - ``boundary_fragmentation`` — edge pixel / perimeter ratio proxy
    """
    h_diff = (pred_mask[:-1, :] != pred_mask[1:, :]).astype(np.uint8)
    v_diff = (pred_mask[:, :-1] != pred_mask[:, 1:]).astype(np.uint8)

    total = pred_mask.size
    edge_pixels = int(h_diff.sum()) + int(v_diff.sum())
    edge_density = edge_pixels / (2 * total)

    # Connected-component fragmentation (approximate — no scipy needed)
    # More edges relative to total boundary = more fragmented
    num_classes_present = len(np.unique(pred_mask))
    fragmentation = edge_density / max(num_classes_present, 1)

    return {
        "edge_density":           float(edge_density),
        "boundary_fragmentation": float(fragmentation),
    }


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------


@torch.no_grad()
def benchmark_fps(
    model: nn.Module,
    device: torch.device,
    image_size: tuple[int, int] = (512, 512),
    batch_size: int = 1,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> dict[str, float]:
    """Measure inference throughput.

    Args:
        model:       Model in eval mode.
        device:      Target device.
        image_size:  ``(H, W)`` input size.
        batch_size:  Batch size for the benchmark.
        num_warmup:  Warm-up iterations (not timed).
        num_runs:    Timed iterations.

    Returns:
        Dict with ``fps``, ``ms_per_image``, ``ms_per_batch``.
    """
    model.eval()
    h, w = image_size
    dummy = torch.randn(batch_size, 3, h, w, device=device)

    # Warm up
    for _ in range(num_warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_batch = 1000 * elapsed / num_runs
    ms_per_image = ms_per_batch / batch_size
    fps = 1000 / ms_per_image

    return {
        "fps":          round(fps, 2),
        "ms_per_image": round(ms_per_image, 2),
        "ms_per_batch": round(ms_per_batch, 2),
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def save_entropy_overlay(
    image: np.ndarray,
    probs: np.ndarray,
    save_path: str,
    alpha: float = 0.65,
) -> None:
    """Save an entropy heatmap blended over the original image.

    Args:
        image:     ``(H, W, 3)`` uint8 RGB image.
        probs:     ``(H, W, C)`` float softmax probabilities.
        save_path: Output PNG path.
        alpha:     Heatmap blend opacity (0 = image only, 1 = heatmap only).
    """
    from PIL import Image as PILImage
    import matplotlib.cm as cm

    entropy = compute_entropy_map(probs)
    # Normalise to [0, 1]
    max_ent = np.log(probs.shape[-1])  # maximum possible entropy
    entropy_norm = np.clip(entropy / (max_ent + 1e-8), 0, 1)

    # Map to colour using 'plasma' colormap
    cmap = cm.get_cmap("plasma")
    heatmap_rgba = cmap(entropy_norm)           # (H, W, 4)
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    blended = (image * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    PILImage.fromarray(blended).save(save_path)


def save_inference_report_charts(
    stats: dict,
    save_dir: str,
    class_names: Optional[list[str]] = None,
) -> None:
    """Save per-image inference metric charts to a directory.

    Args:
        stats:       Combined dict output from ``compute_confidence_stats``
                     + ``compute_class_coverage``.
        save_dir:    Directory where charts are saved.
        class_names: Optional list for labelling coverage chart.
    """
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    _STYLE = {
        "figure.facecolor": "#0f0f0f",
        "axes.facecolor": "#1a1a1a",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#dddddd",
        "axes.titlecolor": "#ffffff",
        "xtick.color": "#aaaaaa",
        "ytick.color": "#aaaaaa",
        "text.color": "#dddddd",
        "grid.color": "#2a2a2a",
    }

    # Class coverage bar chart
    coverage = stats.get("coverage_by_index", {})
    if coverage and class_names:
        items = [(class_names[k], v) for k, v in sorted(coverage.items())
                 if v > 0.001 and k != 0]
        if items:
            names, vals = zip(*items)
            with plt.rc_context(_STYLE):
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor("#0f0f0f")
                ax.set_facecolor("#1a1a1a")
                ax.bar(range(len(names)), vals, color="#00aaff", alpha=0.85)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
                ax.set_title("Predicted Class Coverage (unlabelled inference)")
                ax.grid(True, axis="y")
                fig.tight_layout()
                fig.savefig(out / "class_coverage.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
