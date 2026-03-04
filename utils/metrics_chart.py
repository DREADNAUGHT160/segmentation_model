"""
Post-training metrics visualisation.

Generates a full report of charts and tables saved to the run directory.
Called automatically at the end of training and evaluation.

Charts produced
---------------
training_curves.png     — loss (train+val), mIoU, pixel-accuracy, LR over epochs
per_class_iou.png       — horizontal bar chart, classes sorted by IoU descending
per_class_f1.png        — Precision / Recall / F1 grouped bar chart per class
confusion_matrix.png    — 25×25 normalised heatmap
class_distribution.png  — GT vs Prediction class pixel coverage
metrics_summary.txt     — human-readable text summary of all scalar metrics

All charts are saved to ``<run_dir>/metrics/``.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on HPC / headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

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
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.facecolor": "#222222",
    "legend.edgecolor": "#444444",
    "font.size": 9,
}
_ACCENT = "#00aaff"
_GREEN  = "#00cc88"
_RED    = "#ff4444"
_ORANGE = "#ffaa33"
_PURPLE = "#cc66ff"


def _dark_fig(*args, **kwargs):
    with plt.rc_context(_STYLE):
        return plt.subplots(*args, **kwargs)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(history: dict, save_dir: Path) -> Path:
    """Plot loss, mIoU, pixel-accuracy, and LR over training epochs.

    Args:
        history: Dict produced by ``train.py`` with keys:
                 ``epochs``, ``train_loss``, ``val_loss``,
                 ``val_miou``, ``val_pixel_acc``, ``lr``.
        save_dir: Directory to write the PNG.

    Returns:
        Path of the saved figure.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epochs", list(range(1, len(history["train_loss"]) + 1)))

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Training Curves", fontsize=13, color="white", y=0.98)
        fig.patch.set_facecolor("#0f0f0f")

        # --- Loss ---
        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], color=_ACCENT, label="Train loss", linewidth=1.5)
        if history.get("val_loss"):
            ax.plot(epochs, history["val_loss"], color=_ORANGE, label="Val loss",
                    linestyle="--", linewidth=1.5)
        ax.set_title("Cross-Entropy Loss"); ax.set_xlabel("Epoch"); ax.legend()
        ax.grid(True); ax.set_facecolor("#1a1a1a")

        # --- mIoU ---
        ax = axes[0, 1]
        if history.get("val_miou"):
            ax.plot(epochs, history["val_miou"], color=_GREEN, linewidth=1.8)
            best_epoch = epochs[int(np.argmax(history["val_miou"]))]
            best_val   = max(history["val_miou"])
            ax.axvline(best_epoch, color="#ffffff", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.annotate(f"best {best_val:.4f}", xy=(best_epoch, best_val),
                        xytext=(best_epoch + max(len(epochs) * 0.02, 0.5), best_val),
                        color=_GREEN, fontsize=8,
                        arrowprops=dict(arrowstyle="->", color=_GREEN, lw=0.8))
        ax.set_title("Validation mIoU"); ax.set_xlabel("Epoch"); ax.set_ylim(bottom=0)
        ax.grid(True); ax.set_facecolor("#1a1a1a")

        # --- Pixel accuracy ---
        ax = axes[1, 0]
        if history.get("val_pixel_acc"):
            ax.plot(epochs, history["val_pixel_acc"], color=_PURPLE, linewidth=1.8)
        ax.set_title("Validation Pixel Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(True); ax.set_facecolor("#1a1a1a")

        # --- Learning rate ---
        ax = axes[1, 1]
        if history.get("lr"):
            ax.semilogy(epochs, history["lr"], color=_ORANGE, linewidth=1.5)
        ax.set_title("Learning Rate"); ax.set_xlabel("Epoch")
        ax.grid(True); ax.set_facecolor("#1a1a1a")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = save_dir / "training_curves.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Per-class IoU bar chart
# ---------------------------------------------------------------------------


def plot_per_class_iou(
    iou_per_class: np.ndarray,
    class_names: list[str],
    save_dir: Path,
    title: str = "Per-Class IoU",
    filename: str = "per_class_iou.png",
) -> Path:
    """Horizontal bar chart of per-class IoU, sorted descending.

    Args:
        iou_per_class: Float array of shape ``(C,)``.  NaN = class absent.
        class_names:   List of ``C`` class name strings.
        save_dir:      Output directory.
        title:         Figure title.
        filename:      Output filename.

    Returns:
        Path of the saved figure.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Skip void (index 0) and sort by IoU
    valid = [(i, class_names[i], iou_per_class[i])
             for i in range(1, len(class_names))
             if not np.isnan(iou_per_class[i])]
    valid.sort(key=lambda x: x[2])   # ascending so the bar goes left→right

    indices, names, ious = zip(*valid) if valid else ([], [], [])
    ious = list(ious)
    names = [textwrap.shorten(n, width=18) for n in names]
    miou = float(np.nanmean(iou_per_class[1:])) if len(iou_per_class) > 1 else 0.0

    colors = [_GREEN if v >= miou else _ORANGE if v >= miou * 0.5 else _RED
              for v in ious]

    with plt.rc_context(_STYLE):
        fig_h = max(5, len(names) * 0.35 + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        fig.patch.set_facecolor("#0f0f0f")
        ax.set_facecolor("#1a1a1a")

        bars = ax.barh(range(len(names)), ious, color=colors, height=0.7, alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("IoU")
        ax.set_xlim(0, 1.1)
        ax.set_title(f"{title}  (mIoU = {miou:.4f})")
        ax.axvline(miou, color="#ffffff", linestyle="--", linewidth=0.9,
                   alpha=0.6, label=f"mIoU={miou:.4f}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x")

        # Value labels
        for bar, val in zip(bars, ious):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7, color="#dddddd")

        fig.tight_layout()
        out = save_dir / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------


def compute_prf_from_confusion(
    conf_matrix: np.ndarray,
    ignore_index: int = 0,
) -> dict[str, np.ndarray]:
    """Compute per-class Precision, Recall, and F1 from a confusion matrix.

    Args:
        conf_matrix:   ``(C, C)`` int64 array where ``conf[gt, pred]`` = count.
        ignore_index:  Class index to exclude (typically 0 = void).

    Returns:
        Dict with keys ``"precision"``, ``"recall"``, ``"f1"`` each of shape ``(C,)``.
        Values are NaN for absent classes or the ignored class.
    """
    conf = conf_matrix.astype(np.float64)
    C = conf.shape[0]
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp   # predicted as class c but actually something else
    fn = conf.sum(axis=1) - tp   # actually class c but predicted as something else

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        recall    = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        f1        = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            np.nan,
        )

    if 0 <= ignore_index < C:
        precision[ignore_index] = np.nan
        recall[ignore_index]    = np.nan
        f1[ignore_index]        = np.nan

    return {"precision": precision, "recall": recall, "f1": f1}


def plot_per_class_f1(
    prf: dict[str, np.ndarray],
    class_names: list[str],
    save_dir: Path,
) -> Path:
    """Grouped bar chart — Precision / Recall / F1 per class."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build table of present (non-NaN F1) classes
    rows = []
    for i, name in enumerate(class_names):
        if i == 0:
            continue
        f1 = prf["f1"][i]
        if np.isnan(f1):
            continue
        rows.append((name, prf["precision"][i], prf["recall"][i], f1))

    if not rows:
        return save_dir / "per_class_f1.png"

    rows.sort(key=lambda r: r[3])   # sort by F1 ascending
    names, precs, recs, f1s = zip(*rows)
    names = [textwrap.shorten(n, width=16) for n in names]
    x = np.arange(len(names))
    w = 0.26

    with plt.rc_context(_STYLE):
        fig_h = max(5, len(names) * 0.35 + 2)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        fig.patch.set_facecolor("#0f0f0f")
        ax.set_facecolor("#1a1a1a")

        ax.barh(x - w, precs, w, label="Precision", color=_ACCENT,  alpha=0.85)
        ax.barh(x,     recs,  w, label="Recall",    color=_GREEN,   alpha=0.85)
        ax.barh(x + w, f1s,   w, label="F1",        color=_ORANGE,  alpha=0.85)

        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Score"); ax.set_xlim(0, 1.15)
        ax.set_title("Per-Class Precision / Recall / F1")
        ax.legend(fontsize=8); ax.grid(True, axis="x")

        fig.tight_layout()
        out = save_dir / "per_class_f1.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list[str],
    save_dir: Path,
    title: str = "Normalised Confusion Matrix",
) -> Path:
    """Heatmap of the row-normalised confusion matrix (recall per cell).

    For 25 classes the matrix is 25×25, which can be dense.  Row
    normalisation shows recall — i.e. given ground-truth class R, how
    often was it predicted as class C.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    C = conf_matrix.shape[0]

    # Row-normalise (recall view)
    row_sum = conf_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sum > 0, conf_matrix / row_sum, 0.0)

    # Short class names for axis ticks
    short = [textwrap.shorten(n, width=10) for n in class_names]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(14, 12))
        fig.patch.set_facecolor("#0f0f0f")
        ax.set_facecolor("#1a1a1a")

        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Recall fraction")

        ax.set_xticks(range(C))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(C))
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
        ax.set_title(title)

        # Annotate diagonal (per-class recall)
        for i in range(C):
            val = norm[i, i]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(i, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

        fig.tight_layout()
        out = save_dir / "confusion_matrix.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------


def plot_class_distribution(
    gt_counts: np.ndarray,
    pred_counts: np.ndarray,
    class_names: list[str],
    save_dir: Path,
) -> Path:
    """Side-by-side bar chart of GT vs Prediction pixel counts per class."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Exclude void (index 0) and empty classes
    present = [i for i in range(1, len(class_names))
               if gt_counts[i] > 0 or pred_counts[i] > 0]
    if not present:
        return save_dir / "class_distribution.png"

    names  = [textwrap.shorten(class_names[i], width=14) for i in present]
    gt_f   = gt_counts[present] / max(gt_counts[present].sum(), 1)
    pred_f = pred_counts[present] / max(pred_counts[present].sum(), 1)
    x = np.arange(len(present))
    w = 0.38

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(10, len(present) * 0.55 + 2), 5))
        fig.patch.set_facecolor("#0f0f0f")
        ax.set_facecolor("#1a1a1a")

        ax.bar(x - w / 2, gt_f,   w, label="Ground Truth", color=_ACCENT, alpha=0.85)
        ax.bar(x + w / 2, pred_f, w, label="Prediction",   color=_ORANGE, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.set_ylabel("Pixel fraction"); ax.set_title("Class Distribution: GT vs Prediction")
        ax.legend(fontsize=9); ax.grid(True, axis="y")

        fig.tight_layout()
        out = save_dir / "class_distribution.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Frequency-weighted IoU
# ---------------------------------------------------------------------------


def frequency_weighted_iou(
    iou_per_class: np.ndarray,
    gt_counts: np.ndarray,
    ignore_index: int = 0,
) -> float:
    """FWIoU — each class weighted by its pixel frequency in the GT."""
    total = gt_counts.sum()
    if total == 0:
        return float("nan")
    valid = [i for i in range(len(iou_per_class))
             if i != ignore_index and not np.isnan(iou_per_class[i]) and gt_counts[i] > 0]
    if not valid:
        return float("nan")
    weights = gt_counts[valid] / total
    return float(np.sum(weights * iou_per_class[valid]))


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def write_metrics_summary(
    results: dict,
    prf: dict[str, np.ndarray],
    class_names: list[str],
    save_dir: Path,
    extra: Optional[dict] = None,
) -> Path:
    """Write a human-readable text file summarising all scalar metrics."""
    save_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def sep(c="=", n=56):
        lines.append(c * n)

    sep()
    lines.append("  SEGMENTATION METRICS REPORT")
    sep()
    lines.append("")

    if extra:
        for k, v in extra.items():
            lines.append(f"  {k:<24} {v}")
        lines.append("")

    sep("-")
    lines.append("  AGGREGATE METRICS")
    sep("-")
    lines.append(f"  mIoU (mean over present classes): {results.get('mIoU', float('nan')):.4f}")
    lines.append(f"  Pixel accuracy:                   {results.get('pixel_accuracy', float('nan')):.4f}")
    lines.append(f"  Mean class accuracy:              {results.get('mean_class_acc', float('nan')):.4f}")
    fwiou = results.get("fwiou", float("nan"))
    lines.append(f"  Frequency-weighted IoU:           {fwiou:.4f}")
    mean_f1 = float(np.nanmean(prf["f1"][1:]))
    lines.append(f"  Mean F1 (excl. void):             {mean_f1:.4f}")
    lines.append("")

    sep("-")
    lines.append("  PER-CLASS METRICS")
    sep("-")
    header = f"  {'Class':<22} {'IoU':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'GT%':>6}"
    lines.append(header)
    lines.append("  " + "-" * 54)

    iou  = results.get("iou_per_class",  np.full(len(class_names), np.nan))
    acc  = results.get("acc_per_class",  np.full(len(class_names), np.nan))
    gt_c = results.get("gt_counts",      np.zeros(len(class_names)))
    total_px = max(gt_c.sum(), 1)

    for i, name in enumerate(class_names):
        if i == 0:
            continue
        def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "  N/A"
        gt_pct = f"{100 * gt_c[i] / total_px:.1f}%" if gt_c[i] > 0 else "  0.0%"
        lines.append(
            f"  {name:<22} {fmt(iou[i]):>6}  {fmt(prf['precision'][i]):>6}"
            f"  {fmt(prf['recall'][i]):>6}  {fmt(prf['f1'][i]):>6}  {gt_pct:>6}"
        )

    lines.append("")
    sep()

    text = "\n".join(lines)
    out = save_dir / "metrics_summary.txt"
    out.write_text(text)
    return out


# ---------------------------------------------------------------------------
# Master report generator
# ---------------------------------------------------------------------------


def generate_metrics_report(
    run_dir: Path,
    results: dict,
    conf_matrix: np.ndarray,
    class_names: list[str],
    history: Optional[dict] = None,
    extra_info: Optional[dict] = None,
) -> list[Path]:
    """Generate the complete metrics report for a training/evaluation run.

    Call this at the end of ``train.py`` or ``evaluate.py``.

    Args:
        run_dir:      The run's root directory (e.g. ``runs/my_exp/``).
        results:      Dict from ``SegmentationMetrics.compute()``, augmented
                      with ``"gt_counts"`` (per-class GT pixel counts).
        conf_matrix:  Raw ``(C, C)`` confusion matrix from
                      ``SegmentationMetrics._confusion``.
        class_names:  List of C class name strings.
        history:      Training history dict (only available after training,
                      not evaluation).  Keys: see ``plot_training_curves``.
        extra_info:   Optional dict of extra key-value pairs to include in
                      the text summary (e.g. ``{"Run": "my_exp"}``).

    Returns:
        List of paths of all generated files.
    """
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    prf = compute_prf_from_confusion(conf_matrix, ignore_index=0)

    # Frequency-weighted IoU
    gt_counts = results.get("gt_counts", np.zeros(len(class_names)))
    results["fwiou"] = frequency_weighted_iou(
        results.get("iou_per_class", np.full(len(class_names), np.nan)),
        gt_counts,
    )

    # Training curves (only when history is provided)
    if history and history.get("train_loss"):
        generated.append(plot_training_curves(history, metrics_dir))

    # Per-class IoU bar
    generated.append(
        plot_per_class_iou(
            results.get("iou_per_class", np.full(len(class_names), np.nan)),
            class_names, metrics_dir,
        )
    )

    # Precision / Recall / F1
    generated.append(plot_per_class_f1(prf, class_names, metrics_dir))

    # Confusion matrix
    generated.append(plot_confusion_matrix(conf_matrix, class_names, metrics_dir))

    # Class distribution (only when GT counts available)
    pred_counts = conf_matrix.sum(axis=0)  # column sums = predicted-as counts
    generated.append(
        plot_class_distribution(gt_counts, pred_counts, class_names, metrics_dir)
    )

    # Text summary
    generated.append(
        write_metrics_summary(results, prf, class_names, metrics_dir, extra=extra_info)
    )

    # Save raw numbers as JSON for reproducibility
    json_data = {
        "mIoU": results.get("mIoU"),
        "pixel_accuracy": results.get("pixel_accuracy"),
        "mean_class_acc": results.get("mean_class_acc"),
        "fwiou": results.get("fwiou"),
        "mean_f1": float(np.nanmean(prf["f1"][1:])),
        "iou_per_class": {
            class_names[i]: float(v) if not np.isnan(v) else None
            for i, v in enumerate(results.get("iou_per_class", []))
        },
        "f1_per_class": {
            class_names[i]: float(v) if not np.isnan(v) else None
            for i, v in enumerate(prf["f1"])
        },
        "precision_per_class": {
            class_names[i]: float(v) if not np.isnan(v) else None
            for i, v in enumerate(prf["precision"])
        },
        "recall_per_class": {
            class_names[i]: float(v) if not np.isnan(v) else None
            for i, v in enumerate(prf["recall"])
        },
    }
    json_out = metrics_dir / "metrics.json"
    with open(json_out, "w") as f:
        json.dump(json_data, f, indent=2)
    generated.append(json_out)

    print(f"\nMetrics report saved to: {metrics_dir}")
    for p in generated:
        print(f"  {p.name}")

    return generated
