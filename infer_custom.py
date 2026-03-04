"""
Inference on user-supplied images using a custom dataset config.

This script loads images from YOUR OWN directory (or video), runs the
segmentation model, saves visual outputs, and computes every metric that
is available *without* ground-truth labels:
  • Confidence / uncertainty (Shannon entropy, low-conf fraction)
  • Class coverage (which classes dominate the scene)
  • Temporal consistency (video only — frame flicker, frame IoU)
  • Boundary quality (edge density)
  • Throughput (FPS)

If you also have label masks, add label_dir to the config and you will
additionally get the full labelled metrics (mIoU, pixel accuracy, etc.)
identical to evaluate.py.

Usage
-----
    uv run python infer_custom.py --config configs/custom_dataset.yaml

Override checkpoint on the command line:
    uv run python infer_custom.py \\
        --config configs/custom_dataset.yaml \\
        --checkpoint runs/deeplabv3plus_rugd/checkpoints/best.pt

Override output directory:
    uv run python infer_custom.py \\
        --config configs/custom_dataset.yaml \\
        --output results/my_run

See configs/custom_dataset.yaml for all available options with inline docs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.custom_dataset import CustomImageDataset
from data.rugd_dataset import RUGD_CLASSES, RUGD_COLORMAP
from models.deeplabv3plus import load_checkpoint
from utils.inference_metrics import (
    benchmark_fps,
    compute_boundary_stats,
    compute_class_coverage,
    compute_confidence_stats,
    compute_temporal_consistency,
    save_entropy_overlay,
    save_inference_report_charts,
)
from utils.metrics import SegmentationMetrics
from utils.metrics_chart import generate_metrics_report
from utils.transforms import get_val_transforms
from utils.visualization import colorize_mask, overlay_mask, save_comparison


# ---------------------------------------------------------------------------
# Ground classes for RUGD (indices of traversable surface classes)
# ---------------------------------------------------------------------------

_GROUND_CLASS_NAMES = {
    "dirt", "sand", "grass", "asphalt", "gravel", "mulch", "rock-bed", "water"
}
_GROUND_CLASS_INDICES = {
    i for i, name in enumerate(RUGD_CLASSES) if name in _GROUND_CLASS_NAMES
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_checkpoint(cfg: dict, cli_checkpoint: Optional[str]) -> str:
    """Return checkpoint path from CLI arg (priority) or config."""
    if cli_checkpoint:
        return cli_checkpoint
    ckpt = cfg.get("model", {}).get("checkpoint")
    if not ckpt:
        raise ValueError(
            "No checkpoint specified. Pass --checkpoint or set model.checkpoint in config."
        )
    return ckpt


# ---------------------------------------------------------------------------
# Output savers
# ---------------------------------------------------------------------------


def _save_mask(pred_np: np.ndarray, out_path: Path) -> None:
    """Save raw class-index mask as a PNG (lossless, suitable for downstream use)."""
    from PIL import Image
    Image.fromarray(pred_np.astype(np.uint8)).save(out_path)


def _save_colour(pred_np: np.ndarray, out_path: Path) -> None:
    """Save colourised class mask."""
    from PIL import Image
    Image.fromarray(colorize_mask(pred_np)).save(out_path)


def _save_overlay(
    img_np: np.ndarray,
    pred_np: np.ndarray,
    out_path: Path,
    alpha: float,
) -> None:
    """Save mask blended over original image."""
    from PIL import Image
    Image.fromarray(overlay_mask(img_np, pred_np, alpha=alpha)).save(out_path)


def _unnorm_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor back to HWC uint8 for saving."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * std + mean
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    output_dir: Path,
    has_labels: bool,
) -> dict:
    """
    Run full inference loop.

    Returns a summary dict that will be written to ``inference_metrics.json``.
    """
    model.eval()

    infer_cfg  = cfg.get("infer", {})
    unlbl_cfg  = cfg.get("unlabelled_metrics", {})
    lbl_cfg    = cfg.get("labelled_metrics", {})
    num_classes = cfg["model"]["num_classes"]
    save_fmt   = infer_cfg.get("save_format", "overlay")
    alpha      = float(infer_cfg.get("overlay_alpha", 0.55))
    low_conf_t = float(unlbl_cfg.get("low_conf_threshold", 0.6))

    # Sub-directories
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    if save_fmt in ("colour", "all"):
        (output_dir / "colour").mkdir(exist_ok=True)
    if save_fmt in ("overlay", "all"):
        (output_dir / "overlay").mkdir(exist_ok=True)
    if unlbl_cfg.get("save_entropy_maps", False):
        (output_dir / "entropy").mkdir(exist_ok=True)
    if has_labels and lbl_cfg.get("save_comparisons", False):
        (output_dir / "comparisons").mkdir(exist_ok=True)

    # Accumulators
    all_pred_masks:  list[np.ndarray] = []  # for temporal consistency
    per_image_stats: list[dict]       = []
    seg_metrics: Optional[SegmentationMetrics] = (
        SegmentationMetrics(num_classes=num_classes, ignore_index=0)
        if has_labels else None
    )
    gt_counts = np.zeros(num_classes, dtype=np.int64)

    print(f"\nRunning inference on {len(loader.dataset)} images...")

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        stems  = batch["stem"]
        masks_gt = batch.get("mask")
        if masks_gt is not None:
            masks_gt = masks_gt.to(device, non_blocking=True)

        # Forward pass — get logits and softmax probabilities
        logits = model(images)                    # (B, C, H, W)
        probs_t = torch.softmax(logits, dim=1)   # (B, C, H, W)
        preds   = logits.argmax(dim=1)            # (B, H, W)

        # Update labelled metrics
        if seg_metrics is not None and masks_gt is not None:
            seg_metrics.update(preds, masks_gt)
            for c in range(num_classes):
                gt_counts[c] += int((masks_gt == c).sum().item())

        # Per-image processing
        for i in range(images.shape[0]):
            stem     = stems[i]
            img_np   = _unnorm_image(images[i])             # (H, W, 3) uint8
            pred_np  = preds[i].cpu().numpy().astype(np.uint8)  # (H, W)
            probs_np = probs_t[i].cpu().numpy()             # (C, H, W)
            probs_hw = probs_np.transpose(1, 2, 0)          # (H, W, C)

            all_pred_masks.append(pred_np)

            # --- save outputs ---
            if save_fmt in ("mask", "all"):
                _save_mask(pred_np, output_dir / "masks" / f"{stem}.png")
            if save_fmt in ("colour", "all"):
                _save_colour(pred_np, output_dir / "colour" / f"{stem}.png")
            if save_fmt in ("overlay", "all"):
                _save_overlay(img_np, pred_np, output_dir / "overlay" / f"{stem}.png", alpha)
            if save_fmt == "overlay":
                _save_overlay(img_np, pred_np, output_dir / "masks" / f"{stem}.png", alpha)

            # --- entropy map ---
            if unlbl_cfg.get("save_entropy_maps", False):
                save_entropy_overlay(
                    image=img_np,
                    probs=probs_hw,
                    save_path=str(output_dir / "entropy" / f"{stem}_entropy.png"),
                )

            # --- labelled comparison ---
            if (
                has_labels
                and masks_gt is not None
                and lbl_cfg.get("save_comparisons", False)
            ):
                gt_np = masks_gt[i].cpu().numpy()
                save_comparison(
                    image=img_np,
                    gt_mask=gt_np,
                    pred_mask=pred_np,
                    save_path=output_dir / "comparisons" / f"{stem}_compare.png",
                )

            # --- per-image unlabelled metrics ---
            conf_stats     = compute_confidence_stats(probs_hw, low_conf_threshold=low_conf_t)
            coverage_stats = compute_class_coverage(
                pred_mask=pred_np,
                num_classes=num_classes,
                class_names=RUGD_CLASSES,
                ground_class_indices=_GROUND_CLASS_INDICES,
            )
            boundary_stats = compute_boundary_stats(pred_np)

            per_image_stats.append({
                "stem":     stem,
                **conf_stats,
                **{k: v for k, v in coverage_stats.items()
                   if not isinstance(v, dict)},   # skip nested dicts for now
                **boundary_stats,
            })

        if (batch_idx + 1) % 10 == 0:
            print(f"  {batch_idx + 1}/{len(loader)} batches", flush=True)

    # --- aggregate unlabelled stats ---
    def _mean(key: str) -> float:
        vals = [s[key] for s in per_image_stats if key in s]
        return float(np.mean(vals)) if vals else float("nan")

    unlabelled_summary = {
        "num_images":              len(per_image_stats),
        "mean_confidence":         _mean("mean_confidence"),
        "mean_low_conf_fraction":  _mean("low_conf_pixel_fraction"),
        "mean_entropy":            _mean("mean_entropy"),
        "mean_edge_density":       _mean("edge_density"),
        "mean_ground_coverage":    _mean("ground_coverage"),
        "per_image":               per_image_stats,
    }

    # --- temporal consistency (only meaningful for video / sequential frames) ---
    if len(all_pred_masks) >= 2:
        tc = compute_temporal_consistency(all_pred_masks, ignore_index=0)
        unlabelled_summary["temporal"] = tc

    # --- labelled summary ---
    labelled_summary: Optional[dict] = None
    conf_matrix: Optional[np.ndarray] = None
    if seg_metrics is not None:
        labelled_results = seg_metrics.compute()
        labelled_results["gt_counts"] = gt_counts
        conf_matrix = seg_metrics._confusion.copy()
        labelled_summary = {
            "mIoU":             float(labelled_results["mIoU"]),
            "pixel_accuracy":   float(labelled_results["pixel_accuracy"]),
            "mean_class_acc":   float(labelled_results["mean_class_acc"]),
        }

    return {
        "unlabelled": unlabelled_summary,
        "labelled":   labelled_summary,
        "_labelled_results_full": labelled_results if seg_metrics else None,
        "_conf_matrix":           conf_matrix,
        "_gt_counts":             gt_counts if has_labels else None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DeepLabV3+ inference on custom images"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to custom dataset YAML config (e.g. configs/custom_dataset.yaml)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Override model.checkpoint in config",
    )
    parser.add_argument(
        "--input", default=None,
        help="Override dataset.image_dir in config",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override infer.output_dir in config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Allow CLI overrides
    if args.input:
        cfg.setdefault("dataset", {})["image_dir"] = args.input
    if args.output:
        cfg.setdefault("infer", {})["output_dir"] = args.output
    if args.checkpoint:
        cfg.setdefault("model", {})["checkpoint"] = args.checkpoint

    # ---- device ----
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ---- model ----
    ckpt_path   = resolve_checkpoint(cfg, None)   # CLI override already applied above
    num_classes = cfg["model"]["num_classes"]
    encoder     = cfg["model"].get("encoder", "resnet101")

    model, ckpt_info = load_checkpoint(
        checkpoint_path=ckpt_path,
        device=device,
        encoder=encoder,
        num_classes=num_classes,
    )
    print(
        f"Loaded: {ckpt_path}  "
        f"(epoch {ckpt_info.get('epoch', '?')}, "
        f"best_mIoU={ckpt_info.get('best_miou', 0):.4f})"
    )

    # ---- dataset ----
    infer_cfg  = cfg.get("infer", {})
    ds_cfg     = cfg.get("dataset", {})
    image_size = tuple(infer_cfg.get("image_size", [512, 512]))
    transform  = get_val_transforms(image_size)

    # Remap config → colormap for label conversion
    remap_cfg = cfg.get("remap")   # not implemented yet — pass None for default

    dataset = CustomImageDataset(
        image_dir=ds_cfg["image_dir"],
        label_dir=ds_cfg.get("label_dir"),
        file_list=ds_cfg.get("file_list"),
        extensions=ds_cfg.get("extensions"),
        transform=transform,
        colormap=RUGD_COLORMAP,  # used when label_dir is set
    )
    has_labels = ds_cfg.get("label_dir") is not None

    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=infer_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=infer_cfg.get("num_workers", 4),
        pin_memory=pin_memory,
    )
    print(f"Dataset: {len(dataset)} images  (labels: {has_labels})")

    # ---- output dir ----
    output_dir = Path(infer_cfg.get("output_dir", "inference_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ---- FPS benchmark ----
    unlbl_cfg = cfg.get("unlabelled_metrics", {})
    fps_info: Optional[dict] = None
    if unlbl_cfg.get("benchmark_fps", True):
        print("\nBenchmarking FPS...")
        fps_info = benchmark_fps(
            model=model,
            device=device,
            image_size=image_size,
            batch_size=infer_cfg.get("batch_size", 1),
            num_warmup=unlbl_cfg.get("benchmark_warmup", 5),
            num_runs=unlbl_cfg.get("benchmark_runs", 20),
        )
        print(f"  {fps_info['fps']} FPS  ({fps_info['ms_per_image']} ms/image)")

    # ---- inference ----
    summary = run_inference(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        output_dir=output_dir,
        has_labels=has_labels,
    )

    if fps_info:
        summary["unlabelled"]["fps"] = fps_info

    # ---- save JSON ----
    # Remove internal keys before serialising
    labelled_results_full = summary.pop("_labelled_results_full", None)
    conf_matrix           = summary.pop("_conf_matrix", None)
    gt_counts             = summary.pop("_gt_counts", None)

    # Convert numpy types so json.dumps works
    def _jsonify(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    metrics_json = output_dir / "inference_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(summary, f, indent=2, default=_jsonify)
    print(f"\nMetrics saved → {metrics_json}")

    # ---- charts ----
    # Always save class coverage chart
    if summary["unlabelled"]["per_image"]:
        # Aggregate coverage across all images
        agg_cov: dict[int, float] = {}
        for s in summary["unlabelled"]["per_image"]:
            pass  # coverage_by_index not in per_image (we stripped it above)

    # Generate coverage chart from first image stats as a representative sample
    first_stats = summary["unlabelled"]["per_image"][0] if summary["unlabelled"]["per_image"] else {}
    save_inference_report_charts(
        stats=first_stats,
        save_dir=str(output_dir),
        class_names=RUGD_CLASSES,
    )

    # ---- labelled report ----
    if (
        has_labels
        and labelled_results_full is not None
        and cfg.get("labelled_metrics", {}).get("generate_report", True)
    ):
        generate_metrics_report(
            run_dir=output_dir,
            results=labelled_results_full,
            conf_matrix=conf_matrix,
            class_names=RUGD_CLASSES,
            extra_info={
                "Checkpoint": str(ckpt_path),
                "Input":      str(ds_cfg["image_dir"]),
                "Mode":       "labelled inference",
            },
        )
        print(f"Full metrics report → {output_dir}/")

    # ---- print summary ----
    ul = summary["unlabelled"]
    print("\n" + "=" * 52)
    print(f"  Images processed   : {ul['num_images']}")
    print(f"  Mean confidence    : {ul['mean_confidence']:.4f}")
    print(f"  Low-conf fraction  : {ul['mean_low_conf_fraction']:.4f}")
    print(f"  Mean entropy       : {ul['mean_entropy']:.4f}")
    print(f"  Mean edge density  : {ul['mean_edge_density']:.4f}")
    print(f"  Ground coverage    : {ul['mean_ground_coverage']:.4f}")
    if fps_info:
        print(f"  Throughput         : {fps_info['fps']} FPS")
    if "temporal" in ul:
        tc = ul["temporal"]
        print(f"  Frame IoU (video)  : {tc.get('mean_frame_iou', float('nan')):.4f}")
        print(f"  Flicker rate       : {tc.get('flicker_rate', float('nan')):.4f}")
    if summary.get("labelled"):
        lb = summary["labelled"]
        print(f"\n  --- Labelled ---")
        print(f"  mIoU               : {lb['mIoU']:.4f}")
        print(f"  Pixel accuracy     : {lb['pixel_accuracy']:.4f}")
        print(f"  Mean class acc     : {lb['mean_class_acc']:.4f}")
    print("=" * 52)
    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
