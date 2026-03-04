"""
Evaluation script — compute full metrics on any labelled split.

Usage
-----
    uv run python evaluate.py --checkpoint runs/deeplabv3plus_rugd/checkpoints/best.pt

Also save visualisations + full metrics report:
    uv run python evaluate.py --checkpoint best.pt --save_vis --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data.rugd_dataset import RUGDDataset, RUGD_CLASSES
from models.deeplabv3plus import load_checkpoint
from utils.metrics import SegmentationMetrics
from utils.metrics_chart import generate_metrics_report
from utils.transforms import get_val_transforms
from utils.visualization import save_comparison, tensor_to_numpy_image


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    save_vis: bool = False,
    vis_dir: Path | None = None,
) -> tuple[dict, np.ndarray]:
    """Run evaluation loop.

    Returns:
        ``(results_dict, confusion_matrix)``
        ``results_dict`` contains all scalar metrics plus ``"gt_counts"``.
        ``confusion_matrix`` is the raw ``(C, C)`` int64 array.
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=0)
    gt_counts = np.zeros(num_classes, dtype=np.int64)

    if save_vis and vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        preds  = logits.argmax(dim=1)

        metrics.update(preds, masks)

        for c in range(num_classes):
            gt_counts[c] += int((masks == c).sum().item())

        if save_vis and vis_dir is not None:
            for i in range(images.shape[0]):
                img_np  = tensor_to_numpy_image(images[i])
                gt_np   = masks[i].cpu().numpy()
                pred_np = preds[i].cpu().numpy()
                save_comparison(
                    image=img_np, gt_mask=gt_np, pred_mask=pred_np,
                    save_path=vis_dir / f"batch{batch_idx:04d}_sample{i}.png",
                )

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(loader)} batches", flush=True)

    results = metrics.compute()
    results["gt_counts"] = gt_counts
    return results, metrics._confusion.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepLabV3+ on RUGD")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_vis",   action="store_true", help="Save comparison images")
    parser.add_argument("--vis_dir",    default="eval_vis/")
    parser.add_argument("--report",     action="store_true",
                        help="Generate full metrics report (charts + JSON + text summary)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Model --
    model, ckpt_info = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        encoder=cfg["model"]["encoder"],
        num_classes=cfg["data"]["num_classes"],
    )
    print(f"Loaded checkpoint: epoch {ckpt_info.get('epoch', '?')}, "
          f"best_mIoU={ckpt_info.get('best_miou', 0):.4f}")

    # -- Dataset --
    image_size = tuple(cfg["data"]["image_size"])
    pin_memory = cfg["data"]["pin_memory"] and device.type == "cuda"
    dataset = RUGDDataset(
        root_dir=cfg["data"]["root_dir"], split=args.split,
        transform=get_val_transforms(image_size),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=pin_memory,
    )
    print(f"Evaluating split='{args.split}' — {len(dataset)} samples\n")

    # -- Run --
    results, conf_matrix = evaluate(
        model=model, loader=loader,
        num_classes=cfg["data"]["num_classes"], device=device,
        save_vis=args.save_vis,
        vis_dir=Path(args.vis_dir) if args.save_vis else None,
    )

    # -- Print summary --
    ground = {"dirt", "sand", "grass", "asphalt", "gravel", "mulch", "rock-bed", "water"}
    print("\n" + "=" * 56)
    print(f"  mIoU              : {results['mIoU']:.4f}")
    print(f"  Pixel accuracy    : {results['pixel_accuracy']:.4f}")
    print(f"  Mean class acc    : {results['mean_class_acc']:.4f}")
    print("\n  Per-class IoU:")
    for i, name in enumerate(RUGD_CLASSES):
        iou = results["iou_per_class"][i]
        tag = " ← ground" if name in ground else ""
        val = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
        print(f"    [{i:2d}] {name:<22} {val}{tag}")
    print("=" * 56)

    if args.save_vis:
        print(f"\nVisualisations → {args.vis_dir}")

    # -- Full report --
    if args.report:
        ckpt_path = Path(args.checkpoint)
        run_dir   = ckpt_path.parent.parent  # runs/<name>/checkpoints/best.pt → runs/<name>
        generate_metrics_report(
            run_dir=run_dir,
            results=results,
            conf_matrix=conf_matrix,
            class_names=RUGD_CLASSES,
            extra_info={
                "Checkpoint":  str(ckpt_path),
                "Split":       args.split,
                "Best mIoU":   f"{ckpt_info.get('best_miou', 0):.4f}",
            },
        )


if __name__ == "__main__":
    main()
