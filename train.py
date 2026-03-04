"""
Training script for DeepLabV3+ on RUGD.

Quick start
-----------
    uv run python train.py                              # default config
    uv run python train.py --wandb                      # enable W&B logging
    uv run python train.py --run_name my_exp --epochs 50
    uv run python train.py --resume runs/.../latest.pt  # resume

On HPC (SLURM):
    sbatch hpc/train.slurm
    # Or the one-command launcher:
    bash hpc/launch.sh
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml

# ---------------------------------------------------------------------------
# Project root — all relative paths are resolved from here
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, PROJECT_DIR)

from data.rugd_dataset import RUGDDataset, RUGD_CLASSES, IGNORE_INDEX, RUGD_COLORMAP
from models.deeplabv3plus import build_model, save_checkpoint
from utils.metrics import SegmentationMetrics
from utils.metrics_chart import generate_metrics_report
from utils.transforms import get_train_transforms, get_val_transforms
from utils.visualization import colorize_mask, tensor_to_numpy_image


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------


def init_wandb(cfg: dict, run_dir: Path) -> bool:
    """Initialise W&B if enabled in config.  Returns True if active."""
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enabled", False):
        return False
    try:
        import wandb
    except ImportError:
        warnings.warn("wandb not installed. Run: uv pip install wandb")
        return False

    wandb.init(
        project=wb_cfg.get("project", "rugd-segmentation"),
        entity=wb_cfg.get("entity"),  # None = use default
        name=cfg["output"]["run_name"],
        tags=wb_cfg.get("tags", []),
        dir=str(run_dir),
        config={
            "model": cfg["model"],
            "train": cfg["train"],
            "data": {k: v for k, v in cfg["data"].items() if k != "root_dir"},
        },
        resume="allow",
    )
    print(f"  W&B run: {wandb.run.url}")
    return True


def wandb_watch_model(model: nn.Module, cfg: dict, wb_active: bool) -> None:
    if not wb_active:
        return
    import wandb
    if cfg.get("wandb", {}).get("watch_model", True):
        wandb.watch(model, log="gradients", log_freq=100)


def wandb_log_metrics(
    metrics: dict,
    step: int,
    wb_active: bool,
) -> None:
    if not wb_active:
        return
    import wandb
    wandb.log(metrics, step=step)


def wandb_log_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    wb_active: bool,
    num_samples: int = 4,
) -> None:
    """Log a grid of (image | ground truth | prediction) panels to W&B."""
    if not wb_active:
        return
    import wandb

    model.eval()
    panels = []
    collected = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            for i in range(images.shape[0]):
                if collected >= num_samples:
                    break
                img_np = tensor_to_numpy_image(images[i])     # (H, W, 3) uint8
                gt_np = masks[i].cpu().numpy().astype(np.uint8)
                pred_np = preds[i].cpu().numpy().astype(np.uint8)

                # Build class label dict for W&B mask overlay
                class_labels = {int(j): name for j, name in enumerate(RUGD_CLASSES)}

                panels.append(
                    wandb.Image(
                        img_np,
                        masks={
                            "ground_truth": {
                                "mask_data": gt_np,
                                "class_labels": class_labels,
                            },
                            "prediction": {
                                "mask_data": pred_np,
                                "class_labels": class_labels,
                            },
                        },
                        caption=f"epoch {epoch + 1}",
                    )
                )
                collected += 1

            if collected >= num_samples:
                break

    wandb.log({"val/predictions": panels}, step=epoch)
    model.train()


# ---------------------------------------------------------------------------
# Config / arg helpers
# ---------------------------------------------------------------------------


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    """Overwrite config values with explicitly provided CLI args."""
    overrides = {
        "run_name": ("output", "run_name"),
        "epochs": ("train", "epochs"),
        "batch_size": ("train", "batch_size"),
        "learning_rate": ("train", "learning_rate"),
        "data_root": ("data", "root_dir"),
    }
    for arg_key, (section, key) in overrides.items():
        val = getattr(args, arg_key, None)
        if val is not None:
            cfg[section][key] = val

    if getattr(args, "wandb", False):
        cfg.setdefault("wandb", {})["enabled"] = True

    return cfg


# ---------------------------------------------------------------------------
# Optimiser / scheduler builders
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_name = cfg["train"]["optimizer"].lower()
    lr = cfg["train"]["learning_rate"]
    wd = cfg["train"]["weight_decay"]

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=cfg["train"]["momentum"], weight_decay=wd, nesterov=True,
        )
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt_name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
) -> torch.optim.lr_scheduler.LRScheduler:
    sched = cfg["train"]["scheduler"]
    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"]["warmup_epochs"]

    if sched == "poly":
        power = cfg["train"]["poly_power"]
        def poly_lambda(e: int) -> float:
            if e < warmup:
                return (e + 1) / max(warmup, 1)
            p = (e - warmup) / max(epochs - warmup, 1)
            return (1 - p) ** power
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lambda)

    elif sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup, eta_min=1e-6,
        )
    raise ValueError(f"Unknown scheduler: {sched!r}")


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: dict,
    writer: SummaryWriter,
    wb_active: bool,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    grad_accum = cfg["train"]["grad_accum_steps"]
    log_every = cfg["output"]["log_every"]
    use_amp = cfg["train"]["use_amp"] and device.type == "cuda"
    clip_norm = cfg["train"]["clip_grad_norm"]

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks) / grad_accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0:
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        step_loss = loss.item() * grad_accum
        total_loss += step_loss

        if (batch_idx + 1) % log_every == 0:
            avg = total_loss / (batch_idx + 1)
            writer.add_scalar("train/batch_loss", avg, global_step)
            wandb_log_metrics({"train/batch_loss": avg}, global_step, wb_active)
            print(f"  [{batch_idx + 1}/{len(loader)}] loss={avg:.4f}", flush=True)

        global_step += 1

    return total_loss / max(len(loader), 1), global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict,
    metrics_tracker: Optional[SegmentationMetrics] = None,
) -> dict:
    model.eval()
    metrics = metrics_tracker or SegmentationMetrics(
        num_classes=cfg["data"]["num_classes"], ignore_index=0
    )
    total_loss = 0.0
    use_amp = cfg["train"]["use_amp"] and device.type == "cuda"
    gt_counts = np.zeros(cfg["data"]["num_classes"], dtype=np.int64)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            total_loss += criterion(logits, masks).item()
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
        # Accumulate GT pixel counts for FWIoU and distribution charts
        for c in range(cfg["data"]["num_classes"]):
            gt_counts[c] += int((masks == c).sum().item())

    results = metrics.compute()
    results["loss"] = total_loss / max(len(loader), 1)
    results["gt_counts"] = gt_counts
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on RUGD")
    parser.add_argument("--config", default=os.path.join(PROJECT_DIR, "configs", "config.yaml"))
    parser.add_argument("--data_root", help="Override data.root_dir")
    parser.add_argument("--run_name", help="Override output.run_name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_args(cfg, args)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Output directories — always write under the submitting user's home
    # so bsub jobs (which run as the personal account) have write access.
    # ------------------------------------------------------------------
    USER_HOME = os.path.expanduser("~")
    run_dir = Path(USER_HOME) / "segmentation_model" / "runs" / cfg["output"]["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    vis_dir  = run_dir / "vis"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tb_logs").mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging — TensorBoard + optional W&B
    # ------------------------------------------------------------------
    writer = SummaryWriter(log_dir=str(run_dir / "tb_logs"))
    wb_active = init_wandb(cfg, run_dir)

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    image_size = tuple(cfg["data"]["image_size"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_ds = RUGDDataset(
            root_dir=cfg["data"]["root_dir"],
            split="train",
            transform=get_train_transforms(image_size),
        )
        val_ds = RUGDDataset(
            root_dir=cfg["data"]["root_dir"],
            split="val",
            transform=get_val_transforms(image_size),
        )
    for w in caught:
        print(f"  [WARN] {w.message}")

    if len(train_ds) == 0:
        print("ERROR: Training set is empty. Check data.root_dir in config.")
        sys.exit(1)
    if len(val_ds) == 0:
        print("WARNING: Validation set is empty — skipping validation.")

    # pin_memory only works on CUDA (not MPS or CPU)
    pin_memory = cfg["data"]["pin_memory"] and device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"],
        pin_memory=pin_memory, drop_last=len(train_ds) >= cfg["train"]["batch_size"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
        pin_memory=pin_memory,
    )
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(
        encoder=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["data"]["num_classes"],
        activation=cfg["model"]["activation"],
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    wandb_watch_model(model, cfg, wb_active)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    if cfg["train"]["use_class_weights"] and len(train_ds) > 0:
        print("Computing class weights…")
        weights = RUGDDataset.compute_class_weights(
            root_dir=cfg["data"]["root_dir"],
            split="train",
            num_classes=cfg["data"]["num_classes"],
        ).to(device)
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)  # 0 = void

    # ------------------------------------------------------------------
    # Optimiser / scheduler / scaler
    # ------------------------------------------------------------------
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=cfg["train"]["use_amp"] and device.type == "cuda",
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_miou = 0.0
    global_step = 0

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        global_step = ckpt.get("global_step", 0)

    # ------------------------------------------------------------------
    # Training loop — accumulate history for post-training charts
    # ------------------------------------------------------------------
    history: dict[str, list] = {
        "epochs": [], "train_loss": [], "val_loss": [],
        "val_miou": [], "val_pixel_acc": [], "lr": [],
    }
    # Keep the final confusion matrix and GT pixel counts for the report
    final_conf_matrix = None
    final_gt_counts   = None
    final_val_results = None

    print(f"\nStarting training: {cfg['train']['epochs']} epochs\n")
    wb_img_freq = cfg.get("wandb", {}).get("log_images_every", 5)

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.time()
        print(f"Epoch [{epoch + 1}/{cfg['train']['epochs']}]")

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, cfg, writer, wb_active, global_step,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)

        # Validation
        if len(val_ds) > 0:
            val_metrics_tracker = SegmentationMetrics(
                num_classes=cfg["data"]["num_classes"], ignore_index=0
            )
            val_res = validate(model, val_loader, criterion, device, cfg,
                               metrics_tracker=val_metrics_tracker)
            miou = val_res["mIoU"]
            final_conf_matrix = val_metrics_tracker._confusion.copy()
            final_val_results = val_res

            writer.add_scalar("val/loss", val_res["loss"], epoch)
            writer.add_scalar("val/mIoU", miou, epoch)
            writer.add_scalar("val/pixel_accuracy", val_res["pixel_accuracy"], epoch)

            wandb_log_metrics(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "train/lr": current_lr,
                    "val/loss": val_res["loss"],
                    "val/mIoU": miou,
                    "val/pixel_accuracy": val_res["pixel_accuracy"],
                    "val/mean_class_acc": val_res["mean_class_acc"],
                },
                step=epoch,
                wb_active=wb_active,
            )

            # Log prediction images to W&B
            if wb_active and (epoch + 1) % wb_img_freq == 0 and len(val_ds) > 0:
                wandb_log_predictions(model, val_loader, device, epoch, wb_active)

            elapsed = time.time() - t0
            print(
                f"  train_loss={train_loss:.4f}  val_loss={val_res['loss']:.4f}"
                f"  mIoU={miou:.4f}  pix_acc={val_res['pixel_accuracy']:.4f}"
                f"  lr={current_lr:.2e}  [{elapsed:.0f}s]"
            )
            # Track history for post-training charts
            history["epochs"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_res["loss"])
            history["val_miou"].append(miou)
            history["val_pixel_acc"].append(val_res["pixel_accuracy"])
            history["lr"].append(current_lr)
        else:
            miou = 0.0
            wandb_log_metrics(
                {"epoch": epoch + 1, "train/epoch_loss": train_loss, "train/lr": current_lr},
                step=epoch, wb_active=wb_active,
            )
            print(f"  train_loss={train_loss:.4f}  lr={current_lr:.2e}")
            history["epochs"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["lr"].append(current_lr)

        # Checkpointing — unwrap DataParallel so checkpoints are portable
        _model = model.module if isinstance(model, nn.DataParallel) else model
        extra = {"global_step": global_step}
        save_checkpoint(_model, optimizer, epoch, best_miou, str(ckpt_dir / "latest.pt"), extra)

        if miou > best_miou:
            best_miou = miou
            save_checkpoint(_model, optimizer, epoch, best_miou, str(ckpt_dir / "best.pt"), extra)
            print(f"  *** New best mIoU: {best_miou:.4f} — saved best.pt ***")

        if (epoch + 1) % cfg["output"]["save_every"] == 0:
            save_checkpoint(
                _model, optimizer, epoch, best_miou,
                str(ckpt_dir / f"epoch_{epoch + 1:04d}.pt"), extra,
            )

    writer.close()
    if wb_active:
        import wandb
        wandb.finish()

    # ------------------------------------------------------------------
    # Post-training metrics report
    # ------------------------------------------------------------------
    if final_conf_matrix is not None and final_val_results is not None:
        try:
            generate_metrics_report(
                run_dir=run_dir,
                results=final_val_results,
                conf_matrix=final_conf_matrix,
                class_names=RUGD_CLASSES,
                history=history if history["train_loss"] else None,
                extra_info={
                    "Run":        cfg["output"]["run_name"],
                    "Encoder":    cfg["model"]["encoder"],
                    "Epochs":     cfg["train"]["epochs"],
                    "Batch size": cfg["train"]["batch_size"],
                    "Best mIoU":  f"{best_miou:.4f}",
                },
            )
        except Exception as e:
            print(f"[WARN] Could not generate metrics report: {e}")

    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"TensorBoard: tensorboard --logdir {run_dir / 'tb_logs'}")


if __name__ == "__main__":
    main()
