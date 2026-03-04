"""
Inference script — run segmentation on an image or video file.

Usage
-----
Single image:
    uv run python infer.py --checkpoint best.pt --input photo.jpg --output pred.png

Directory of images:
    uv run python infer.py --checkpoint best.pt --input frames/ --output results/

Video file:
    uv run python infer.py --checkpoint best.pt --input drive.mp4 --output drive_seg.mp4

Overlay (blended) instead of pure mask:
    uv run python infer.py --checkpoint best.pt --input photo.jpg --overlay --alpha 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data.rugd_dataset import RUGD_COLORMAP
from models.deeplabv3plus import load_checkpoint
from utils.visualization import colorize_mask, overlay_mask

# Image extensions recognized as single-image inputs
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Pre / post-processing
# ---------------------------------------------------------------------------


def preprocess_frame(
    frame_bgr: np.ndarray,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert a BGR frame to a normalised CHW float tensor.

    Args:
        frame_bgr:  OpenCV BGR frame (H, W, 3) uint8.
        image_size: ``(H, W)`` to resize to before inference.
        device:     Target device.

    Returns:
        ``(tensor, original_hw)`` where tensor is ``(1, 3, H', W')``.
    """
    original_hw = frame_bgr.shape[:2]
    h, w = image_size
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.451, 0.224, 0.225], dtype=np.float32)
    tensor = (rgb.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor, original_hw


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    original_hw: tuple[int, int],
) -> np.ndarray:
    """Run inference and return a class-index mask at original resolution.

    Args:
        model:       Model in eval mode.
        tensor:      ``(1, 3, H, W)`` float tensor.
        original_hw: ``(H, W)`` to resize the mask back to.

    Returns:
        Class-index mask ``(H, W)`` uint8.
    """
    logits = model(tensor)                        # (1, C, H', W')
    pred = logits.argmax(dim=1).squeeze(0)        # (H', W')
    pred_np = pred.cpu().numpy().astype(np.uint8)

    if pred_np.shape != original_hw:
        pred_np = cv2.resize(
            pred_np, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_NEAREST
        )
    return pred_np


# ---------------------------------------------------------------------------
# Per-input handlers
# ---------------------------------------------------------------------------


def infer_image(
    model: torch.nn.Module,
    input_path: Path,
    output_path: Path,
    image_size: tuple[int, int],
    device: torch.device,
    use_overlay: bool,
    alpha: float,
) -> None:
    """Run inference on a single image file."""
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    tensor, orig_hw = preprocess_frame(frame, image_size, device)
    mask = predict_mask(model, tensor, orig_hw)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if use_overlay:
        result = overlay_mask(rgb_frame, mask, alpha=alpha)
    else:
        result = colorize_mask(mask)

    Image.fromarray(result).save(output_path)
    print(f"Saved: {output_path}")


def infer_directory(
    model: torch.nn.Module,
    input_dir: Path,
    output_dir: Path,
    image_size: tuple[int, int],
    device: torch.device,
    use_overlay: bool,
    alpha: float,
) -> None:
    """Run inference on all images in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    for i, img_path in enumerate(image_files):
        out_path = output_dir / img_path.name
        infer_image(model, img_path, out_path, image_size, device, use_overlay, alpha)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(image_files)}")


def infer_video(
    model: torch.nn.Module,
    input_path: Path,
    output_path: Path,
    image_size: tuple[int, int],
    device: torch.device,
    use_overlay: bool,
    alpha: float,
    fps_override: float | None,
) -> None:
    """Run inference on a video file, writing a segmentation video."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = fps_override if fps_override else src_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (orig_w, orig_h))

    print(f"Processing {total_frames} frames at {src_fps:.1f} fps → {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor, orig_hw = preprocess_frame(frame, image_size, device)
        mask = predict_mask(model, tensor, orig_hw)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if use_overlay:
            result_rgb = overlay_mask(rgb_frame, mask, alpha=alpha)
        else:
            result_rgb = colorize_mask(mask)

        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        writer.write(result_bgr)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames}", flush=True)

    cap.release()
    writer.release()
    print(f"Video saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLabV3+ inference on image or video")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--input", required=True, help="Input: image, directory, or video")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--overlay", action="store_true", help="Blend mask over original image")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay blend factor [0–1]")
    parser.add_argument("--video_fps", type=float, default=None, help="Override output FPS")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, _ = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        encoder=cfg["model"]["encoder"],
        num_classes=cfg["data"]["num_classes"],
    )

    image_size = tuple(cfg["data"]["image_size"])
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        infer_directory(model, input_path, output_path, image_size, device, args.overlay, args.alpha)

    elif input_path.suffix.lower() in VIDEO_EXTS:
        infer_video(
            model, input_path, output_path, image_size, device,
            args.overlay, args.alpha, args.video_fps,
        )

    elif input_path.suffix.lower() in IMAGE_EXTS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        infer_image(model, input_path, output_path, image_size, device, args.overlay, args.alpha)

    else:
        print(f"Unrecognised input type: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
