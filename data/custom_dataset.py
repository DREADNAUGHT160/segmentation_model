"""
Custom image dataset for inference on user-supplied images.

Supports three input modes:
  1. Directory  — scans recursively for image files
  2. File list  — reads paths from a plain-text file (one path per line)
  3. Video      — extracts frames from an MP4/AVI/MOV file

Labels are optional; if provided they are loaded and returned alongside images
so that labelled_metrics can be computed in infer_custom.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_EXTS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
)

VIDEO_EXTS: tuple[str, ...] = (
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
)


# ---------------------------------------------------------------------------
# Helper: RGB annotation → class index  (same as rugd_dataset.py)
# ---------------------------------------------------------------------------

def _rgb_to_class_index(
    rgb_mask: np.ndarray,
    colormap: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour lookup: RGB colourmap annotation → class-index mask.

    Args:
        rgb_mask:  ``(H, W, 3)`` uint8 RGB image.
        colormap:  ``(C, 3)`` uint8 array mapping class index → RGB.

    Returns:
        ``(H, W)`` uint8 class-index array.
    """
    h, w = rgb_mask.shape[:2]
    flat = rgb_mask.reshape(-1, 3).astype(np.int32)
    cmap = colormap.astype(np.int32)
    dists = np.sum((flat[:, None, :] - cmap[None, :, :]) ** 2, axis=2)
    return dists.argmin(axis=1).reshape(h, w).astype(np.uint8)


# ---------------------------------------------------------------------------
# Frame extractor for video inputs
# ---------------------------------------------------------------------------

def extract_video_frames(
    video_path: str | Path,
    max_frames: Optional[int] = None,
) -> list[Image.Image]:
    """Extract frames from a video file as PIL Images.

    Args:
        video_path:  Path to the video file.
        max_frames:  Optional cap on number of frames to extract.

    Returns:
        List of RGB PIL Images.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for video input. "
            "Install with: uv pip install opencv-python-headless"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames: list[Image.Image] = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CustomImageDataset(Dataset):
    """Dataset for user-supplied images (inference or labelled evaluation).

    Args:
        image_dir:     Path to a directory of images  *or*  a single image
                       file  *or*  a video file.
        label_dir:     Optional path to corresponding label masks.
                       Must contain files with the same stem as images.
        file_list:     Optional path to a text file listing image paths
                       (one per line).  Overrides recursive directory scan.
        extensions:    Image extensions to include in directory scan.
        transform:     albumentations transform applied to image (and mask
                       if labels are present).
        colormap:      ``(C, 3)`` uint8 array for RGB→index conversion of
                       label masks.  Pass ``None`` if labels are already
                       single-channel index PNGs.
        max_video_frames:  For video input — cap the number of frames.
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: Optional[str | Path] = None,
        file_list: Optional[str | Path] = None,
        extensions: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        colormap: Optional[np.ndarray] = None,
        max_video_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.transform  = transform
        self.colormap   = colormap
        self.label_dir  = Path(label_dir) if label_dir else None

        exts = tuple(
            (f".{e.lstrip('.')}" for e in extensions)
            if extensions else DEFAULT_IMAGE_EXTS
        )

        # ---- resolve input path ----
        src = Path(image_dir)
        if not src.exists():
            raise FileNotFoundError(f"image_dir not found: {src}")

        if src.is_file() and src.suffix.lower() in VIDEO_EXTS:
            # Video mode — extract frames into memory
            self._frames = extract_video_frames(src, max_video_frames)
            self._stems  = [f"frame_{i:06d}" for i in range(len(self._frames))]
            self._paths  = [src] * len(self._frames)  # placeholder
            self._is_video = True
        else:
            self._is_video = False
            self._frames   = []

            if file_list is not None:
                # Explicit file list
                fl = Path(file_list)
                lines = fl.read_text().splitlines()
                self._paths = [Path(l.strip()) for l in lines if l.strip()]
            elif src.is_file():
                # Single image
                self._paths = [src]
            else:
                # Directory scan
                self._paths = sorted(
                    p for p in src.rglob("*")
                    if p.suffix.lower() in exts
                )

            if not self._paths:
                warnings.warn(
                    f"No images found in '{src}' with extensions {exts}",
                    UserWarning,
                    stacklevel=2,
                )

            self._stems = [p.stem for p in self._paths]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._frames) if self._is_video else len(self._paths)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        """Return a sample dict with keys:

        - ``"image"``  : ``(3, H, W)`` float32 tensor
        - ``"mask"``   : ``(H, W)`` int64 tensor  *(only when label_dir given)*
        - ``"stem"``   : filename stem (str), useful for saving outputs
        - ``"path"``   : original file path (str)
        """
        # ---- image ----
        if self._is_video:
            pil_img = self._frames[idx].convert("RGB")
            path_str = f"frame_{idx:06d}"
        else:
            path_str = str(self._paths[idx])
            pil_img  = Image.open(path_str).convert("RGB")

        img_np = np.array(pil_img, dtype=np.uint8)  # (H, W, 3)

        # ---- label (optional) ----
        mask_np: Optional[np.ndarray] = None
        if self.label_dir is not None:
            stem = self._stems[idx]
            # Try each common extension
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                lbl_path = self.label_dir / f"{stem}{ext}"
                if lbl_path.exists():
                    lbl_img = Image.open(lbl_path)
                    if lbl_img.mode == "L":
                        # Already a single-channel index mask
                        mask_np = np.array(lbl_img, dtype=np.uint8)
                    else:
                        lbl_rgb = np.array(lbl_img.convert("RGB"), dtype=np.uint8)
                        if self.colormap is not None:
                            mask_np = _rgb_to_class_index(lbl_rgb, self.colormap)
                        else:
                            # Treat first channel as index
                            mask_np = lbl_rgb[:, :, 0]
                    break
            else:
                warnings.warn(
                    f"Label not found for '{stem}' in '{self.label_dir}'",
                    UserWarning,
                    stacklevel=2,
                )

        # ---- transform ----
        if self.transform is not None:
            if mask_np is not None:
                out = self.transform(image=img_np, mask=mask_np)
                img_tensor  = out["image"]          # (3, H, W) float32
                mask_tensor = out["mask"]
                if isinstance(mask_tensor, torch.Tensor):
                    if mask_tensor.ndim == 3:
                        mask_tensor = mask_tensor.squeeze(0)
                    mask_tensor = mask_tensor.long()
                else:
                    mask_tensor = torch.from_numpy(mask_np).long()
            else:
                out = self.transform(image=img_np)
                img_tensor = out["image"]
                mask_tensor = None
        else:
            # No transform — return raw (not suitable for batching, but useful
            # for quick visualisation)
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask_np).long() if mask_np is not None else None

        sample: dict = {
            "image": img_tensor,
            "stem":  self._stems[idx],
            "path":  path_str,
        }
        if mask_tensor is not None:
            sample["mask"] = mask_tensor

        return sample
