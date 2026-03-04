"""
Quick smoke test — verifies all components work without needing RUGD data.
Runs a full forward + backward pass on synthetic random data.

Usage:
    uv run python smoke_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def run(name, fn):
    try:
        result = fn()
        print(f"[{PASS}] {name}" + (f" — {result}" if result else ""))
    except Exception as e:
        print(f"[{FAIL}] {name}: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
print("\n=== Imports ===")
run("torch",              lambda: f"torch {torch.__version__}")
run("segmentation-models-pytorch", lambda: __import__("segmentation_models_pytorch").__version__)
run("albumentations",     lambda: __import__("albumentations").__version__)
run("opencv-python",      lambda: __import__("cv2").__version__)
run("data module",        lambda: str(__import__("data")))
run("models module",      lambda: str(__import__("models")))
run("utils module",       lambda: str(__import__("utils")))

# ---------------------------------------------------------------------------
# 2. Model build + forward pass
# ---------------------------------------------------------------------------
print("\n=== Model ===")
from models.deeplabv3plus import build_model, save_checkpoint, load_checkpoint

def test_model_forward():
    model = build_model(encoder="resnet101", encoder_weights=None, num_classes=25)
    model.eval()
    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, 25, H, W), f"Wrong output shape: {out.shape}"
    return f"output shape {tuple(out.shape)}"

run("model build + forward pass (resnet101, no pretrain)", test_model_forward)

def test_checkpoint_roundtrip():
    import tempfile, os
    model = build_model(encoder="resnet101", encoder_weights=None, num_classes=25)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    save_checkpoint(model, opt, epoch=3, best_miou=0.42, path=path)
    loaded_model, ckpt = load_checkpoint(path, device=torch.device("cpu"),
                                          encoder="resnet101", num_classes=25)
    assert ckpt["epoch"] == 3
    assert abs(ckpt["best_miou"] - 0.42) < 1e-6
    os.unlink(path)
    return f"epoch={ckpt['epoch']}, best_miou={ckpt['best_miou']}"

run("checkpoint save + load roundtrip", test_checkpoint_roundtrip)

# ---------------------------------------------------------------------------
# 3. Transforms
# ---------------------------------------------------------------------------
print("\n=== Transforms ===")
from utils.transforms import get_train_transforms, get_val_transforms, denormalize

def test_train_transform():
    t = get_train_transforms((256, 256))
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mask = np.random.randint(0, 24, (480, 640), dtype=np.uint8)
    out = t(image=img, mask=mask)
    assert out["image"].shape == (3, 256, 256), f"image shape: {out['image'].shape}"
    assert out["mask"].shape == (256, 256), f"mask shape: {out['mask'].shape}"
    return f"image {tuple(out['image'].shape)}, mask {tuple(out['mask'].shape)}"

def test_val_transform():
    t = get_val_transforms((256, 256))
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mask = np.random.randint(0, 24, (720, 1280), dtype=np.uint8)
    out = t(image=img, mask=mask)
    assert out["image"].shape == (3, 256, 256)
    return f"image {tuple(out['image'].shape)}"

run("train transforms (random resize crop + flip)", test_train_transform)
run("val transforms (deterministic resize)", test_val_transform)

# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
print("\n=== Metrics ===")
from utils.metrics import SegmentationMetrics

def test_metrics():
    m = SegmentationMetrics(num_classes=25, ignore_index=0)
    # Perfect predictions for classes 1-5
    pred = torch.zeros(10, 64, 64, dtype=torch.long)
    gt   = torch.zeros(10, 64, 64, dtype=torch.long)
    for c in range(1, 6):
        pred[:, c*10:(c+1)*10, :] = c
        gt[:, c*10:(c+1)*10, :]   = c
    m.update(pred, gt)
    res = m.compute()
    assert res["mIoU"] == 1.0, f"Expected mIoU=1.0 for perfect preds, got {res['mIoU']}"
    return f"mIoU={res['mIoU']:.4f}, pix_acc={res['pixel_accuracy']:.4f}"

def test_metrics_reset():
    m = SegmentationMetrics(num_classes=25, ignore_index=0)
    pred = torch.ones(2, 32, 32, dtype=torch.long)
    gt   = torch.ones(2, 32, 32, dtype=torch.long)
    m.update(pred, gt)
    m.reset()
    # After reset, confusion matrix should be all zeros — compute returns NaN mIoU
    import numpy as np
    res = m.compute()
    assert np.isnan(res["mIoU"]) or res["mIoU"] == 0.0
    return "reset clears state correctly"

run("perfect predictions → mIoU=1.0", test_metrics)
run("reset clears confusion matrix", test_metrics_reset)

# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------
print("\n=== Visualization ===")
from utils.visualization import colorize_mask, overlay_mask

def test_colorize():
    mask = np.random.randint(0, 24, (256, 256), dtype=np.uint8)
    rgb = colorize_mask(mask)
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
    return f"output shape {rgb.shape}"

def test_overlay():
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask  = np.random.randint(0, 24, (256, 256), dtype=np.uint8)
    out = overlay_mask(image, mask, alpha=0.5)
    assert out.shape == (256, 256, 3)
    return f"output shape {out.shape}"

run("colorize_mask", test_colorize)
run("overlay_mask", test_overlay)

# ---------------------------------------------------------------------------
# 6. Synthetic end-to-end: forward + loss + backward
# ---------------------------------------------------------------------------
print("\n=== End-to-end: forward + loss + backward ===")

def test_training_step():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(encoder="resnet101", encoder_weights=None, num_classes=25).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    model.train()
    B, H, W = 2, 256, 256
    images = torch.randn(B, 3, H, W, device=device)
    masks  = torch.randint(0, 25, (B, H, W), device=device)

    t0 = time.time()
    logits = model(images)
    loss = criterion(logits, masks)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    elapsed = time.time() - t0

    return f"loss={loss.item():.4f}, device={device}, step_time={elapsed:.2f}s"

run("single training step (random data)", test_training_step)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Dataset (no RUGD data needed for this check) ===")
from data.rugd_dataset import RUGD_CLASSES, RUGD_COLORMAP, GROUND_CLASS_INDICES, RUGD_SPLITS

run("RUGD_CLASSES length", lambda: f"{len(RUGD_CLASSES)} classes (0=void … 24=picnic-table)")
run("RUGD_COLORMAP shape", lambda: f"{RUGD_COLORMAP.shape}")
run("GROUND_CLASS_INDICES", lambda: f"{sorted(GROUND_CLASS_INDICES)}")
run("splits defined", lambda: {k: len(v) for k, v in RUGD_SPLITS.items()})

print("\n" + "="*50)
print("Done. Provide RUGD data path in configs/config.yaml to train.")
print("="*50 + "\n")
