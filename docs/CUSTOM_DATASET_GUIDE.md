# Custom Dataset Guide

This guide explains how to plug in your own images (and optionally labels) so you can run inference, evaluate the model, or fine-tune it on your specific environment.

---

## Part 1: Running Inference on Your Images (No Labels Needed)

This is the fastest path — no annotation required.

### Step 1: Organise Your Images

Put your images in a folder. Any of the supported formats work:

```
my_data/
└── images/
    ├── frame_0001.jpg
    ├── frame_0002.jpg
    └── ...
```

Or just point at a video file directly (`my_robot_drive.mp4`).

### Step 2: Edit the Config

Copy `configs/custom_dataset.yaml` and fill in your paths:

```bash
cp configs/custom_dataset.yaml configs/my_robot.yaml
nano configs/my_robot.yaml
```

Minimum required changes:

```yaml
dataset:
  image_dir: "my_data/images"   # ← your images folder (or video path)

model:
  checkpoint: "runs/deeplabv3plus_rugd/checkpoints/best.pt"   # ← your checkpoint
```

### Step 3: Run

```bash
uv run python infer_custom.py --config configs/my_robot.yaml
```

Results will appear in `inference_output/` (override with `--output my_results/`).

---

## Part 2: Evaluating with Your Own Labels

If you have annotation masks (PNG files), the script will compute full metrics (mIoU, pixel accuracy, confusion matrix, etc.).

### Label Format

The easiest format is **RGB colourmap PNGs** using the RUGD colour scheme — the same format as the RUGD dataset. See `data/rugd_dataset.py → RUGD_COLORMAP` for the exact RGB values.

If your annotations are **single-channel index PNGs** (pixel value = class index), that also works — the loader detects single-channel images automatically.

### Folder Layout

```
my_data/
├── images/
│   ├── frame_0001.jpg
│   └── frame_0002.jpg
└── labels/
    ├── frame_0001.png   ← same filename stem as the image
    └── frame_0002.png
```

### Config

```yaml
dataset:
  image_dir: "my_data/images"
  label_dir: "my_data/labels"   # ← add this

labelled_metrics:
  generate_report: true
  save_comparisons: true
```

### Run

```bash
uv run python infer_custom.py --config configs/my_robot.yaml
```

After running, you will find:
- `inference_output/metrics_summary.txt` — per-class IoU table
- `inference_output/metrics.json` — all numbers in JSON
- `inference_output/per_class_iou.png` — bar chart
- `inference_output/confusion_matrix.png` — heatmap
- `inference_output/comparisons/*.png` — side-by-side image | GT | prediction

---

## Part 3: Using a Custom Class Set (Remap)

If your environment only has a subset of RUGD classes (e.g. you only care about `grass`, `asphalt`, and `dirt`), you can use the `remap` feature to collapse classes:

```yaml
remap:
  # Collapse all ground-like classes into a single "traversable" class
  dirt:     traversable
  sand:     traversable
  grass:    traversable
  asphalt:  traversable
  gravel:   traversable
  # Everything else becomes "void" (ignored in metrics)
```

> **Note**: The `remap` feature is currently reserved for future implementation. For now, use the RUGD 25-class output directly and focus on the classes relevant to your environment when reading per-class IoU values.

---

## Part 4: Fine-Tuning on Your Own Data

If inference quality is poor (e.g. low confidence, wrong class predictions), fine-tuning on a small set of labelled frames from your actual environment makes a large difference. Even 50–100 annotated frames can significantly improve performance.

### 4a: Annotate Your Images

Recommended free tools:

| Tool | Notes |
|------|-------|
| [LabelMe](https://github.com/labelmeai/labelme) | Polygon annotation, exports to PNG |
| [CVAT](https://cvat.org) | Web-based, supports semantic segmentation |
| [Labelbox](https://labelbox.com) | Cloud-based, free tier available |

When annotating:
1. Use the **RUGD class set** (25 classes) if possible — this lets you start from the pretrained weights
2. If your environment has classes not in RUGD (e.g. `road_line`, `curb`), you will need to extend the number of classes and retrain from scratch
3. Export labels as single-channel index PNGs or RGB colourmap PNGs

### 4b: Organise the Data

Follow the RUGD layout pattern:

```
my_robot_data/
├── train/
│   ├── images/
│   └── annotations/
└── val/
    ├── images/
    └── annotations/
```

### 4c: Create a Training Config

Copy `configs/config.yaml` and modify:

```yaml
data:
  root_dir: "my_robot_data"
  num_classes: 25
  image_size: [512, 512]

train:
  epochs: 30
  batch_size: 4
  lr: 0.0001        # lower than default — fine-tuning
  freeze_encoder: false
```

### 4d: Fine-Tune

```bash
uv run python train.py \
    --config configs/my_robot_config.yaml \
    --checkpoint runs/deeplabv3plus_rugd/checkpoints/best.pt
```

The `--checkpoint` flag loads the pretrained weights as a starting point. Training will resume from those weights, converging much faster than training from scratch.

### 4e: Monitor Training

Enable W&B for live dashboards:

```bash
uv run python train.py \
    --config configs/my_robot_config.yaml \
    --checkpoint runs/deeplabv3plus_rugd/checkpoints/best.pt \
    --wandb
```

---

## Part 5: Using a File List

If you want to run inference on a specific subset of images (e.g. from a validation split), create a text file listing the paths:

```
# val_files.txt
my_data/images/frame_0010.jpg
my_data/images/frame_0020.jpg
my_data/images/frame_0030.jpg
```

Then in the config:

```yaml
dataset:
  image_dir: "my_data/images"
  file_list: "my_data/val_files.txt"   # only these images are processed
```

---

## Part 6: Video Input

Point `image_dir` at a video file instead of a directory:

```yaml
dataset:
  image_dir: "my_robot_drive.mp4"
```

The script will:
1. Extract all frames from the video
2. Run segmentation on each frame
3. Save outputs as individual PNGs (named `frame_000000.png`, etc.)
4. Compute temporal consistency metrics (frame flicker, frame-to-frame IoU)

For video output (writing a segmented video file), use `infer.py` instead:

```bash
uv run python infer.py \
    --checkpoint runs/deeplabv3plus_rugd/checkpoints/best.pt \
    --input my_robot_drive.mp4 \
    --output segmented_drive.mp4 \
    --overlay --alpha 0.5
```

---

## Troubleshooting

### Images are all predicted as "void" (black output)

- Check that `num_classes` in the config matches the checkpoint (should be 25 for RUGD-pretrained)
- Check image dimensions are reasonable (e.g. not 1×1 pixel)
- Run with `save_format: "all"` to see all output types

### "Label not found for ..." warning

The label file stem must exactly match the image file stem. For example:
- Image: `images/frame_0001.jpg`
- Label: `labels/frame_0001.png` ✓
- Label: `labels/frame_0001_label.png` ✗ (different stem)

### Low mIoU on my data (labelled evaluation)

This is expected if you're applying the RUGD-trained model to a different domain. Fine-tune the model on your data (see Part 4) or check that your label format is correct.

### Memory error during inference

Reduce `batch_size` in the config:

```yaml
infer:
  batch_size: 1
```

---

## Configuration Reference

All options in `configs/custom_dataset.yaml` with their defaults:

```yaml
dataset:
  image_dir: "my_data/images"     # required
  label_dir: null                  # optional — enables labelled metrics
  file_list: null                  # optional — subset selection
  extensions: ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]

model:
  encoder: "resnet101"
  num_classes: 25
  checkpoint: "runs/.../best.pt"  # required

infer:
  image_size: [512, 512]
  batch_size: 4
  num_workers: 4
  output_dir: "inference_output"
  save_format: "overlay"           # mask | colour | overlay | all
  overlay_alpha: 0.55
  video_fps: null                  # null = match source FPS

unlabelled_metrics:
  save_entropy_maps: true
  low_conf_threshold: 0.6
  benchmark_fps: true
  benchmark_warmup: 5
  benchmark_runs: 20

labelled_metrics:
  generate_report: true
  save_comparisons: true

remap: null                        # advanced — class remapping
```

---

## See Also

- `docs/INFERENCE_GUIDE.md` — detailed explanation of every metric
- `docs/HPC_GUIDE.md` — running on a SLURM cluster
- `evaluate.py` — evaluate on RUGD test split
- `train.py` — full training script
