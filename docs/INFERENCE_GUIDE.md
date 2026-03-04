# Inference Guide: Running the Model on Real-World Data

This guide explains how to run the trained DeepLabV3+ model on your own images or videos and how to interpret the outputs and metrics — even when you have **no ground-truth labels**.

---

## Quick Start

```bash
# 1. Edit the config (image path + checkpoint path)
nano configs/custom_dataset.yaml

# 2. Run
uv run python infer_custom.py --config configs/custom_dataset.yaml

# 3. Results land in:   inference_output/
#                       inference_output/inference_metrics.json
```

You can also pass overrides directly without editing the config:

```bash
uv run python infer_custom.py \
    --config configs/custom_dataset.yaml \
    --checkpoint runs/deeplabv3plus_rugd/checkpoints/best.pt \
    --input  /path/to/my/images/ \
    --output results/field_test_01/
```

---

## Output Files

After running `infer_custom.py`, the output directory contains:

| Path | Description |
|------|-------------|
| `masks/*.png` | Raw class-index PNGs (0–24 = class index). Lossless, suitable for downstream algorithms. |
| `colour/*.png` | Colourised class maps (each class gets a fixed RGB colour). Good for quick visual inspection. |
| `overlay/*.png` | Original image with colourised mask blended on top. Best for presentations. |
| `entropy/*_entropy.png` | Uncertainty heatmap — bright pixels = model is unsure. |
| `comparisons/*_compare.png` | Side-by-side: original \| GT \| prediction *(only when labels are provided)*. |
| `inference_metrics.json` | All scalar metrics (see below). |
| `class_coverage.png` | Bar chart of predicted class proportions. |
| `metrics_summary.txt` | Human-readable metrics table *(labelled mode only)*. |

---

## Metrics You Can Get Without Labels

When you run on **unlabelled images** (no annotation masks), you cannot compute accuracy metrics. Instead, `infer_custom.py` computes the following:

### 1. Confidence / Uncertainty

The model outputs a probability for each class at each pixel (softmax). From these we derive:

| Metric | What it means | Good sign | Bad sign |
|--------|---------------|-----------|----------|
| `mean_confidence` | Average max-softmax prob per image | Close to 1.0 | Below 0.5 — model is very unsure |
| `min_confidence` | Lowest-confidence pixel in the image | — | Near 0.0 — at least one region is ambiguous |
| `low_conf_pixel_fraction` | Fraction of pixels below the confidence threshold | Near 0.0 | High — large uncertain regions |
| `mean_entropy` | Average Shannon entropy (nats) | Near 0.0 | High — uncertainty spread across image |
| `max_entropy` | Highest entropy pixel | — | Close to ln(25) ≈ 3.22 — uniform confusion |
| `entropy_std` | How localised vs spread the uncertainty is | High std — uncertainty in specific regions only | Low std — uniform uncertainty everywhere |

**How to use entropy maps:** Open `entropy/*.png`. Blue/dark areas = confident predictions. Yellow/bright areas = uncertain. If uncertainty concentrates near class boundaries, that is normal. If whole regions are uncertain, the domain may be very different from the RUGD training set.

### 2. Class Coverage

How much of the scene is predicted as each class:

| Metric | What it means |
|--------|---------------|
| `coverage_by_name` | Fraction of pixels per class (e.g. `{"grass": 0.35, "dirt": 0.20, ...}`) |
| `ground_coverage` | Combined fraction of all traversable ground classes (dirt, sand, grass, asphalt, gravel, mulch, rock-bed, water) |
| `dominant_class` | Index of the most-predicted class |

**How to use:** Check `class_coverage.png`. A healthy outdoor scene should have a significant `ground_coverage` (> 0.3). If `void` (class 0) dominates, the model is confused and likely encountering out-of-distribution images.

### 3. Boundary Quality

| Metric | What it means |
|--------|---------------|
| `edge_density` | Fraction of pixels that lie on a class boundary (horizontal or vertical neighbour differs) |
| `boundary_fragmentation` | Edge density normalised by number of classes present — high = many small fragments |

**How to use:** High `edge_density` (> 0.15) combined with high `boundary_fragmentation` suggests noisy, salt-and-pepper predictions — a sign of domain mismatch or under-trained model. A good model on clean images typically produces `edge_density` of 0.03–0.10.

### 4. Temporal Consistency (Video Only)

If your input is a video file (MP4 / AVI), consecutive frames are compared:

| Metric | What it means |
|--------|---------------|
| `mean_frame_iou` | Average IoU between predictions of consecutive frames. Higher = more stable. |
| `flicker_rate` | Fraction of pixels that change class between consecutive frames. Lower = more stable. |
| `stable_pixel_frac` | 1 - flicker_rate |

**How to use:** A well-trained model on smooth video should have `mean_frame_iou` > 0.85 and `flicker_rate` < 0.05. High flicker on static camera (no motion) suggests the model is unstable, likely due to domain shift.

### 5. Throughput

| Metric | What it means |
|--------|---------------|
| `fps` | Frames per second (images processed per second) |
| `ms_per_image` | Average inference time per image |
| `ms_per_batch` | Average inference time per batch |

These are measured during a separate benchmarking pass (configurable warmup + timed runs).

---

## Metrics Available WITH Labels

If you have annotation masks, set `label_dir` in the config and you get the full supervised evaluation suite:

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union across all non-void classes. Primary benchmark metric. |
| **Pixel accuracy** | Fraction of pixels with correct class. Biased toward large classes. |
| **Mean class accuracy** | Per-class accuracy averaged across classes. More balanced than pixel accuracy. |
| **FWIoU** | Frequency-Weighted IoU — IoU weighted by class frequency. |
| **Per-class IoU** | IoU for every individual class. Identifies which classes the model handles well/poorly. |
| **Per-class Precision/Recall/F1** | Precision = correct / predicted, Recall = correct / actual. |
| **Confusion matrix** | (C × C) matrix showing which classes are confused with each other. |

These are saved in `metrics_summary.txt`, `metrics.json`, and as chart PNGs.

---

## Interpreting Your Results

### "The model output looks strange on my images"

Check the following:

1. **Image preprocessing**: The model was trained with ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). `infer_custom.py` applies this automatically.
2. **Image size**: The model expects 512×512 inputs. Larger/smaller images are resized automatically, but extreme aspect ratios may look distorted.
3. **Domain gap**: RUGD is a trail/off-road dataset. Indoor scenes, roads with painted markings, or urban environments will produce worse predictions. Fine-tuning is recommended (see CUSTOM_DATASET_GUIDE.md).

### "Confidence is very low everywhere"

This means the model is encountering images that look very different from RUGD. Common causes:
- Indoor environment (model was trained outdoors)
- Heavy rain, fog, or night-time conditions
- Very different camera characteristics (e.g., wide-angle fisheye vs standard lens)

**Solution**: Fine-tune on a small set of labelled frames from your environment (see CUSTOM_DATASET_GUIDE.md).

### "Ground coverage is near zero"

The traversable ground classes (dirt, sand, grass, asphalt, etc.) are not being detected. Either:
- The scene genuinely contains little ground (looking up, side view, etc.)
- The model is misclassifying ground as another class — check the colour overlay to see what class is dominating

### "Predictions look good but IoU is low"

This often happens when:
- Your label format is different (check that `label_dir` contains correctly formatted annotations)
- Class indices don't match (the model uses RUGD's 25-class scheme)
- You need to use the `remap` feature to collapse your classes into RUGD classes

---

## Output Format Options

Set `save_format` in `configs/custom_dataset.yaml`:

| Value | Saved files | Best for |
|-------|-------------|---------|
| `"mask"` | Raw class-index PNGs only | Downstream processing, feeding to another model |
| `"colour"` | Colourised PNG only | Quick visual inspection |
| `"overlay"` | Blended original + mask (default) | Presentations, reports |
| `"all"` | All three formats | Full analysis |

---

## Command-Line Reference

```bash
uv run python infer_custom.py --help

# Arguments:
#   --config      YAML config file (required)
#   --checkpoint  Override checkpoint path in config
#   --input       Override dataset.image_dir in config
#   --output      Override infer.output_dir in config
```

---

## See Also

- `configs/custom_dataset.yaml` — all settings with inline documentation
- `docs/CUSTOM_DATASET_GUIDE.md` — how to organise your own data and fine-tune
- `evaluate.py` — evaluate on labelled RUGD test split
- `utils/inference_metrics.py` — Python API for all unlabelled metrics
