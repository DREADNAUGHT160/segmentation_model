# Ground Surface Segmentation — DeepLabV3+ on RUGD

Semantic segmentation of ground surfaces in unstructured outdoor environments using **DeepLabV3+** (ResNet-101 backbone) trained on the **RUGD dataset** (24 classes).

---

## Dataset

The **RUGD (Robot Unstructured Ground Dataset)** contains 7,545 images (720×1280) across 24 semantic classes including dirt, grass, asphalt, gravel, sand, and more.  Website: http://rugd.vision/

> **Download requires filling out a form** on the RUGD website.  Once approved you will receive a link to download the archive.

After downloading, extract so the directory looks like:

```
data/RUGD/
├── RUGD_frames-with-annotations/
│   ├── creek/
│   │   ├── creek_00000.png
│   │   └── ...
│   └── trail/  ...
└── RUGD_annotations/
    ├── creek/
    │   ├── creek_00000.png   ← label mask (pixel value = class index 0-23)
    │   └── ...
    └── trail/  ...
```

Then update `configs/config.yaml → data.root_dir` to point at `data/RUGD`.

### Classes

| Index | Class | Ground? |
|------:|-------|---------|
| 0 | void | |
| 1 | dirt | ✓ |
| 2 | sand | ✓ |
| 3 | grass | ✓ |
| 4 | tree | |
| 5 | pole | |
| 6 | water | ✓ |
| 7 | sky | |
| 8 | vehicle | |
| 9 | container | |
| 10 | asphalt | ✓ |
| 11 | gravel | ✓ |
| 12 | building | |
| 13 | mulch | ✓ |
| 14 | rock-bed | ✓ |
| 15 | log | |
| 16 | bicycle | |
| 17 | person | |
| 18 | fence | |
| 19 | bush | |
| 20 | sign | |
| 21 | rock | |
| 22 | bridge | |
| 23 | obstacle | |

Ground-traversable classes are marked ✓.  You can filter predictions to these indices post-inference using `data.GROUND_CLASS_INDICES`.

---

## Local Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
cd segmentation_model/
uv sync

# Install CUDA-compatible PyTorch (replace cu121 with your CUDA version)
# Check your CUDA version with: nvcc --version  OR  nvidia-smi
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Activate the environment
source .venv/bin/activate
```

---

## Training

```bash
# Train with default config
uv run python train.py

# Override specific config values via CLI
uv run python train.py --run_name my_exp --epochs 50 --batch_size 4

# Resume from a checkpoint
uv run python train.py --resume runs/deeplabv3plus_rugd/checkpoints/latest.pt
```

Checkpoints are saved to `runs/<run_name>/checkpoints/`.
TensorBoard logs are in `runs/<run_name>/tb_logs/`.

```bash
# Monitor training in TensorBoard
tensorboard --logdir runs/
```

Key config values (edit `configs/config.yaml`):

| Key | Default | Notes |
|-----|---------|-------|
| `data.root_dir` | `data/RUGD` | Path to extracted RUGD dataset |
| `data.image_size` | `[512, 512]` | Resize H×W; reduce if OOM |
| `train.batch_size` | `8` | Per-GPU batch size |
| `train.epochs` | `100` | Total training epochs |
| `train.learning_rate` | `0.007` | Initial LR for SGD |
| `model.encoder` | `resnet101` | Backbone — see smp encoder list |

---

## Evaluation

```bash
# Compute mIoU on the test split
uv run python evaluate.py --checkpoint runs/deeplabv3plus_rugd/checkpoints/best.pt

# Also save side-by-side visualisations (image | GT | prediction)
uv run python evaluate.py --checkpoint best.pt --save_vis --vis_dir eval_vis/
```

---

## Inference

```bash
# Single image → segmentation mask PNG
uv run python infer.py --checkpoint best.pt --input photo.jpg --output pred.png

# With overlay (mask blended over the original image)
uv run python infer.py --checkpoint best.pt --input photo.jpg --output pred.png --overlay

# Directory of images
uv run python infer.py --checkpoint best.pt --input frames/ --output results/

# Video file
uv run python infer.py --checkpoint best.pt --input drive.mp4 --output drive_seg.mp4

# Video with overlay, custom alpha
uv run python infer.py --checkpoint best.pt --input drive.mp4 --output out.mp4 --overlay --alpha 0.6
```

---

## HPC / SLURM Deployment

### 1. Transfer files

Copy the project to your HPC home directory:

```bash
# From your local machine
rsync -avz --exclude '.venv' --exclude '__pycache__' \
    segmentation_model/ <user>@<hpc-host>:~/segmentation_model/
```

Also transfer the RUGD dataset (or use the cluster's shared data storage if available).

### 2. Set up the environment (run once on the login node)

```bash
cd ~/segmentation_model/
bash hpc/setup_env.sh
```

The script auto-detects the CUDA version and installs the matching PyTorch build.  If auto-detection fails, edit the `TORCH_CUDA` variable in the script.

### 3. Edit the SLURM script

Open `hpc/train.slurm` and adjust:
- `--partition` — your cluster's GPU partition name (`sinfo` to list options)
- `--time` — wall-time limit
- `--mail-user` — your email for job notifications
- `module load` lines — match your cluster's exact module names

### 4. Submit the job

```bash
sbatch hpc/train.slurm

# Monitor
squeue -u $USER
tail -f logs/train_<JOBID>.log
```

### Notes for HPC

- **No Docker** — the venv approach is fully compatible with HPC systems that prohibit containers.
- **Data path** — update `configs/config.yaml → data.root_dir` to the absolute path of RUGD on the cluster's storage.
- **num_workers** — increase `data.num_workers` to match `--cpus-per-task` in your SLURM script for faster data loading.
- **OOM** — if you hit GPU memory errors, reduce `data.image_size` to `[384, 384]` or lower `train.batch_size`.
- **Multi-GPU** — the current training script uses a single GPU.  For multi-GPU, wrap the model with `torch.nn.DataParallel` or use `torchrun` with `DistributedDataParallel`.

---

## Project Structure

```
segmentation_model/
├── configs/
│   └── config.yaml          # All hyperparameters and paths
├── data/
│   └── rugd_dataset.py      # Dataset class, class names, colormap, splits
├── models/
│   └── deeplabv3plus.py     # Model builder, checkpoint save/load
├── utils/
│   ├── metrics.py           # Streaming confusion-matrix IoU tracker
│   ├── transforms.py        # Albumentations train/val pipelines
│   └── visualization.py     # Colorize, overlay, save comparison
├── hpc/
│   ├── setup_env.sh         # One-time HPC venv setup (uv)
│   └── train.slurm          # SLURM job submission script
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation / test-split metrics
├── infer.py                 # Image / video inference
├── pyproject.toml           # uv project definition + dependencies
└── .python-version          # Pins Python 3.11
```
