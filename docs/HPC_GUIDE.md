# HPC Deployment Guide

Step-by-step guide to go from your local machine to a running training job on an HPC cluster with SLURM, no Docker required.

---

## One-Command Launch (TL;DR)

```bash
# SSH into the HPC login node, clone the project, then:
cd ~/segmentation_model
bash hpc/launch.sh --sample          # download sample data + submit job

# With W&B monitoring:
bash hpc/launch.sh --sample --wandb-key YOUR_WANDB_API_KEY

# With your own pre-existing RUGD data:
bash hpc/launch.sh --data /scratch/username/RUGD
```

That's it. The script handles: uv install → venv → data download → config → sbatch.

---

## Step-by-Step Breakdown

### 1. Transfer the project to HPC

From your **local machine**, transfer the project (exclude the venv and cached data):

```bash
rsync -avz \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'data/RUGD' \
  --exclude 'runs/' \
  segmentation_model/ \
  <username>@<hpc-hostname>:~/segmentation_model/
```

**Alternatives:**

```bash
# If the HPC has GitHub access:
git clone https://github.com/<you>/segmentation_model ~/segmentation_model

# If using a shared filesystem (e.g., /project/), just copy:
cp -r segmentation_model/ /project/<group>/segmentation_model/
```

---

### 2. SSH into the Login Node

```bash
ssh <username>@<hpc-hostname>
cd ~/segmentation_model
```

> **Important:** All setup commands below run on the **login node** (non-GPU). The actual training runs on compute nodes via SLURM.

---

### 3. Customise the Cluster Settings

Before launching, edit two files to match your cluster:

**`hpc/launch.sh`** — top section:
```bash
PARTITION="gpu"      # ← your GPU partition (run: sinfo to list options)
TIME="24:00:00"      # ← max wall time allowed by your cluster
GPUS="1"             # ← usually 1 for ResNet-101
MEM="32G"            # ← 32 GB is safe; reduce to 16G if quota is tight
CPUS="8"             # ← match --cpus-per-task in your job script
```

**`hpc/setup_env.sh`** — module names:
```bash
# Find your cluster's exact module names:
module spider Python   # lists Python versions
module spider CUDA     # lists CUDA versions

# Then update these lines in setup_env.sh:
module load Python/3.11.5-GCCcore-13.2.0   # ← change to what your cluster has
module load CUDA/12.1.0                      # ← change to your cluster's CUDA
```

---

### 4. Dataset Setup

#### Option A — Download from HuggingFace (easiest)

```bash
# Sample (75 images, ~60 MB) — for a quick pipeline test:
python data/download_rugd.py --output data/RUGD

# Full dataset (~7,500 images, ~3 GB):
python data/download_rugd.py --output data/RUGD --full

# Specific sequences only:
python data/download_rugd.py --output data/RUGD --sequences creek park-2 trail-7
```

#### Option B — Use your existing RUGD data

If you already have RUGD downloaded (from the official site rugd.vision):

```bash
# Verify the structure looks like this:
ls /path/to/RUGD/
# RUGD_frames-with-annotations/  RUGD_annotations/

# Then pass it directly to launch.sh:
bash hpc/launch.sh --data /path/to/RUGD
```

#### Option C — Use cluster shared storage

Many HPC clusters have shared datasets. Check with your admin:

```bash
ls /datasets/  OR  ls /scratch/shared/  OR  ls /project/datasets/
# If RUGD is there, use: bash hpc/launch.sh --data /datasets/RUGD
```

#### Expected Directory Layout

```
data/RUGD/
├── RUGD_frames-with-annotations/
│   ├── creek/
│   │   ├── creek_00001.png   ← RGB image (720×1280)
│   │   └── ...
│   ├── trail/
│   └── ...
└── RUGD_annotations/
    ├── creek/
    │   ├── creek_00001.png   ← label mask (pixel value = class index 0-23)
    │   └── ...
    └── ...
```

---

### 5. Set Up the Environment (One Time)

```bash
bash hpc/setup_env.sh
```

This script:
1. Loads Python + CUDA modules
2. Installs `uv` to `~/.local/bin`
3. Creates `.venv/` with Python 3.11
4. Detects your CUDA version and installs the matching PyTorch build
5. Installs all other dependencies

After setup, activate the venv in any interactive session:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"  # should print True
```

---

### 6. Configure W&B (Optional but Recommended)

Weights & Biases gives you a live dashboard with loss curves, mIoU, and prediction previews.

```bash
# Option A: login interactively on the login node (saves key to ~/.netrc)
wandb login

# Option B: set environment variable (better for SLURM — no interactive prompt)
export WANDB_API_KEY="your-api-key-here"
# Add to your ~/.bashrc to persist it:
echo 'export WANDB_API_KEY="your-api-key-here"' >> ~/.bashrc

# Option C: pass to launch.sh directly
bash hpc/launch.sh --wandb-key "your-api-key-here"
```

Get your API key at: https://wandb.ai/settings → API keys

Then enable W&B in the config:
```yaml
# configs/config.yaml
wandb:
  enabled: true
  project: "rugd-segmentation"
  entity: "your-username-or-team"   # null = personal account
```

Or activate it at launch time:
```bash
python train.py --wandb   # enables W&B without editing config
```

---

### 7. Submit the Training Job

#### One-command (recommended):

```bash
bash hpc/launch.sh --data /path/to/RUGD
```

#### Manual submission:

```bash
# Edit hpc/train.slurm first, then:
sbatch hpc/train.slurm
```

#### Monitor the job:

```bash
squeue -u $USER                          # see all your jobs
squeue -j <JOB_ID>                       # specific job status
scancel <JOB_ID>                         # cancel a job

# Live log output:
tail -f logs/train_<JOB_ID>.log

# Check GPU usage (on compute node):
srun --jobid=<JOB_ID> nvidia-smi
```

---

### 8. Monitor Training

#### TensorBoard (from HPC)

```bash
# On the HPC node:
tensorboard --logdir runs/ --host 0.0.0.0 --port 6006 &

# On your LOCAL machine — create SSH tunnel:
ssh -L 6006:localhost:6006 <username>@<hpc-hostname>

# Then open in browser: http://localhost:6006
```

> Some clusters require tunnelling through a specific port or using `srun` to run on a compute node. Check your cluster's documentation for "interactive TensorBoard".

#### W&B Dashboard

Simply open https://wandb.ai in your browser. Everything is logged automatically if `wandb.enabled: true`.

The W&B run will show:
- **Charts** — `train/loss`, `val/mIoU`, `val/pixel_accuracy`, LR schedule
- **Media** — prediction grids every N epochs (side-by-side: image | GT | prediction)
- **System** — GPU utilisation, memory, temperature
- **Config** — all hyperparameters logged automatically

---

### 9. Download Results

After training completes, copy the best checkpoint and results back to your local machine:

```bash
# From your LOCAL machine:
rsync -avz \
  <username>@<hpc-hostname>:~/segmentation_model/runs/ \
  ./runs/
```

---

## Config Reference

Key settings to tune for your HPC environment (`configs/config.yaml`):

| Setting | Default | HPC Note |
|---------|---------|----------|
| `data.root_dir` | `data/RUGD` | Use absolute path on HPC |
| `data.num_workers` | `4` | Set to `--cpus-per-task` value |
| `data.image_size` | `[512, 512]` | Reduce to `[384, 384]` if GPU OOM |
| `train.batch_size` | `8` | Increase if GPU has >16 GB VRAM |
| `train.use_amp` | `true` | Keep true on CUDA for speed |
| `model.encoder_weights` | `imagenet` | First run downloads ~170 MB |
| `wandb.enabled` | `false` | Set true for dashboard |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
→ The venv isn't activated. Run: `source .venv/bin/activate`

**`RuntimeError: CUDA out of memory`**
→ Reduce `data.image_size` to `[384, 384]` or `train.batch_size` to `4`.

**`FileNotFoundError: RUGD directory not found`**
→ Check `data.root_dir` in config. Use the absolute path.

**Job runs but GPU is idle / `torch.cuda.is_available()` = False**
→ The CUDA module wasn't loaded. Check `module list` in the job log and compare with `module spider CUDA`.

**W&B "offline" mode / not syncing**
→ `WANDB_API_KEY` isn't set. Run `wandb login` or `export WANDB_API_KEY=...` in `~/.bashrc`.

**Job killed before finishing (`TIMEOUT` / `OOM`)**
→ The job exceeded `--time` or `--mem`. Resume training from the latest checkpoint:
```bash
# Edit hpc/train.slurm to add --resume:
python train.py --config configs/exp_<TIMESTAMP>.yaml \
    --resume runs/<RUN_NAME>/checkpoints/latest.pt
```

**Slow data loading**
→ Increase `data.num_workers` to match `--cpus-per-task`. Also consider copying the dataset to local node scratch (`$TMPDIR` or `/tmp`) at job start for faster I/O on some clusters.

---

## Multi-GPU Training

The current setup uses a single GPU. For multi-GPU:

1. Change `--gres=gpu:1` to `--gres=gpu:4` (or however many you want) in the SLURM script.
2. Use `torchrun` instead of `python`:

```bash
# In train.slurm, replace the python line with:
torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE train.py --config ...
```

3. Wrap the model in `train.py`:
```python
if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(model)
```

This is left as a future extension — single-GPU is sufficient for ResNet-101 on RUGD.
