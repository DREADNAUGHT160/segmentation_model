#!/usr/bin/env bash
# =============================================================================
# ONE-COMMAND HPC LAUNCHER
#
# Does everything needed to go from a fresh clone to a running SLURM job:
#   1. Checks / installs uv
#   2. Sets up the Python virtual environment
#   3. Optionally downloads a sample RUGD dataset
#   4. Updates config.yaml with the correct data path
#   5. Submits the SLURM training job
#   6. Prints live monitoring commands
#
# Usage:
#   bash hpc/launch.sh                        # interactive prompts
#   bash hpc/launch.sh --data /path/to/RUGD  # skip data download
#   bash hpc/launch.sh --sample              # download sample (75 images)
#   bash hpc/launch.sh --full                # download full dataset (~3 GB)
#   bash hpc/launch.sh --dry-run             # setup only, don't sbatch
#
# Prerequisites:
#   - Run from the project root: cd ~/segmentation_model && bash hpc/launch.sh
#   - Your cluster must have CUDA-capable GPUs and SLURM
#   - Edit PARTITION and TIME below to match your cluster
# =============================================================================

set -euo pipefail

# ---- Cluster config — EDIT THESE ----------------------------------------
PARTITION="gpu"            # Your GPU partition name (check with: sinfo)
TIME="24:00:00"            # Max wall time
GPUS="1"                   # Number of GPUs per job
CPUS="8"                   # CPU cores per task
MEM="32G"                  # RAM
# -------------------------------------------------------------------------

# ---- Parse arguments -----------------------------------------------------
DATA_DIR=""
DOWNLOAD_MODE=""           # "sample" | "full" | ""
DRY_RUN=false
WANDB_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)       DATA_DIR="$2"; shift 2 ;;
    --sample)     DOWNLOAD_MODE="sample"; shift ;;
    --full)       DOWNLOAD_MODE="full"; shift ;;
    --dry-run)    DRY_RUN=true; shift ;;
    --wandb-key)  WANDB_KEY="$2"; shift 2 ;;
    --partition)  PARTITION="$2"; shift 2 ;;
    --time)       TIME="$2"; shift 2 ;;
    --gpus)       GPUS="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | grep -v '#!/' | sed 's/^# *//'
      exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# =========================================================================
# Step 1: Install uv
# =========================================================================
echo "=== [1/5] Checking uv ==="
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bash_profile" 2>/dev/null || true
else
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# =========================================================================
# Step 2: Set up virtual environment
# =========================================================================
echo ""
echo "=== [2/5] Setting up environment ==="

if [[ ! -f ".venv/bin/python" ]]; then
  echo "Creating virtual environment..."
  bash hpc/setup_env.sh
else
  echo "Virtual environment already exists (.venv/) — skipping setup."
  echo "  To force re-setup: rm -rf .venv && bash hpc/launch.sh"
fi

source .venv/bin/activate

# =========================================================================
# Step 3: Download / verify RUGD data
# =========================================================================
echo ""
echo "=== [3/5] Dataset ==="

if [[ -n "$DATA_DIR" ]]; then
  # User provided a path — just verify it
  if [[ ! -d "$DATA_DIR/RUGD_frames-with-annotations" ]]; then
    echo "ERROR: $DATA_DIR does not look like a valid RUGD root."
    echo "  Expected: $DATA_DIR/RUGD_frames-with-annotations/"
    exit 1
  fi
  echo "Using existing dataset at: $DATA_DIR"
  RUGD_PATH="$DATA_DIR"

elif [[ -d "$PROJECT_DIR/data/RUGD/RUGD_frames-with-annotations" ]]; then
  echo "Found existing dataset at: $PROJECT_DIR/data/RUGD"
  RUGD_PATH="$PROJECT_DIR/data/RUGD"

else
  # Prompt if not specified
  if [[ -z "$DOWNLOAD_MODE" ]]; then
    echo "No RUGD dataset found."
    echo "Options:"
    echo "  1) Download sample  (~75 images, ~60 MB, fast)"
    echo "  2) Download full    (~7500 images, ~3 GB, slow)"
    echo "  3) Enter path to existing dataset"
    echo "  4) Skip (configure data manually later)"
    read -r -p "Choice [1/2/3/4]: " CHOICE
    case "$CHOICE" in
      1) DOWNLOAD_MODE="sample" ;;
      2) DOWNLOAD_MODE="full" ;;
      3)
        read -r -p "Path to RUGD root: " DATA_DIR
        RUGD_PATH="$DATA_DIR" ;;
      *) RUGD_PATH="" ;;
    esac
  fi

  if [[ "$DOWNLOAD_MODE" == "sample" ]]; then
    echo "Downloading sample RUGD dataset (~75 images)..."
    python data/download_rugd.py --output "$PROJECT_DIR/data/RUGD"
    RUGD_PATH="$PROJECT_DIR/data/RUGD"
  elif [[ "$DOWNLOAD_MODE" == "full" ]]; then
    echo "Downloading full RUGD dataset (~3 GB, this will take a while)..."
    python data/download_rugd.py --output "$PROJECT_DIR/data/RUGD" --full
    RUGD_PATH="$PROJECT_DIR/data/RUGD"
  fi
fi

# =========================================================================
# Step 4: Write an experiment-specific config
# =========================================================================
echo ""
echo "=== [4/5] Writing experiment config ==="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="slurm_${TIMESTAMP}"
EXP_CONFIG="$PROJECT_DIR/configs/exp_${TIMESTAMP}.yaml"

# Copy base config and patch data path and run name
python - <<PYEOF
import yaml, copy, sys
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["root_dir"] = "${RUGD_PATH:-data/RUGD}"
cfg["output"]["run_name"] = "${RUN_NAME}"
cfg["data"]["num_workers"] = ${CPUS}
cfg["data"]["pin_memory"] = True
with open("${EXP_CONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print("Config written to: ${EXP_CONFIG}")
PYEOF

echo "Run name: $RUN_NAME"
echo "Config  : $EXP_CONFIG"

# =========================================================================
# Step 5: Submit SLURM job
# =========================================================================
echo ""
echo "=== [5/5] Submitting SLURM job ==="

# Write a self-contained job script for this run
JOB_SCRIPT="$LOG_DIR/job_${TIMESTAMP}.slurm"
WANDB_EXPORT=""
if [[ -n "$WANDB_KEY" ]]; then
  WANDB_EXPORT="export WANDB_API_KEY=${WANDB_KEY}"
fi

cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name=seg_${TIMESTAMP}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/train_%j.log
#SBATCH --error=${LOG_DIR}/train_%j.err

set -euo pipefail
echo "Job started: \$(date)"
echo "Node: \$SLURMD_NODENAME"

module purge
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || module load python/3.11 2>/dev/null || true
module load CUDA/12.1.0 2>/dev/null || module load cuda/12.1 2>/dev/null || true

cd ${PROJECT_DIR}
source .venv/bin/activate
${WANDB_EXPORT}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python train.py --config ${EXP_CONFIG} --wandb

echo "Job complete: \$(date)"
SLURM

if [[ "$DRY_RUN" == "true" ]]; then
  echo ""
  echo "DRY RUN — job script written but NOT submitted."
  echo "  Job script: $JOB_SCRIPT"
  echo "  To submit manually: sbatch $JOB_SCRIPT"
else
  JOB_ID=$(sbatch "$JOB_SCRIPT" | grep -oP '\d+')
  echo ""
  echo "============================================================"
  echo "Job submitted! ID: ${JOB_ID}"
  echo ""
  echo "Monitor:"
  echo "  squeue -j ${JOB_ID}              # job status"
  echo "  tail -f ${LOG_DIR}/train_${JOB_ID}.log   # live log"
  echo ""
  echo "TensorBoard:"
  echo "  tensorboard --logdir runs/${RUN_NAME}/tb_logs --host 0.0.0.0"
  echo "  Then SSH tunnel: ssh -L 6006:localhost:6006 <user>@<hpc>"
  echo ""
  echo "W&B: https://wandb.ai  (if --wandb-key was set)"
  echo "============================================================"
fi
