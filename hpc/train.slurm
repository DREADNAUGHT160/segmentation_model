#!/bin/bash
# =============================================================================
# SLURM job script — DeepLabV3+ training on RUGD
#
# Submit with:
#   sbatch hpc/train.slurm
#
# Monitor with:
#   squeue -u $USER
#   tail -f logs/train_<JOBID>.log
#
# Adjust the #SBATCH directives below to match your cluster's partitions,
# GPU types, and time limits.  Run `sinfo` to see available partitions.
# =============================================================================

#SBATCH --job-name=seg_rugd
#SBATCH --partition=gpu              # Change to your cluster's GPU partition
#SBATCH --gres=gpu:1                 # Number of GPUs (1 is enough for ResNet-101)
#SBATCH --cpus-per-task=8            # DataLoader workers + headroom
#SBATCH --mem=32G                    # 32 GB RAM recommended for RUGD
#SBATCH --time=24:00:00              # Max wall time (HH:MM:SS)
#SBATCH --output=logs/train_%j.log   # %j = SLURM job ID
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL         # Email on job end or failure
# #SBATCH --mail-user=you@example.com   # ← uncomment and fill in your email

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
set -euo pipefail

# Make log directory in case it doesn't exist
mkdir -p logs

echo "============================================================"
echo "SLURM job ID   : $SLURM_JOB_ID"
echo "Node           : $SLURMD_NODENAME"
echo "GPUs allocated : $SLURM_JOB_GPUS"
echo "Start time     : $(date)"
echo "============================================================"

# Load modules — adjust to match your cluster
module purge
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || module load python/3.11 2>/dev/null || true
module load CUDA/12.1.0 2>/dev/null || module load cuda/12.1 2>/dev/null || true
module list 2>&1

# Activate the virtual environment created by hpc/setup_env.sh
source .venv/bin/activate

# Print GPU info for the log
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
echo ""
echo "Starting training..."

python train.py \
  --config configs/config.yaml \
  --run_name "slurm_${SLURM_JOB_ID}"

# ---------------------------------------------------------------------------
# Optional: run evaluation immediately after training
# ---------------------------------------------------------------------------
# echo "Running evaluation..."
# python evaluate.py \
#   --checkpoint "runs/slurm_${SLURM_JOB_ID}/checkpoints/best.pt" \
#   --split test \
#   --save_vis \
#   --vis_dir "runs/slurm_${SLURM_JOB_ID}/eval_vis"

echo "============================================================"
echo "Job complete: $(date)"
echo "============================================================"
