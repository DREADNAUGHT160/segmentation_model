#!/usr/bin/env bash
# =============================================================================
# HPC Environment Setup — run ONCE on the login node before training.
#
# This script:
#   1. Installs uv (if not already present)
#   2. Creates a virtual environment and installs all dependencies
#   3. Installs the correct CUDA-compatible PyTorch build
#
# Usage:
#   bash hpc/setup_env.sh
#
# After this runs, activate the venv before each session:
#   source .venv/bin/activate
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Detect project root (the directory containing this script's parent)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "Project root: $PROJECT_DIR"

# ---------------------------------------------------------------------------
# 2. Load required HPC modules
#    Adjust module names to match YOUR cluster's module system.
#    Common variants: "CUDA/12.1.0", "cuda/12.1", "cudatoolkit/12.1"
# ---------------------------------------------------------------------------
echo "Loading HPC modules..."
module purge

# Load Python — check available versions with: module spider python
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load python/3.11 2>/dev/null \
  || echo "WARNING: Could not load a Python module. Using system Python."

# Load CUDA — check available versions with: module spider CUDA
module load CUDA/12.1.0 2>/dev/null \
  || module load cuda/12.1 2>/dev/null \
  || module load cudatoolkit/12.1 2>/dev/null \
  || echo "WARNING: Could not load CUDA module. GPU training may not work."

echo "Active modules:"
module list 2>&1 || true

# ---------------------------------------------------------------------------
# 3. Install uv (fast Python package manager)
#    Installed to ~/.local/bin — no sudo required.
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
else
  echo "uv already installed: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 4. Create virtual environment and install base dependencies
# ---------------------------------------------------------------------------
echo "Creating virtual environment with uv..."
uv venv --python 3.11

# Activate for remainder of this script
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 5. Detect CUDA version and install matching PyTorch
#    uv uses the extra-index-url to pull the right CUDA build.
#
#    To check your cluster's CUDA version:
#      nvcc --version   OR   nvidia-smi
# ---------------------------------------------------------------------------
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | head -1 || echo "")
echo "Detected CUDA: ${CUDA_VER:-unknown}"

# Map CUDA version → PyTorch index suffix
if [[ "$CUDA_VER" == 12.4* || "$CUDA_VER" == 12.5* || "$CUDA_VER" == 12.6* ]]; then
  TORCH_CUDA="cu124"
elif [[ "$CUDA_VER" == 12.1* || "$CUDA_VER" == 12.2* || "$CUDA_VER" == 12.3* ]]; then
  TORCH_CUDA="cu121"
elif [[ "$CUDA_VER" == 11.8* ]]; then
  TORCH_CUDA="cu118"
else
  echo "WARNING: Unknown CUDA version '${CUDA_VER}'. Defaulting to cu121."
  echo "  If training fails, edit TORCH_CUDA in this script and re-run."
  TORCH_CUDA="cu121"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
echo "Installing PyTorch with index: $TORCH_INDEX"

uv pip install torch torchvision --extra-index-url "$TORCH_INDEX"

# ---------------------------------------------------------------------------
# 6. Install remaining project dependencies
# ---------------------------------------------------------------------------
echo "Installing project dependencies from pyproject.toml..."
uv pip install -e .

# ---------------------------------------------------------------------------
# 7. Verify GPU access
# ---------------------------------------------------------------------------
echo ""
echo "Verifying GPU access..."
python -c "
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU             : {torch.cuda.get_device_name(0)}')
    print(f'CUDA version    : {torch.version.cuda}')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "To activate the environment in future sessions:"
echo "  source .venv/bin/activate"
echo ""
echo "Then train with:"
echo "  sbatch hpc/train.slurm"
echo "============================================================"
