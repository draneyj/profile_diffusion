#!/usr/bin/env bash
set -euo pipefail

# Simple Princeton-style GPU PyTorch install script.
#
# This follows:
# https://researchcomputing.princeton.edu/support/knowledge-base/pytorch
#
# It does:
# 1) module load anaconda3/2025.6
# 2) conda create --name torch-env python=3.12
# 3) pip install torch torchvision from cu130 index
# 4) installs common extras used in this repo
# 5) verifies GPU visibility

ENV_NAME="diffusion-pytorch"
PYTHON_VERSION="3.12"
ANACONDA_MODULE="anaconda3/2025.6"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"

echo "Loading module: ${ANACONDA_MODULE}"
module load "${ANACONDA_MODULE}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found after module load. Exiting."
  exit 1
fi

# source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create --name "${ENV_NAME}" "python=${PYTHON_VERSION}" -y

echo "Activating ${ENV_NAME}"
conda activate "${ENV_NAME}"

echo "Installing PyTorch (GPU) + torchvision"
pip3 install torch torchvision --index-url "${PYTORCH_INDEX_URL}"

echo "Installing extra packages used by this project"
pip3 install numpy tqdm matplotlib ipykernel

echo "Testing PyTorch GPU visibility"
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

echo
echo "Done."
echo "Use this env with:"
echo "  module load ${ANACONDA_MODULE}"
echo "  conda activate ${ENV_NAME}"

