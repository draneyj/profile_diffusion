#!/usr/bin/env bash
set -euo pipefail

# Install CUDA-enabled PyTorch on a RedHat/Linux HPC cluster.
#
# Assumptions:
# - You have `conda` available (Miniconda/Anaconda) and can create environments.
# - Your machine has NVIDIA driver compatibility for the selected CUDA wheel.
#
# Usage:
#   bash scripts/install_cuda_pytorch_redhat.sh --env diffusion-cuda124 --python 3.12 --torch_tag cu124
#
# Where `--torch_tag` must be one of: cu118, cu121, cu124

ENV_NAME="diffusion-cuda124"
PYTHON_VERSION="3.12"
TORCH_TAG="cu124"
CUDA_MODULE_DEFAULT="cudatoolkit/13.1"
ANACONDA_MODULE_DEFAULT="anaconda3/2025.6"
CUDA_MODULE="$CUDA_MODULE_DEFAULT"
ANACONDA_MODULE="$ANACONDA_MODULE_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="${2:-}"; shift 2 ;;
    --python) PYTHON_VERSION="${2:-}"; shift 2 ;;
    --torch_tag) TORCH_TAG="${2:-}"; shift 2 ;;
    --cuda_module) CUDA_MODULE="${2:-}"; shift 2 ;;
    --anaconda_module) ANACONDA_MODULE="${2:-}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--env ENV_NAME] [--python PY_VERSION] [--torch_tag cu118|cu121|cu124] [--cuda_module cudatoolkit/<ver>] [--anaconda_module anaconda3/<ver>]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

if command -v module >/dev/null 2>&1; then
  echo "Loading modules:"
  echo "  module load ${ANACONDA_MODULE}"
  echo "  module load ${CUDA_MODULE}"
  module load "${ANACONDA_MODULE}" || true
  module load "${CUDA_MODULE}" || true
fi

if [[ ! -x "$(command -v conda)" ]]; then
  echo "Error: conda not found in PATH after attempting module loads." >&2
  echo "Ensure you can run: module load ${ANACONDA_MODULE}" >&2
  echo "Then activate: conda activate <env> (or re-run this script)." >&2
  exit 1
fi

if [[ -f /etc/redhat-release ]]; then
  echo "Detected RedHat-like system: $(cat /etc/redhat-release)"
fi

case "$TORCH_TAG" in
  cu118|cu121|cu124) ;;
  *)
    echo "Unsupported --torch_tag: $TORCH_TAG (expected cu118, cu121, or cu124)" >&2
    exit 2
    ;;
esac

echo "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip numpy tqdm matplotlib

echo "Upgrading pip"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

# Install CUDA-enabled PyTorch from PyTorch's wheel index for the tag.
# For example:
#   cu124 -> https://download.pytorch.org/whl/cu124/...
WHEEL_INDEX_URL="https://download.pytorch.org/whl/${TORCH_TAG}"
echo "Installing torch from index: ${WHEEL_INDEX_URL}"
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir torch --index-url "${WHEEL_INDEX_URL}"

echo "Done."
echo "Installing extra plotting deps (matplotlib)"
conda run -n "${ENV_NAME}" python -m pip install --upgrade matplotlib || true

echo "Activate:"
echo "  conda activate ${ENV_NAME}"
echo "Quick check:"
echo "  python -c \"import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())\""

