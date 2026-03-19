#!/usr/bin/env bash
set -euo pipefail

# Create and populate the conda environment needed for this project.
#
# Run this on your machine (outside the sandbox) where conda has access to:
# - writable env directories
# - network access to fetch packages

ENV_NAME="diffusion-torch"
PYTHON_VERSION="3.12"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Install Miniconda/Anaconda and re-run." >&2
  exit 1
fi

echo "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip numpy tqdm

echo "Installing torch via pip (lets pip pick the correct macOS/MPS wheel)"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install torch

echo "Installing plotting deps (matplotlib)"
conda run -n "${ENV_NAME}" python -m pip install matplotlib

echo "Done."
echo "Next: run with: conda run -n ${ENV_NAME} python -m diffusion.train --help"

