#!/usr/bin/env bash
set -euo pipefail

# Installs Python dependencies for this project.
#
# Defaults:
# - CPU-only PyTorch (works on macOS)
# - Creates no virtualenv automatically; run inside your venv if you want isolation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Installing dependencies in: $(pwd)"
echo "Project root: ${ROOT_DIR}"

python3 -m pip install --upgrade pip

# Core numerical stack
python3 -m pip install --upgrade numpy tqdm matplotlib

# PyTorch (CPU wheels). If you already have torch, this will be quick.
# NOTE: If you want a specific torch version, pin it here.
if python3 -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
  echo "torch already installed; skipping."
else
  python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
fi

echo "Done."

