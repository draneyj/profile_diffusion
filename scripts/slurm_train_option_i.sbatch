#!/bin/bash
#SBATCH --job-name=diffusion-train-i
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Time limit
#SBATCH --time=12:00:00
#
# Partition
#SBATCH --partition=gpu

set -eo pipefail

# Simple template:
# - Copy to repo root and edit values below.
# - Submit from repo root: sbatch train_i.sbatch
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_DIR="$(cd "${REPO_DIR}" && pwd)"

# ---- User config (edit as needed) ----
ENV_NAME="diffusion-cuda124"   # change to your conda env name

DATASET_PATH_BASE="data/processed/dataset.pt"
DATASET_PATH_MANIFEST="${DATASET_PATH_BASE%.pt}_by_xy_manifest.json"

# Training hyperparameters (adjust as needed)
EPOCHS=5
BATCH_SIZE=2
LR=1e-3
HIDDEN_CHANNELS=64
NUM_REFINE_STEPS=1
NOISE_STD=0.1

VAL_FRACTION=0.1
SAVE_EVERY_N_EPOCHS=5

echo "Running on host: $(hostname)"
echo "Repo dir: ${REPO_DIR}"
cd "$REPO_DIR"

if command -v conda >/dev/null 2>&1; then
  echo "Activating conda env: $ENV_NAME"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
else
  echo "conda not found in PATH. Activate your env manually before running this job."
  exit 1
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# If make_data split by (nx,ny), use the manifest-generated dataset files.
if [[ -f "$DATASET_PATH_MANIFEST" ]]; then
  DATASET_PATHS="$(
    python - <<PY
import json
from pathlib import Path
obj = json.loads(Path("$DATASET_PATH_MANIFEST").read_text(encoding="utf-8"))
print(",".join(d["dataset_path"] for d in obj["generated_datasets"]))
PY
  )"
else
  DATASET_PATHS="$DATASET_PATH_BASE"
fi

echo "Training on dataset paths:"
echo "$DATASET_PATHS"

CMD=(python -m diffusion.train
  --option i
  --dataset_paths "$DATASET_PATHS"
  --shape_balance_mode proportional
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --learning_rate "$LR"
  --hidden_channels "$HIDDEN_CHANNELS"
  --num_refine_steps "$NUM_REFINE_STEPS"
  --noise_std "$NOISE_STD"
  --val_fraction "$VAL_FRACTION"
  --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
  --device cuda
)

echo "Command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"

