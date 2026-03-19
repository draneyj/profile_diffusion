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

set -euo pipefail

# Resolve repo root from submit directory (preferred under Slurm).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -d "${SUBMIT_DIR}/diffusion" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
elif [[ -d "${SUBMIT_DIR}/../diffusion" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# ---- User config (edit as needed) ----
ENV_NAME="diffusion-cuda124"   # change to your conda env name

DATASET_PATH="data/processed/dataset.pt"

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
  # Conda activation can reference unset internal vars on some clusters;
  # temporarily disable nounset around activation.
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  set -u
else
  echo "conda not found in PATH. Activate your env manually before running this job."
  exit 1
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
if [[ ! -d "${REPO_DIR}/diffusion" ]]; then
  echo "Could not locate python package directory: ${REPO_DIR}/diffusion" >&2
  exit 1
fi

CMD=(python -m diffusion.train
  --option i
  --dataset_path "$DATASET_PATH"
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

