#!/bin/bash
#SBATCH --job-name=diffusion-train-iv
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=azp
#SBATCH --mail-type=all

set -eo pipefail

module load anaconda3/2025.6
conda activate diffusion-pytorch

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_DIR="$(cd "${REPO_DIR}" && pwd)"

# ---- User config ----
DATASET_PATH_BASE="data/processed/dataset.pt"
DATASET_PATH_MANIFEST="${DATASET_PATH_BASE%.pt}_by_xy_manifest.json"

EPOCHS=100
BATCH_SIZE=100
LR=1e-3
HIDDEN_CHANNELS=64
VAL_FRACTION=0.1
SAVE_EVERY_N_EPOCHS=5
# Optional: CHECKPOINT_STEM=my_run sbatch ...
CHECKPOINT_STEM="${CHECKPOINT_STEM:-}"

echo "Running on host: $(hostname)"
echo "Repo dir: ${REPO_DIR}"
cd "$REPO_DIR"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

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

export PYTHONUNBUFFERED=1

CMD=(python -u -m diffusion.train
  --option iv
  --dataset_paths "$DATASET_PATHS"
  --shape_balance_mode proportional
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --learning_rate "$LR"
  --hidden_channels "$HIDDEN_CHANNELS"
  --val_fraction "$VAL_FRACTION"
  --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
  --learning_curve_path data/processed/learning_curve_optioniv.csv
  --device cuda
)
if [[ -n "$CHECKPOINT_STEM" ]]; then
  CMD+=(--checkpoint_stem "$CHECKPOINT_STEM")
fi

echo "Command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"
