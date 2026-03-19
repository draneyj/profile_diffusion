#!/bin/bash
#SBATCH --job-name=diffusion-preprocess
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# CPU/memory
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#
# Time limit
#SBATCH --time=24:00:00
#
# Partition/GPU: not needed for preprocessing
#SBATCH --partition=cpu

set -euo pipefail

module load anaconda3/2025.6
module load cudatoolkit/13.1
conda activate diffusion-cuda124

# Coarse-graining / dataset options
SPECIES_TYPES="1,3"
MASSES="12.011,39.948"
A="3.5657157"
STRIDE_K=1

# Padding along z (recommended since nz can differ across dumps)
PAD_TO_COMMON_NZ=1
MASK_LOSS_PADDDED_CELLS=1

# Optional: limit number of dump files for smoke testing (leave empty for all)
MAX_DUMP_FILES=""

# ---- Load/activate Python environment ----
echo "Running on host: $(hostname)"

if command -v conda >/dev/null 2>&1; then
  echo "Activating conda env: $ENV_NAME"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
else
  echo "conda not found in PATH. Activate your env manually before running this job."
  exit 1
fi

# ---- Build dataset (combined inputs/targets) ----
CMD=(python -m diffusion.data.make_data
  --dumps_dir data/dumps
  --out_path data/processed/dataset.pt
  --stride_k "$STRIDE_K"
  --species_types "$SPECIES_TYPES"
  --masses "$MASSES"
  --a "$A"
)

if [[ "$PAD_TO_COMMON_NZ" -eq 1 ]]; then
  CMD+=(--pad_to_common_nz)
fi
if [[ "$MASK_LOSS_PADDDED_CELLS" -eq 1 ]]; then
  CMD+=(--mask_loss_padded_cells)
fi
if [[ -n "$MAX_DUMP_FILES" ]]; then
  CMD+=(--max_dump_files "$MAX_DUMP_FILES")
fi

echo "Command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Preprocessing complete: data/processed/dataset.pt"

