#!/bin/bash
#SBATCH --job-name=diffusion-preprocess
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=azp

set -eo pipefail

# Simple template:
# - Copy to repo root and edit values below.
# - Submit from repo root: sbatch preprocess.sbatch
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_DIR="$(cd "${REPO_DIR}" && pwd)"

module load anaconda3/2025.6

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffusion-pytorch

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
echo "Repo dir: ${REPO_DIR}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# ---- Build dataset (combined inputs/targets) ----
CMD=(python -m diffusion.data.make_data
  --dumps_dir data/dumps
  --out_path data/processed/dataset.pt
  --stride_k "$STRIDE_K"
  --species_types "$SPECIES_TYPES"
  --masses "$MASSES"
  --a "$A"
  --num_workers "${SLURM_CPUS_PER_TASK:-1}"
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

echo "If your dumps had mixed (nx,ny) shapes, make_data will also generate:"
echo "  - data/processed/dataset_nx*__ny*.pt"
echo "  - data/processed/dataset_by_xy_manifest.json"

