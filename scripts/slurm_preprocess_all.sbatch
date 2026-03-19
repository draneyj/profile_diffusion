#!/bin/bash
#SBATCH --job-name=diffusion-preprocess
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=azp

set -eo pipefail

# Template usage:
# 1) Copy this file to your repo top-level (optional):
#      cp scripts/slurm_preprocess_all.sbatch preprocess.sbatch
# 2) Edit variables below as needed.
# 3) Submit from repo root:
#      sbatch preprocess.sbatch
#
# Repo root resolution:
# - If REPO_DIR is set externally, use it.
# - Else default to SLURM_SUBMIT_DIR (or PWD), which works when you submit from repo root.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_DIR="$(cd "${REPO_DIR}" && pwd)"

module load anaconda3/2025.6

# Robust conda activation in non-interactive shells.
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

if [[ ! -d "${REPO_DIR}/diffusion" ]]; then
  echo "Could not locate python package directory: ${REPO_DIR}/diffusion" >&2
  echo "Hint: submit from repo root or set REPO_DIR=/path/to/repo in the sbatch script." >&2
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

