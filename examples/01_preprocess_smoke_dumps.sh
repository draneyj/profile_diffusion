#!/usr/bin/env bash
set -euo pipefail

# Preprocess a tiny smoke dataset from all (or a limited number of) LAMMPS dumps.
# If dumps produce mixed (nx, ny), this will generate multiple dataset files:
#   data/processed/smoke_dataset_nx{nx}_ny{ny}.pt
# and a manifest:
#   data/processed/smoke_dataset_by_xy_manifest.json

python -m diffusion.data.make_data \
  --dumps_dir data/dumps \
  --out_path data/processed/smoke_dataset.pt \
  --stride_k 1 \
  --species_types 1,3 \
  --masses 12.011,39.948 \
  --a 3.5657157 \
  --max_dump_files 4 \
  --max_pairs 200 \
  --pad_to_common_nz \
  --pad_nz_mode max \
  --mask_loss_padded_cells

echo "Done. Look for data/processed/smoke_dataset*.pt"

