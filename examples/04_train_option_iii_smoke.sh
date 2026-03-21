#!/usr/bin/env bash
set -euo pipefail

# Train Option III on smoke datasets.
#
# If data/processed/smoke_dataset_by_xy_manifest.json exists, interleave all
# per-(nx,ny) datasets with proportional sampling.

BASE_OUT="data/processed/smoke_dataset.pt"
MANIFEST="data/processed/smoke_dataset_by_xy_manifest.json"

DATASET_PATHS=""
if [[ -f "$MANIFEST" ]]; then
  DATASET_PATHS="$(
    python - <<'PY'
import json
from pathlib import Path
manifest = Path("data/processed/smoke_dataset_by_xy_manifest.json")
obj = json.loads(manifest.read_text(encoding="utf-8"))
print(",".join(d["dataset_path"] for d in obj["generated_datasets"]))
PY
  )"
else
  DATASET_PATHS="$BASE_OUT"
fi

python -m diffusion.train \
  --option iii \
  --dataset_paths "$DATASET_PATHS" \
  --shape_balance_mode proportional \
  --epochs 3 \
  --batch_size 2 \
  --learning_rate 1e-3 \
  --hidden_channels 32 \
  --val_fraction 0.1 \
  --save_every_n_epochs 1 \
  --device cpu

echo "Training done. Checkpoint: data/processed/checkpoint_optioniii_epoch3.pt"

