#!/usr/bin/env bash
set -euo pipefail

# Train Option I on smoke datasets.
#
# If `data/processed/smoke_dataset_by_xy_manifest.json` exists, we load dataset paths
# from it and interleave them during training.

BASE_OUT="data/processed/smoke_dataset.pt"
MANIFEST="data/processed/smoke_dataset_by_xy_manifest.json"

DATASET_PATHS=""
if [[ -f "$MANIFEST" ]]; then
  # Build a comma-separated list: dataset_paths="path1,path2,..."
  DATASET_PATHS="$(
    python - <<'PY'
import json
from pathlib import Path
manifest = Path("data/processed/smoke_dataset_by_xy_manifest.json")
obj = json.loads(manifest.read_text(encoding="utf-8"))
paths = [d["dataset_path"] for d in obj["generated_datasets"]]
print(",".join(paths))
PY
  )"
else
  DATASET_PATHS="$BASE_OUT"
fi

python -m diffusion.train \
  --option i \
  --dataset_paths "$DATASET_PATHS" \
  --shape_balance_mode proportional \
  --epochs 3 \
  --batch_size 2 \
  --learning_rate 1e-3 \
  --hidden_channels 32 \
  --num_refine_steps 1 \
  --noise_std 0.1 \
  --val_fraction 0.1 \
  --save_every_n_epochs 1 \
  --device cpu

echo "Training done. Checkpoints should be in data/processed/"

