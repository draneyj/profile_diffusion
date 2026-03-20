#!/usr/bin/env bash
set -euo pipefail

# Run a quick Option I inference rollout starting from a random sample.
#
# This script picks one dataset file:
# - If a manifest exists, it uses the first generated dataset path.
# - Otherwise it uses data/processed/smoke_dataset.pt.

MANIFEST="data/processed/smoke_dataset_by_xy_manifest.json"
BASE_DATASET="data/processed/smoke_dataset.pt"

DATASET_PATH=""
if [[ -f "$MANIFEST" ]]; then
  DATASET_PATH="$(
    python - <<'PY'
import json
from pathlib import Path
manifest = Path("data/processed/smoke_dataset_by_xy_manifest.json")
obj = json.loads(manifest.read_text(encoding="utf-8"))
print(obj["generated_datasets"][0]["dataset_path"])
PY
  )"
else
  DATASET_PATH="$BASE_DATASET"
fi

# Change this if your training run used different epochs.
CKPT="data/processed/checkpoint_optioni_epoch3.pt"

python -m diffusion.infer_random \
  --option i \
  --dataset_path "$DATASET_PATH" \
  --checkpoint "$CKPT" \
  --num_steps 20 \
  --device cpu

echo "Rollout written under data/processed/*.npz"

