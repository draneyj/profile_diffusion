#!/usr/bin/env bash
set -euo pipefail

# Run a quick Option III inference rollout from a random sample.

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

# Override: CKPT=/path/to/checkpoint.pt ./05_infer_random_option_iii_smoke.sh
CKPT="${CKPT:-data/processed/checkpoint_optioniii_epoch100.pt}"

python -m diffusion.infer_random \
  --option iii \
  --dataset_path "$DATASET_PATH" \
  --checkpoint "$CKPT" \
  --num_steps 20 \
  --sample_seed 0 \
  --device cpu

echo "Rollout written under data/processed/*.npz"

