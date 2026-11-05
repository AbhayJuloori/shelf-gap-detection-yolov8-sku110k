#!/usr/bin/env bash
# Push the GPU kernel. --smoke pushes a separate tiny end-to-end check kernel.
set -euo pipefail
cd "$(dirname "$0")"
STAGE=$(mktemp -d)
cp train_eval.py "$STAGE/"
if [[ "${1:-}" == "--smoke" ]]; then
  sed -i '' 's/^SMOKE = False/SMOKE = True/' "$STAGE/train_eval.py"
  sed 's/sku110k-yolov8-train-eval/sku110k-yolov8-smoke/g' kernel-metadata.json > "$STAGE/kernel-metadata.json"
else
  cp kernel-metadata.json "$STAGE/"
fi
kaggle kernels push -p "$STAGE"
