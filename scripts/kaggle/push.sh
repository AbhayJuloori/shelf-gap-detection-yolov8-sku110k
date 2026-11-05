#!/usr/bin/env bash
# Push the GPU kernel. --smoke pushes a separate tiny end-to-end check kernel.
set -euo pipefail
cd "$(dirname "$0")"
STAGE=$(mktemp -d)
cp train_eval.py "$STAGE/"
if [[ "${1:-}" == "--smoke" ]]; then
  sed -i '' 's/^SMOKE = False/SMOKE = True/' "$STAGE/train_eval.py"
  sed 's/sku110k-yolov8-train-eval/sku110k-yolov8-smoke/g' kernel-metadata.json > "$STAGE/kernel-metadata.json"
elif [[ "${1:-}" == "--export" ]]; then
  sed -i '' 's/^EXPORT_ONLY = False/EXPORT_ONLY = True/' "$STAGE/train_eval.py"
  python3 -c "import json,sys; m=json.load(open('kernel-metadata.json')); m['id']='julooriabhay/sku110k-yolov8-export'; m['title']='sku110k-yolov8-export'; m['dataset_sources'].append('julooriabhay/sku110k-yolov8s-weights'); json.dump(m,open(sys.argv[1],'w'))" "$STAGE/kernel-metadata.json"
else
  cp kernel-metadata.json "$STAGE/"
fi
kaggle kernels push -p "$STAGE"
