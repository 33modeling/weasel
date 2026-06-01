#!/usr/bin/env bash
# Register the WEASEL training file with LLaMA-Factory.
#   1) copy $WEASEL_TRAIN_JSON  -> $LLAMAFACTORY_DIR/data/weasel_agenttrek_train_10k.json
#   2) merge configs/llamafactory/dataset_info.json into LLaMA-Factory's
#      data/dataset_info.json (adds the "weasel_agenttrek" entry, keeps the rest)
#
# Run once after install (train) + download_data. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${LLAMAFACTORY_DIR:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi

[ -d "$LLAMAFACTORY_DIR" ] || { echo "[error] LLaMA-Factory not installed: $LLAMAFACTORY_DIR (run: bash scripts/install.sh train)" >&2; exit 1; }
[ -f "$WEASEL_TRAIN_JSON" ] || { echo "[error] training file missing: $WEASEL_TRAIN_JSON (run: bash scripts/download_data.sh)" >&2; exit 1; }

DEST_DATA_DIR="$LLAMAFACTORY_DIR/data"
mkdir -p "$DEST_DATA_DIR"
cp -f "$WEASEL_TRAIN_JSON" "$DEST_DATA_DIR/weasel_agenttrek_train_10k.json"
echo "[prepare] copied training file -> $DEST_DATA_DIR/weasel_agenttrek_train_10k.json"

python - "$DEST_DATA_DIR/dataset_info.json" "configs/llamafactory/dataset_info.json" <<'PY'
import json, sys, os
target, addition = sys.argv[1], sys.argv[2]
base = json.load(open(target)) if os.path.exists(target) else {}
add = json.load(open(addition))
base.update(add)
json.dump(base, open(target, "w"), indent=2, ensure_ascii=False)
print(f"[prepare] merged {list(add)} into {target} ({len(base)} datasets total)")
PY

echo "[prepare_dataset] done. dataset name to use in configs: weasel_agenttrek"
