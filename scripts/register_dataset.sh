#!/usr/bin/env bash
# Register an arbitrary JSON dataset (messages/sharegpt schema) with LLaMA-Factory.
#   bash scripts/register_dataset.sh --name newdata_full   --file $NEWDATA_FULL_JSON
#   bash scripts/register_dataset.sh --name newdata_weasel --file $NEWDATA_WEASEL_JSON
# Copies the file into LLaMA-Factory/data and adds a sharegpt entry (role/content
# tags) to data/dataset_info.json, keeping existing entries. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${LLAMAFACTORY_DIR:-}" ]; then source scripts/setup_env.sh; fi

NAME=""; FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;  --name=*) NAME="${1#*=}"; shift ;;
    --file) FILE="$2"; shift 2 ;;  --file=*) FILE="${1#*=}"; shift ;;
    *) echo "[warn] unknown arg: $1" >&2; shift ;;
  esac
done
[ -n "$NAME" ] && [ -n "$FILE" ] || { echo "usage: register_dataset.sh --name <ds> --file <path.json>" >&2; exit 2; }
[ -d "$LLAMAFACTORY_DIR" ] || { echo "[error] LLaMA-Factory missing: $LLAMAFACTORY_DIR (bash scripts/install.sh train)" >&2; exit 1; }
[ -f "$FILE" ] || { echo "[error] dataset file missing: $FILE" >&2; exit 1; }

DEST="$LLAMAFACTORY_DIR/data"
BASENAME="${NAME}.json"
mkdir -p "$DEST"
cp -f "$FILE" "$DEST/$BASENAME"
echo "[register] $FILE -> $DEST/$BASENAME"

python - "$DEST/dataset_info.json" "$NAME" "$BASENAME" <<'PY'
import json, os, sys
info_path, name, fname = sys.argv[1], sys.argv[2], sys.argv[3]
base = json.load(open(info_path)) if os.path.exists(info_path) else {}
base[name] = {
    "file_name": fname,
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {"role_tag": "role", "content_tag": "content",
             "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
}
json.dump(base, open(info_path, "w"), indent=2, ensure_ascii=False)
print(f"[register] dataset '{name}' registered in {info_path}")
PY
