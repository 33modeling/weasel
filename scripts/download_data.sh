#!/usr/bin/env bash
# Get the WEASEL training data into $WEASEL_DATA (group-volume).
#
# Usage:
#   bash scripts/download_data.sh                 # pre-built WEASEL-selected 10K (DEFAULT, fast)
#   bash scripts/download_data.sh prebuilt        # same as above
#   bash scripts/download_data.sh agenttrek       # raw AgentTrek pool + convert -> train.json
#                                                 #   (needed only to re-run scripts/run_select.sh)
#
# 'prebuilt' downloads the authors' final 10K SFT file (already pruned +
# selected + reasoning-synthesised) from the Google-Drive id in the README,
# so you can skip the whole selection pipeline and train directly.
# Skips downloads whose target already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${WEASEL_DATA:-}" ] || ! type weasel_activate >/dev/null 2>&1; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
weasel_activate select 2>/dev/null || true
mkdir -p "$WEASEL_DATA"

WHAT="${1:-prebuilt}"
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0

dl_prebuilt() {
  if [ -f "$WEASEL_TRAIN_JSON" ] && [ -s "$WEASEL_TRAIN_JSON" ]; then
    echo "[skip] pre-built file present: $WEASEL_TRAIN_JSON"; return 0
  fi
  echo "[fetch] pre-built WEASEL 10K (gdrive id $WEASEL_TRAIN_GDRIVE_ID) -> $WEASEL_TRAIN_JSON"
  python -m gdown "$WEASEL_TRAIN_GDRIVE_ID" -O "$WEASEL_TRAIN_JSON" \
    || gdown "$WEASEL_TRAIN_GDRIVE_ID" -O "$WEASEL_TRAIN_JSON"
  python -c "import json;d=json.load(open('$WEASEL_TRAIN_JSON'));print('[done] records:',len(d))"
}

dl_agenttrek() {
  if [ -f "$TRAIN_INPUT_JSON" ] && [ -s "$TRAIN_INPUT_JSON" ]; then
    echo "[skip] converted AgentTrek present: $TRAIN_INPUT_JSON"; return 0
  fi
  echo "[fetch] xlangai/AgentTrek (dataset) -> $AGENTTREK_DIR"
  mkdir -p "$AGENTTREK_DIR"
  huggingface-cli download xlangai/AgentTrek --repo-type dataset --local-dir "$AGENTTREK_DIR" \
    || hf download xlangai/AgentTrek --repo-type dataset --local-dir "$AGENTTREK_DIR"
  echo "[convert] AgentTrek -> $TRAIN_INPUT_JSON  (list of {\"messages\":[...]})"
  python - "$AGENTTREK_DIR" "$TRAIN_INPUT_JSON" <<'PY'
import sys, json
from datasets import load_dataset
src, out = sys.argv[1], sys.argv[2]
ds = load_dataset(src, split="train")
key = "messages" if "messages" in ds.column_names else None
if key is None:
    raise SystemExit(f"[error] no 'messages' column in AgentTrek; columns={ds.column_names}. "
                     f"Adjust this converter so each record becomes {{'messages':[...]}}.")
recs = [{"messages": ex[key]} for ex in ds]
json.dump(recs, open(out, "w"), ensure_ascii=False)
print(f"[done] wrote {len(recs)} records -> {out}")
PY
  echo "[note] AgentTrek conversion done. NOTE: weasel.prune_axtree is MISSING from the repo"
  echo "       (README says 'added soon'); run_select.sh therefore skips pruning and scores"
  echo "       the un-pruned AXTrees. To match the paper exactly you need that script."
}

case "$WHAT" in
  prebuilt) dl_prebuilt ;;
  agenttrek) dl_agenttrek ;;
  *) echo "usage: bash scripts/download_data.sh {prebuilt|agenttrek}" >&2; exit 2 ;;
esac
echo "[download_data] done: $WHAT"
