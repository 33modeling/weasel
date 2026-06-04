#!/usr/bin/env bash
# Download the base checkpoints WEASEL fine-tunes, into $MODELS_DIR (group-volume).
#
# Usage:
#   bash scripts/download_models.sh all          # all four
#   bash scripts/download_models.sh qwen25_7b    # one of: qwen25_7b | gemma3_4b | qwen3_8b | qwen35_9b
#
# google/gemma-3-4b-it is GATED: run `huggingface-cli login` and accept the
# license at https://huggingface.co/google/gemma-3-4b-it first.
# qwen35_9b uses HFID_QWEN35_9B (best-guess 'Qwen/Qwen3.5-9B') — verify/override it,
# or point MODEL_QWEN35_9B at an existing local checkpoint dir to skip the download.
# Skips any model whose target dir already has a config.json.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${MODELS_DIR:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
weasel_activate select 2>/dev/null || weasel_activate train 2>/dev/null || true

WHAT="${1:-all}"
# download needs the network; setup_env.sh defaults to offline.
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0

_fetch() {  # $1 = HF repo id   $2 = local dir
  local repo="$1" dest="$2"
  if [ -f "$dest/config.json" ]; then
    echo "[skip] $repo already at $dest"; return 0
  fi
  echo "[fetch] $repo -> $dest"
  mkdir -p "$dest"
  huggingface-cli download "$repo" --local-dir "$dest" \
    --exclude "*.pth" "original/*" || hf download "$repo" --local-dir "$dest"
  echo "[done] $repo"
}

case "$WHAT" in
  qwen25_7b) _fetch "$HFID_QWEN25_7B" "$MODEL_QWEN25_7B" ;;
  gemma3_4b) _fetch "$HFID_GEMMA3_4B" "$MODEL_GEMMA3_4B" ;;
  qwen3_8b)  _fetch "$HFID_QWEN3_8B"  "$MODEL_QWEN3_8B" ;;
  qwen35_9b) _fetch "$HFID_QWEN35_9B" "$MODEL_QWEN35_9B" ;;
  all)
    _fetch "$HFID_QWEN25_7B" "$MODEL_QWEN25_7B"
    _fetch "$HFID_QWEN3_8B"  "$MODEL_QWEN3_8B"
    _fetch "$HFID_QWEN35_9B" "$MODEL_QWEN35_9B"   # repo id is a best guess — verify HFID_QWEN35_9B
    _fetch "$HFID_GEMMA3_4B" "$MODEL_GEMMA3_4B"   # gated; needs HF login
    ;;
  *) echo "usage: bash scripts/download_models.sh {all|qwen25_7b|gemma3_4b|qwen3_8b|qwen35_9b}" >&2; exit 2 ;;
esac
echo "[download_models] done: $WHAT"
