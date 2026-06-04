#!/usr/bin/env bash
# Serve a merged WEASEL model as an OpenAI-compatible endpoint for AgentLab.
#
#   bash scripts/serve_vllm.sh qwen25                  # serves on $VLLM_PORT (default 8000)
#   TP=2 VLLM_PORT=8001 bash scripts/serve_vllm.sh qwen3 --gpus 0,1
#   VARIANT=full bash scripts/serve_vllm.sh qwen25     # serve the full-data merged model
#
# Leave this running in one terminal/tmux pane; run scripts/run_eval.sh in another.
# AgentLab reaches it via OPENAI_API_BASE=http://$VLLM_HOST:$VLLM_PORT/v1
# and the served model name $VLLM_SERVED_NAME (default 'weasel').
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${MERGED_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
weasel_activate eval

MODEL_KEY="${1:-qwen25}"; shift || true
GPUS="${GPUS:-0}"
TP="${TP:-1}"                  # tensor-parallel size; a 7-8B fits on ONE 80GB A100 (TP=1)
MAXLEN="${MAXLEN:-32768}"      # context window for long AXTree web prompts
VARIANT="${VARIANT:-weasel}"  # which data variant's merged model to serve
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;; --gpus=*) GPUS="${1#*=}"; shift ;;
    --tp) TP="$2"; shift 2 ;;     --tp=*) TP="${1#*=}"; shift ;;
    --variant) VARIANT="$2"; shift 2 ;; --variant=*) VARIANT="${1#*=}"; shift ;;
    *) echo "[warn] unknown arg: $1" >&2; shift ;;
  esac
done

MODEL_PATH="$MERGED_ROOT/$MODEL_KEY/$VARIANT"
[ -d "$MODEL_PATH" ] || { echo "[error] merged model not found: $MODEL_PATH (run: VARIANT=$VARIANT bash scripts/run_merge.sh)" >&2; exit 1; }

echo "[serve] $MODEL_KEY  variant=$VARIANT  path=$MODEL_PATH  gpus=$GPUS  TP=$TP  port=$VLLM_PORT  name=$VLLM_SERVED_NAME"
CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "$MODEL_PATH" \
  --served-model-name "$VLLM_SERVED_NAME" \
  --host "$VLLM_HOST" --port "$VLLM_PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAXLEN" \
  --trust-remote-code
