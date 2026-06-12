#!/usr/bin/env bash
# Merge each LoRA adapter into a standalone fp16/bf16 model (for vLLM serving)
# with scripts/merge_lora.py (transformers+peft — no LLaMA-Factory).
#   $OUTPUT_ROOT/<model>/<variant>  (adapter)  ->  $MERGED_ROOT/<model>/<variant>  (merged)
#
#   bash scripts/run_merge.sh                 # all models found under OUTPUT_ROOT (VARIANT=weasel)
#   MODELS="qwen25" bash scripts/run_merge.sh
#   VARIANT=full MODELS="qwen25" bash scripts/run_merge.sh   # merge the full-data adapter
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${OUTPUT_ROOT:-}" ] || ! type weasel_activate >/dev/null 2>&1; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
weasel_activate train

MODELS=${MODELS:-"qwen25 gemma3 qwen3"}
VARIANT=${VARIANT:-weasel}     # which data variant's adapter to merge

base_model() {
  case "$1" in
    qwen25)    echo "$MODEL_QWEN25_7B" ;;
    gemma3)    echo "$MODEL_GEMMA3_4B" ;;
    qwen3)     echo "$MODEL_QWEN3_8B" ;;
    qwen35_9b) echo "$MODEL_QWEN35_9B" ;;
    *) return 1 ;;
  esac
}

mkdir -p logs
for model in $MODELS; do
  mpath="$(base_model "$model")" || { echo "[error] unknown model key: $model (qwen25|gemma3|qwen3|qwen35_9b)" >&2; exit 2; }
  adapter="$OUTPUT_ROOT/$model/$VARIANT"
  export_dir="$MERGED_ROOT/$model/$VARIANT"
  if [ ! -d "$adapter" ]; then echo "[skip] no adapter: $adapter" >&2; continue; fi
  echo "=== merge $model -> $export_dir ==="
  python scripts/merge_lora.py --base "$mpath" --adapter "$adapter" \
    --output "$export_dir" 2>&1 | tee "logs/merge_${model}.log"
done
echo "[run_merge] done. merged models under $MERGED_ROOT/<model>/$VARIANT"
echo "[run_merge] next: VARIANT=$VARIANT bash scripts/serve_vllm.sh qwen25"
