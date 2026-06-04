#!/usr/bin/env bash
# Merge each LoRA adapter into a standalone fp16 model (for vLLM serving / eval).
#   $OUTPUT_ROOT/<model>/<variant>  (adapter)  ->  $MERGED_ROOT/<model>/<variant>  (merged)
#
#   bash scripts/run_merge.sh                 # all models found under OUTPUT_ROOT (VARIANT=weasel)
#   MODELS="qwen25" bash scripts/run_merge.sh
#   VARIANT=full MODELS="qwen25" bash scripts/run_merge.sh   # merge the full-data adapter
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${OUTPUT_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
weasel_activate train

MODELS=${MODELS:-"qwen25 gemma3 qwen3"}
VARIANT=${VARIANT:-weasel}     # which data variant's adapter to merge

base_and_template() {
  case "$1" in
    qwen25)    echo "$MODEL_QWEN25_7B qwen" ;;
    gemma3)    echo "$MODEL_GEMMA3_4B gemma3" ;;
    qwen3)     echo "$MODEL_QWEN3_8B qwen3" ;;
    qwen35_9b) echo "$MODEL_QWEN35_9B $QWEN35_9B_TEMPLATE" ;;
    *) return 1 ;;
  esac
}

mkdir -p logs
for model in $MODELS; do
  read -r mpath template <<< "$(base_and_template "$model")"
  adapter="$OUTPUT_ROOT/$model/$VARIANT"
  export_dir="$MERGED_ROOT/$model/$VARIANT"
  if [ ! -d "$adapter" ]; then echo "[skip] no adapter: $adapter" >&2; continue; fi
  mkdir -p "$export_dir"
  cfg="$export_dir/merge_config.yaml"
  cat > "$cfg" <<YAML
model_name_or_path: $mpath
adapter_name_or_path: $adapter
template: $template
finetuning_type: lora
trust_remote_code: true
export_dir: $export_dir
export_size: 5
export_legacy_format: false
YAML
  echo "=== merge $model -> $export_dir ==="
  llamafactory-cli export "$cfg" 2>&1 | tee "logs/merge_${model}.log"
done
echo "[run_merge] done. merged models under $MERGED_ROOT/<model>/$VARIANT"
echo "[run_merge] next: VARIANT=$VARIANT bash scripts/serve_vllm.sh qwen25"
