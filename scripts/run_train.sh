#!/usr/bin/env bash
# LoRA-SFT the WEASEL-selected data with LLaMA-Factory on the A100 cluster.
#
# Default mode: sequential DDP — each model trains across ALL --gpus via
# torchrun. A 7-8B LoRA also fits on ONE 80GB A100, so --parallel runs the
# three models concurrently, one GPU each (cycled), to use the cluster fully.
#
#   bash scripts/run_train.sh --gpus 0,1,2,3,4,5,6,7          # sequential DDP
#   bash scripts/run_train.sh --gpus 0,1,2 --parallel         # 1 model per GPU, concurrent
#   MODELS="qwen25 qwen3" bash scripts/run_train.sh --gpus 0,1
#   DS=1 bash scripts/run_train.sh --gpus 0,1,2,3             # add DeepSpeed ZeRO-2
#
# Per-model recipe is paper-faithful (Table 9); global batch is held constant
# regardless of GPU count by adjusting gradient_accumulation_steps.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${OUTPUT_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
weasel_activate train

MODELS=${MODELS:-"qwen25 gemma3 qwen3"}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
PARALLEL=0
DS=${DS:-0}
PER_DEVICE=${PER_DEVICE:-1}
CUTOFF=${CUTOFF:-8192}
while [ $# -gt 0 ]; do
  case "$1" in
    --parallel) PARALLEL=1; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;  --gpus=*) GPUS="${1#*=}"; shift ;;
    --cutoff) CUTOFF="$2"; shift 2 ;; --cutoff=*) CUTOFF="${1#*=}"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "[warn] unknown arg: $1" >&2; shift ;;
  esac
done
IFS=',' read -r -a _gpus_arr <<< "$GPUS"
NPROC=${#_gpus_arr[@]}

# Per-model: <base-model path> <chat template> <lr> <epochs> <global-batch>
model_spec() {
  case "$1" in
    qwen25) echo "$MODEL_QWEN25_7B qwen 2.0e-5 4.0 8" ;;
    gemma3) echo "$MODEL_GEMMA3_4B gemma3 2.0e-5 2.0 16" ;;
    qwen3)  echo "$MODEL_QWEN3_8B qwen3 1.0e-6 2.0 8" ;;
    *) return 1 ;;
  esac
}

DS_LINE=""
if [ "$DS" = "1" ]; then
  DS_CFG="$LLAMAFACTORY_DIR/examples/deepspeed/ds_z2_config.json"
  [ -f "$DS_CFG" ] && DS_LINE="deepspeed: $DS_CFG" || echo "[warn] DeepSpeed cfg not found: $DS_CFG"
fi

mkdir -p logs
echo "[run_train] MODELS=$MODELS  GPUS=$GPUS  NPROC=$NPROC  PARALLEL=$PARALLEL  DS=$DS  cutoff=$CUTOFF"

render_cfg() {  # $1=model  $2=ngpu_for_this_job   -> prints rendered yaml path
  local model="$1" ngpu="$2"
  read -r mpath template lr epochs gbatch <<< "$(model_spec "$model")"
  local accum=$(( gbatch / (ngpu * PER_DEVICE) )); [ "$accum" -lt 1 ] && accum=1
  local out_dir="$OUTPUT_ROOT/$model/weasel"
  local cfg="$out_dir/train_config.rendered.yaml"
  mkdir -p "$out_dir"
  sed -e "s|@MODEL_PATH@|$mpath|g" \
      -e "s|@TEMPLATE@|$template|g" \
      -e "s|@DATASET@|${DATASET_NAME:-weasel_agenttrek}|g" \
      -e "s|@DATASET_DIR@|$LLAMAFACTORY_DIR/data|g" \
      -e "s|@CUTOFF@|$CUTOFF|g" \
      -e "s|@OUTPUT_DIR@|$out_dir|g" \
      -e "s|@LR@|$lr|g" \
      -e "s|@EPOCHS@|$epochs|g" \
      -e "s|@PER_DEVICE@|$PER_DEVICE|g" \
      -e "s|@GRAD_ACCUM@|$accum|g" \
      -e "s|@DS_LINE@|$DS_LINE|g" \
      configs/llamafactory/weasel_sft.template.yaml > "$cfg"
  echo "$cfg"
}

train_ddp() {  # all GPUS, torchrun
  local model="$1"
  read -r mpath _ _ _ _ <<< "$(model_spec "$model")"
  [ -d "$mpath" ] || { echo "[skip] base model missing for $model: $mpath (bash scripts/download_models.sh $model)" >&2; return 0; }
  local cfg; cfg=$(render_cfg "$model" "$NPROC")
  local log="logs/train_${model}.log"
  echo "=== train $model (DDP, GPUs=$GPUS, nproc=$NPROC) cfg=$cfg ===" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPUS" FORCE_TORCHRUN=1 \
    llamafactory-cli train "$cfg" 2>&1 | tee -a "$log"
}

train_one_gpu() {  # single GPU, backgrounded
  local model="$1" gpu="$2"
  read -r mpath _ _ _ _ <<< "$(model_spec "$model")"
  [ -d "$mpath" ] || { echo "[skip] base model missing for $model: $mpath" >&2; return 0; }
  local cfg; cfg=$(render_cfg "$model" 1)
  local log="logs/train_${model}.log"
  echo "=== train $model (GPU $gpu, background) cfg=$cfg ===" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup llamafactory-cli train "$cfg" >> "$log" 2>&1 &
}

if [ "$PARALLEL" = "1" ]; then
  idx=0; pids=()
  for model in $MODELS; do
    gpu="${_gpus_arr[$(( idx % NPROC ))]}"
    train_one_gpu "$model" "$gpu"; pids+=($!); idx=$((idx+1))
  done
  echo "[run_train] launched ${#pids[@]} jobs across GPUs $GPUS. Tail logs/train_*.log"
  wait "${pids[@]}"
else
  for model in $MODELS; do train_ddp "$model"; done
fi
echo "[run_train] done. adapters under $OUTPUT_ROOT/<model>/weasel"
echo "[run_train] next: bash scripts/run_merge.sh"
