#!/usr/bin/env bash
# LoRA-SFT the WEASEL-selected data on the A100 cluster with the standalone
# trainer (scripts/train_lora_sft.py — transformers+peft, no LLaMA-Factory).
# Data is a FILE path (JSON list or JSONL of messages[+tools]); no registration.
#
# Default mode: sequential DDP — each model trains across ALL --gpus via
# torchrun. A 7-8B LoRA also fits on ONE 80GB A100, so --parallel runs the
# three models concurrently, one GPU each (cycled), to use the cluster fully.
#
#   bash scripts/run_train.sh --gpus 0,1,2,3,4,5,6,7          # sequential DDP
#   bash scripts/run_train.sh --gpus 0,1,2 --parallel         # 1 model per GPU, concurrent
#   MODELS="qwen25 qwen3" bash scripts/run_train.sh --gpus 0,1
#   --qlora : 4-bit base + Liger fused CE — for 24GB cards (e.g. 2x RTX 4090)
#   --liger : Liger fused CE only — recommended at CUTOFF>=32768 even on A100
#
# Data variant (separates checkpoints so full-data and WEASEL-subset don't clobber):
#   bash scripts/run_train.sh --gpus 0                        # VARIANT=weasel, data=$WEASEL_TRAIN_JSON
#   VARIANT=full DATA_FILE="$NEWDATA_FULL_JSON" bash scripts/run_train.sh --gpus 0
#   -> adapters in $OUTPUT_ROOT/<model>/<variant>. Carry the SAME VARIANT into
#      run_merge.sh / serve_vllm.sh / run_eval.sh.
#
# Per-model recipe is paper-faithful (Table 9); global batch is held constant
# regardless of GPU count by adjusting --grad-accum.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${OUTPUT_ROOT:-}" ] || ! type weasel_activate >/dev/null 2>&1; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
weasel_activate train

MODELS=${MODELS:-"qwen25 gemma3 qwen3"}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
PARALLEL=0
PER_DEVICE=${PER_DEVICE:-1}
CUTOFF=${CUTOFF:-8192}
VARIANT=${VARIANT:-weasel}     # data variant -> checkpoints land in $OUTPUT_ROOT/<model>/<variant>
QLORA=${QLORA:-0}              # 1 = 4-bit base + Liger fused CE (24GB cards)
LIGER=${LIGER:-0}              # 1 = Liger fused CE only (long cutoffs on big cards)
while [ $# -gt 0 ]; do
  case "$1" in
    --parallel) PARALLEL=1; shift ;;
    --qlora) QLORA=1; shift ;;
    --liger) LIGER=1; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;  --gpus=*) GPUS="${1#*=}"; shift ;;
    --cutoff) CUTOFF="$2"; shift 2 ;; --cutoff=*) CUTOFF="${1#*=}"; shift ;;
    --variant) VARIANT="$2"; shift 2 ;; --variant=*) VARIANT="${1#*=}"; shift ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "[warn] unknown arg: $1" >&2; shift ;;
  esac
done
EXTRA_FLAGS=""
[ "$QLORA" = "1" ] && EXTRA_FLAGS="--load-4bit --liger"
[ "$QLORA" = "0" ] && [ "$LIGER" = "1" ] && EXTRA_FLAGS="--liger"

# Default training file per variant; override with DATA_FILE=<path>.
if [ -z "${DATA_FILE:-}" ]; then
  case "$VARIANT" in
    weasel) DATA_FILE="$WEASEL_TRAIN_JSON" ;;
    *) echo "[error] VARIANT=$VARIANT needs DATA_FILE=<training .json/.jsonl> " \
            "(e.g. \$NEWDATA_FULL_JSON). weasel variant defaults to \$WEASEL_TRAIN_JSON." >&2; exit 2 ;;
  esac
fi
[ -f "$DATA_FILE" ] || { echo "[error] training file missing: $DATA_FILE" >&2; exit 1; }
IFS=',' read -r -a _gpus_arr <<< "$GPUS"
NPROC=${#_gpus_arr[@]}

# Per-model: <base-model path> <lr> <epochs> <global-batch>
# (chat template comes from each model's tokenizer; no template arg needed)
model_spec() {
  case "$1" in
    qwen25)    echo "$MODEL_QWEN25_7B 2.0e-5 4.0 8" ;;
    gemma3)    echo "$MODEL_GEMMA3_4B 2.0e-5 2.0 16" ;;
    qwen3)     echo "$MODEL_QWEN3_8B 1.0e-6 2.0 8" ;;
    qwen35_9b) echo "$MODEL_QWEN35_9B 1.0e-6 2.0 8" ;;  # paper Qwen3-8B recipe; tune if needed
    *) return 1 ;;
  esac
}

dl_key() {  # model key here -> download_models.sh key
  case "$1" in
    qwen25) echo qwen25_7b ;; gemma3) echo gemma3_4b ;; qwen3) echo qwen3_8b ;; *) echo "$1" ;;
  esac
}

mkdir -p logs
echo "[run_train] MODELS=$MODELS  GPUS=$GPUS  NPROC=$NPROC  PARALLEL=$PARALLEL  cutoff=$CUTOFF  extra=${EXTRA_FLAGS:-none}"
echo "[run_train] VARIANT=$VARIANT  DATA_FILE=$DATA_FILE  -> checkpoints under \$OUTPUT_ROOT/<model>/$VARIANT"

train_args() {  # $1=model  $2=ngpu_for_this_job  -> prints CLI args for train_lora_sft.py
  local model="$1" ngpu="$2"
  read -r mpath lr epochs gbatch <<< "$(model_spec "$model")"
  local accum=$(( gbatch / (ngpu * PER_DEVICE) )); [ "$accum" -lt 1 ] && accum=1
  local out_dir="$OUTPUT_ROOT/$model/$VARIANT"
  mkdir -p "$out_dir"
  echo "--model-path $mpath --data $DATA_FILE --output-dir $out_dir" \
       "--lr $lr --epochs $epochs --grad-accum $accum --per-device-batch $PER_DEVICE" \
       "--cutoff-len $CUTOFF --lora-r 8 --lora-alpha 8 $EXTRA_FLAGS"
}

train_ddp() {  # all GPUS, torchrun
  local model="$1"
  read -r mpath _ _ _ <<< "$(model_spec "$model")"
  [ -d "$mpath" ] || { echo "[skip] base model missing for $model: $mpath (bash scripts/download_models.sh $(dl_key "$model"))" >&2; return 0; }
  local args; args=$(train_args "$model" "$NPROC")
  local log="logs/train_${model}.log"
  echo "=== train $model (DDP, GPUs=$GPUS, nproc=$NPROC) ===" | tee -a "$log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node "$NPROC" \
    scripts/train_lora_sft.py $args 2>&1 | tee -a "$log"
}

train_one_gpu() {  # single GPU, backgrounded; sets TRAIN_PID ('' when skipped)
  local model="$1" gpu="$2"
  TRAIN_PID=""
  read -r mpath _ _ _ <<< "$(model_spec "$model")"
  [ -d "$mpath" ] || { echo "[skip] base model missing for $model: $mpath (bash scripts/download_models.sh $(dl_key "$model"))" >&2; return 0; }
  local args; args=$(train_args "$model" 1)
  local log="logs/train_${model}.log"
  echo "=== train $model (GPU $gpu, background) ===" | tee -a "$log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" nohup python scripts/train_lora_sft.py $args >> "$log" 2>&1 &
  TRAIN_PID=$!
}

if [ "$PARALLEL" = "1" ]; then
  idx=0; pids=()
  for model in $MODELS; do
    gpu="${_gpus_arr[$(( idx % NPROC ))]}"
    train_one_gpu "$model" "$gpu"
    [ -n "$TRAIN_PID" ] && pids+=("$TRAIN_PID")
    idx=$((idx+1))
  done
  if [ "${#pids[@]}" -gt 0 ]; then
    echo "[run_train] launched ${#pids[@]} jobs across GPUs $GPUS. Tail logs/train_*.log"
    wait "${pids[@]}"
  else
    echo "[run_train] nothing launched — all models were skipped." >&2
  fi
else
  for model in $MODELS; do train_ddp "$model"; done
fi
echo "[run_train] done. adapters under $OUTPUT_ROOT/<model>/$VARIANT"
echo "[run_train] next: VARIANT=$VARIANT bash scripts/run_merge.sh"
