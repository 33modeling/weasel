#!/usr/bin/env bash
# Run one of the two paper-faithful experiments end-to-end on qwen3.5-9b:
#
#   bash scripts/run_experiment.sh --exp full   --gpus 0,1,2,3,4,5,6,7
#   bash scripts/run_experiment.sh --exp weasel --gpus 0,1,2,3,4,5,6,7
#
#   full   : LoRA SFT on FULL data ($NEWDATA_FULL_JSON)            -> merge -> serve -> eval
#   weasel : WEASEL-select a subset of $NEWDATA_FULL_JSON -> LoRA  -> merge -> serve -> eval
#
# Same recipe for both (paper Qwen3-8B row: LoRA r8/a8, lr 1e-6, 2 epochs,
# global batch 8, bf16 — passed straight to scripts/train_lora_sft.py); only
# the data differs. Eval reuses the EXISTING benchmark harness
# (scripts/run_eval.sh + scripts/agentlab_eval.py). Default bench = miniwob.
#
# Flags: --bench --cutoff --per-device --serve-gpu --tp --no-eval  (env: LR= EPOCHS= QLORA=1 LIGER=1)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${EXP_OUTPUT_ROOT:-}" ] || ! type weasel_activate >/dev/null 2>&1; then source scripts/setup_env.sh; fi

EXP=""; GPUS="0,1,2,3,4,5,6,7"; BENCH="miniwob"; CUTOFF="${CUTOFF:-8192}"
PER_DEVICE="${PER_DEVICE:-1}"; SERVE_GPU=""; TP="${TP:-1}"; DO_EVAL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --exp) EXP="$2"; shift 2 ;; --exp=*) EXP="${1#*=}"; shift ;;
    --gpus) GPUS="$2"; shift 2 ;; --gpus=*) GPUS="${1#*=}"; shift ;;
    --bench) BENCH="$2"; shift 2 ;; --bench=*) BENCH="${1#*=}"; shift ;;
    --cutoff) CUTOFF="$2"; shift 2 ;; --cutoff=*) CUTOFF="${1#*=}"; shift ;;
    --per-device) PER_DEVICE="$2"; shift 2 ;;
    --serve-gpu) SERVE_GPU="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --no-eval) DO_EVAL=0; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "[warn] unknown arg: $1" >&2; shift ;;
  esac
done
[ "$EXP" = "full" ] || [ "$EXP" = "weasel" ] || { echo "[error] --exp must be full|weasel" >&2; exit 2; }
IFS=',' read -r -a _g <<< "$GPUS"; NPROC=${#_g[@]}; FIRST_GPU="${_g[0]}"
[ -n "$SERVE_GPU" ] || SERVE_GPU="$FIRST_GPU"
GBATCH=8                                   # paper Qwen3-8B global batch
ACCUM=$(( GBATCH / (NPROC * PER_DEVICE) )); [ "$ACCUM" -lt 1 ] && ACCUM=1

MODEL_PATH="$MODEL_QWEN35_9B"
[ -d "$MODEL_PATH" ] || { echo "[error] model not found: $MODEL_PATH  (set MODEL_QWEN35_9B to your group-volume checkpoint)" >&2; exit 1; }

if [ "$EXP" = "full" ]; then
  DATA_FILE="$NEWDATA_FULL_JSON"
else
  DATA_FILE="$NEWDATA_WEASEL_JSON"
fi
EXP_DIR="$EXP_OUTPUT_ROOT/$EXP/qwen35_9b"
ADAPTER_DIR="$EXP_DIR/adapter"; MERGED_DIR="$EXP_DIR/merged"
mkdir -p "$EXP_DIR" logs
echo "[exp:$EXP] model=$MODEL_PATH gpus=$GPUS nproc=$NPROC accum=$ACCUM cutoff=$CUTOFF bench=$BENCH"

# ---------- 0. (weasel only) build the selected subset from the full data ----------
if [ "$EXP" = "weasel" ]; then
  [ -f "$NEWDATA_FULL_JSON" ] || { echo "[error] need full data first: $NEWDATA_FULL_JSON (see EXPERIMENTS.md step 2)" >&2; exit 1; }
  if ! grep -q "## Goal:" "$NEWDATA_FULL_JSON" 2>/dev/null; then
    echo "[exp:weasel][warn] '## Goal:' not found in $NEWDATA_FULL_JSON — WEASEL grouping will collapse."
    echo "                   Re-run convert_dataset.py with --goal-field, or tell me the data's goal/step structure."
  fi
  echo "[exp:weasel] running WEASEL selection -> $NEWDATA_WEASEL_JSON"
  GOALS_SCORES_JSON="$WEASEL_DATA/newdata/goals_with_scores.json" \
  SELECTED_INDICES_JSON="$WEASEL_DATA/newdata/selected_indices_T0_3.json" \
  WEASEL_TRAIN_JSON="$NEWDATA_WEASEL_JSON" \
    bash scripts/run_select.sh --input "$NEWDATA_FULL_JSON" --gpus "$FIRST_GPU"
fi

# ---------- 1. LoRA SFT (standalone trainer, paper Qwen3-8B recipe) ----------
[ -f "$DATA_FILE" ] || { echo "[error] dataset file missing: $DATA_FILE" >&2; exit 1; }
echo "[exp:$EXP] training -> $ADAPTER_DIR  (data=$DATA_FILE)"
weasel_activate train
EXTRA_FLAGS=()
[ "${QLORA:-0}" = "1" ] && EXTRA_FLAGS+=(--load-4bit --liger)
[ "${QLORA:-0}" = "0" ] && [ "${LIGER:-0}" = "1" ] && EXTRA_FLAGS+=(--liger)
CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node "$NPROC" \
  scripts/train_lora_sft.py \
  --model-path "$MODEL_PATH" --data "$DATA_FILE" --output-dir "$ADAPTER_DIR" \
  --lr "${LR:-1e-6}" --epochs "${EPOCHS:-2}" \
  --grad-accum "$ACCUM" --per-device-batch "$PER_DEVICE" \
  --cutoff-len "$CUTOFF" --lora-r 8 --lora-alpha 8 ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
  2>&1 | tee "logs/exp_${EXP}_train.log"

# ---------- 2. merge LoRA -> bf16 ----------
echo "[exp:$EXP] merging -> $MERGED_DIR"
python scripts/merge_lora.py --base "$MODEL_PATH" --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR" 2>&1 | tee "logs/exp_${EXP}_merge.log"

# ---------- 3. serve (background) + eval via existing benchmark harness ----------
if [ "$DO_EVAL" = "0" ]; then
  echo "[exp:$EXP] --no-eval: done. merged model at $MERGED_DIR"; exit 0
fi
echo "[exp:$EXP] serving merged model on GPU $SERVE_GPU (vLLM) + evaluating on $BENCH"
SERVE_LOG="logs/exp_${EXP}_serve.log"
# exec makes SERVE_PID the vllm process itself (not a wrapper bash), so the
# EXIT trap actually frees the GPU/port instead of orphaning the server.
bash -c "source scripts/setup_env.sh >/dev/null 2>&1; weasel_activate eval >/dev/null 2>&1; \
  CUDA_VISIBLE_DEVICES=$SERVE_GPU exec vllm serve '$MERGED_DIR' \
    --served-model-name '$VLLM_SERVED_NAME' --host '$VLLM_HOST' --port '$VLLM_PORT' \
    --tensor-parallel-size $TP --max-model-len 32768 --trust-remote-code" \
  > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT
echo "[exp:$EXP] waiting for vLLM endpoint $OPENAI_API_BASE ..."
READY=0
for i in $(seq 1 120); do
  if curl -fsS "$OPENAI_API_BASE/models" >/dev/null 2>&1; then echo "[exp:$EXP] endpoint ready."; READY=1; break; fi
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "[error] vLLM died — see $SERVE_LOG" >&2; exit 1; }
  sleep 10
done
[ "$READY" = "1" ] || { echo "[error] vLLM endpoint not ready after 20 min — see $SERVE_LOG" >&2; exit 1; }

EXP_RESULT_DIR="$EXP_OUTPUT_ROOT/$EXP/qwen35_9b/eval"
# --variant "$EXP" keeps full vs weasel results (and log/CSV names) separate.
EVAL_RESULTS_ROOT="$EXP_RESULT_DIR" bash scripts/run_eval.sh --bench "$BENCH" --variant "$EXP"
echo "[exp:$EXP] DONE. SR / results under $EXP_RESULT_DIR/$EXP  (adapter=$ADAPTER_DIR merged=$MERGED_DIR)"
