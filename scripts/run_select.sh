#!/usr/bin/env bash
# Re-run the WEASEL data-selection pipeline from raw trajectories.
# (Skip this entirely if you used scripts/download_data.sh prebuilt.)
#
#   prepare_scores  -> goals_with_scores.json   (GPU: BERTScore roberta-large)
#   select_greedy   -> selected_indices.json    (CPU, stdlib)
#   postprocess     -> weasel_*_train_10k.json  (CPU, stdlib)
#
# NOTE: README step 0 'weasel.prune_axtree' is NOT in the repo, so this skips
# pruning and scores the AXTrees as-is. On A100 80GB that is fine (just more
# tokens); results won't be byte-identical to the paper's pruned inputs.
#
# Usage:
#   bash scripts/run_select.sh --gpus 0
#   T0=3 LAMBDA=1.0 BATCH=64 bash scripts/run_select.sh --gpus 0
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${WEASEL_DATA:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
weasel_activate select

GPUS="${GPUS:-0}"
T0="${T0:-3}"                 # paper default per-trajectory budget
LAMBDA="${LAMBDA:-1.0}"       # paper default importance/diversity tradeoff
BATCH="${BATCH:-64}"          # A100 80GB: 64 is safe for roberta-large; lower if OOM
INPUT="${TRAIN_INPUT_JSON}"
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;  --gpus=*) GPUS="${1#*=}"; shift ;;
    --input) INPUT="$2"; shift 2 ;; --input=*) INPUT="${1#*=}"; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "[warn] unknown arg: $1" >&2; shift ;;
  esac
done
FIRST_GPU="${GPUS%%,*}"

if [ ! -f "$INPUT" ]; then
  echo "[error] input not found: $INPUT" >&2
  echo "        run: bash scripts/download_data.sh agenttrek" >&2
  exit 1
fi

mkdir -p logs "$WEASEL_DATA"
LOG="logs/run_select.log"
echo "[run_select] input=$INPUT  T0=$T0  lambda=$LAMBDA  batch=$BATCH  gpu=$FIRST_GPU" | tee "$LOG"

echo "=== 1/3 prepare_scores (GPU $FIRST_GPU) ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$FIRST_GPU" python -m weasel.prepare_scores \
  --input "$INPUT" \
  --output "$GOALS_SCORES_JSON" \
  --augmented-dataset-output "$WEASEL_DATA/train_with_phi_scores.json" \
  --batch-size "$BATCH" 2>&1 | tee -a "$LOG"

echo "=== 2/3 select_greedy (CPU) ===" | tee -a "$LOG"
python -m weasel.select_greedy \
  --input "$GOALS_SCORES_JSON" \
  --output "$SELECTED_INDICES_JSON" \
  --t0-fixed "$T0" --lambda-weight "$LAMBDA" 2>&1 | tee -a "$LOG"

echo "=== 3/3 postprocess_dataset (CPU) -> $WEASEL_TRAIN_JSON ===" | tee -a "$LOG"
python -m weasel.postprocess_dataset \
  --dataset "$INPUT" \
  --selected-indices "$SELECTED_INDICES_JSON" \
  --output "$WEASEL_TRAIN_JSON" \
  --max-user-chars 40000 --max-examples 10000 --seed 0 2>&1 | tee -a "$LOG"

echo "[run_select] done -> $WEASEL_TRAIN_JSON" | tee -a "$LOG"
