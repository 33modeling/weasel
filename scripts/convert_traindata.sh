#!/usr/bin/env bash
# Convert the locally-generated function-calling trajectories (Gemini / GPT-5.4-mini
# exports under train_data/) into WEASEL-ingestible datasets.
#
# Produces, under $WEASEL_DATA:
#   gemini_steps.jsonl  — (a) PAPER-FAITHFUL per-step ShareGPT for the WEASEL
#                          selection pipeline (run_select.sh). Plain ShareGPT, so
#                          it reuses the existing weasel_agenttrek train path.
#   gemini_traj.jsonl   — (b) REAL-USE native function-calling ShareGPT (tool_calls
#                          preserved) for training; subset it with
#                          weasel.select_trajectories after selection.
#
# Usage:
#   bash scripts/convert_traindata.sh                       # both train_data/*.jsonl
#   INPUTS="train_data/a.jsonl train_data/b.jsonl" bash scripts/convert_traindata.sh
#   MODE=step bash scripts/convert_traindata.sh             # only (a) or: traj | both
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${WEASEL_DATA:-}" ] || ! type weasel_activate >/dev/null 2>&1; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
weasel_activate select 2>/dev/null || weasel_activate train 2>/dev/null || true

# Default to every .jsonl under train_data/ unless INPUTS is given.
INPUTS="${INPUTS:-$(ls train_data/*.jsonl 2>/dev/null | tr '\n' ' ')}"
[ -n "$INPUTS" ] || { echo "[error] no inputs; put .jsonl files in train_data/ or set INPUTS=" >&2; exit 1; }
MODE="${MODE:-both}"

STEPS_OUT="$WEASEL_DATA/gemini_steps.jsonl"
TRAJ_OUT="$WEASEL_DATA/gemini_traj.jsonl"
STATS_OUT="$WEASEL_DATA/gemini_convert_stats.json"
mkdir -p "$WEASEL_DATA" logs

echo "[convert] inputs: $INPUTS"
echo "[convert] mode=$MODE  steps=$STEPS_OUT  traj=$TRAJ_OUT"

ARGS=(--input $INPUTS --mode "$MODE" --stats-output "$STATS_OUT")
[ "$MODE" != "traj" ] && ARGS+=(--steps-output "$STEPS_OUT")
[ "$MODE" != "step" ] && ARGS+=(--traj-output "$TRAJ_OUT")

python -m weasel.convert_gemini "${ARGS[@]}" 2>&1 | tee logs/convert_traindata.log

echo "[convert] done."
echo "[convert] (a) paper-faithful selection:"
echo "    TRAIN_INPUT_JSON=$STEPS_OUT bash scripts/run_select.sh --gpus 0"
echo "    CUTOFF=32768 bash scripts/run_train.sh --gpus 0"
echo "[convert] (b) real-use trajectory training (optionally WEASEL-filtered):"
echo "    python -m weasel.select_trajectories \\"
echo "      --selected-dataset \$WEASEL_TRAIN_JSON --traj-dataset $TRAJ_OUT \\"
echo "      --output \$WEASEL_DATA/gemini_traj_selected.jsonl"
echo "    DATA_FILE=\$WEASEL_DATA/gemini_traj_selected.jsonl CUTOFF=32768 bash scripts/run_train.sh --gpus 0"
echo "[convert] note: CUTOFF=32768 is required — these exports carry a ~20K-token system prompt."
