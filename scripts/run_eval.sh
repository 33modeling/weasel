#!/usr/bin/env bash
# Zero-shot benchmark evaluation of a served WEASEL model via AgentLab/BrowserGym.
# Start the model server first:  bash scripts/serve_vllm.sh <model>   (other pane)
#
#   bash scripts/run_eval.sh --bench miniwob                 # fully local, no docker/account
#   bash scripts/run_eval.sh --bench workarena_l1            # needs ServiceNow dev instance
#   bash scripts/run_eval.sh --bench webarena --n-jobs 4     # needs self-hosted WebArena (scripts/setup_webarena.sh)
#   VARIANT=full bash scripts/run_eval.sh --bench miniwob    # eval the full-data model (must match serve_vllm)
#
# bench: miniwob | webarena | webarena_lite | workarena_l1 | workarena_l2
# VARIANT (default weasel) routes results to $EVAL_RESULTS_ROOT/<variant>; set the
# SAME VARIANT you served, so full-data vs WEASEL-subset success rates stay separate.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${EVAL_RESULTS_ROOT:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
weasel_activate eval

BENCH="miniwob"; NJOBS=1; LIMIT=""
VARIANT="${VARIANT:-weasel}"   # which data variant is being evaluated (must match serve_vllm)
while [ $# -gt 0 ]; do
  case "$1" in
    --bench) BENCH="$2"; shift 2 ;; --bench=*) BENCH="${1#*=}"; shift ;;
    --n-jobs) NJOBS="$2"; shift 2 ;; --n-jobs=*) NJOBS="${1#*=}"; shift ;;
    --limit) LIMIT="$2"; shift 2 ;; --limit=*) LIMIT="${1#*=}"; shift ;;
    --variant) VARIANT="$2"; shift 2 ;; --variant=*) VARIANT="${1#*=}"; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "[warn] unknown arg: $1" >&2; shift ;;
  esac
done

# Per-variant results dir so full-data and WEASEL-subset studies never mix
# (summarize_results.py picks the newest study under the root it is given).
RESULTS_DIR="$EVAL_RESULTS_ROOT/$VARIANT"

# AgentLab reaches the served model over the OpenAI-compatible vLLM endpoint.
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://$VLLM_HOST:$VLLM_PORT/v1}"
export OPENAI_BASE_URL="$OPENAI_API_BASE"
export AGENTLAB_EXP_ROOT="$RESULTS_DIR"
mkdir -p "$RESULTS_DIR" logs

# --- per-benchmark preconditions ---
case "$BENCH" in
  miniwob)
    [ -d "${MINIWOB_URL#file://}" ] || echo "[warn] MINIWOB_URL not found: $MINIWOB_URL (run: bash scripts/install.sh eval)"
    ;;
  webarena|webarena_lite)
    : "${WA_SHOPPING:?set WebArena site URLs first — see scripts/setup_webarena.sh (WA_SHOPPING, WA_SHOPPING_ADMIN, WA_REDDIT, WA_GITLAB, WA_WIKIPEDIA, WA_MAP, WA_HOMEPAGE)}"
    [ -n "${OPENAI_API_KEY:-}" ] || echo "[warn] OPENAI_API_KEY unset — the GPT success judge will fail."
    ;;
  workarena_l1|workarena_l2)
    : "${SNOW_INSTANCE_URL:?set SNOW_INSTANCE_URL / SNOW_INSTANCE_UNAME / SNOW_INSTANCE_PWD (free ServiceNow dev instance)}"
    echo "[run_eval] one-time WorkArena data install (idempotent)..."
    workarena-install || echo "[warn] 'workarena-install' failed — verify ServiceNow creds/instance is awake."
    ;;
  *) echo "[error] unknown --bench: $BENCH" >&2; exit 2 ;;
esac

echo "[run_eval] bench=$BENCH  variant=$VARIANT  model=$VLLM_SERVED_NAME  endpoint=$OPENAI_API_BASE  n_jobs=$NJOBS"
EXTRA=(); [ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")
python scripts/agentlab_eval.py \
  --bench "$BENCH" \
  --model-name "$VLLM_SERVED_NAME" \
  --base-url "$OPENAI_API_BASE" \
  --n-jobs "$NJOBS" \
  --out-root "$RESULTS_DIR" \
  "${EXTRA[@]}" 2>&1 | tee "logs/eval_${VARIANT}_${BENCH}.log"
echo "[run_eval] done. results under $RESULTS_DIR"

# Summarize success rate from the study just produced (newest under the variant root).
echo "[run_eval] summarizing success rate..."
python scripts/summarize_results.py --root "$RESULTS_DIR" \
  --csv "logs/eval_${VARIANT}_${BENCH}_summary.csv" 2>&1 | tee -a "logs/eval_${VARIANT}_${BENCH}.log" || \
  echo "[run_eval][warn] summary failed; results still under $RESULTS_DIR"
