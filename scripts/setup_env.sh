#!/usr/bin/env bash
# Source this file once per shell before running any WEASEL stage:
#   source scripts/setup_env.sh
#
# - Exports every path the install / download / run scripts reference.
# - Sends ALL heavy artefacts (venvs, models, datasets, HF cache, checkpoints)
#   to /group-volume; keeps the small /user-volume ($HOME) untouched.
# - **Warns** (does not error) about any input path that doesn't exist yet, and
#   prints the exact script to create it. Never aborts your shell.
# - Provides `weasel_activate {select|train|eval}` to enter the right venv.
#
# Layered design (3 isolated venvs — their deps genuinely conflict):
#   select : torch + bert-score        -> WEASEL data-selection pipeline (this repo)
#   train  : hiyouga/LLaMA-Factory     -> LoRA SFT of Qwen2.5-7B / Gemma3-4B / Qwen3-8B
#   eval   : vllm + agentlab/browsergym-> serve fine-tuned model + run benchmarks

# -----------------------------------------------------------------------------
# Volumes (override BEFORE sourcing if your mounts differ)
# -----------------------------------------------------------------------------
export GROUP_VOLUME="${GROUP_VOLUME:-/group-volume}"     # large, shared  -> everything heavy
export USER_VOLUME="${USER_VOLUME:-$HOME}"               # small, per-user -> code + secrets only
export WEASEL_USER="${WEASEL_USER:-${USER:-$(whoami)}}"

# Repo root (this checkout). Works when sourced via BASH_SOURCE.
export WEASEL_REPO="${WEASEL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

# Single workspace root on group-volume. Everything large hangs off this.
export WEASEL_WORK="${WEASEL_WORK:-$GROUP_VOLUME/$WEASEL_USER/weasel}"

# -----------------------------------------------------------------------------
# Python venvs (on group-volume — torch+vllm+deepspeed are many GB each)
# -----------------------------------------------------------------------------
export WEASEL_VENV_SELECT="${WEASEL_VENV_SELECT:-$WEASEL_WORK/venvs/select}"
export WEASEL_VENV_TRAIN="${WEASEL_VENV_TRAIN:-$WEASEL_WORK/venvs/train}"
export WEASEL_VENV_EVAL="${WEASEL_VENV_EVAL:-$WEASEL_WORK/venvs/eval}"

# hiyouga/LLaMA-Factory checkout (cloned by scripts/install.sh train)
export LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-$WEASEL_WORK/LLaMA-Factory}"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
export WEASEL_DATA="${WEASEL_DATA:-$WEASEL_WORK/data}"
# Raw offline trajectory pools (only needed if you re-run selection yourself).
export AGENTTREK_DIR="${AGENTTREK_DIR:-$WEASEL_DATA/agenttrek}"          # xlang-ai/AgentTrek (~1.8GB)
export TRAIN_INPUT_JSON="${TRAIN_INPUT_JSON:-$WEASEL_DATA/train.json}"   # converted {"messages":[...]} list
# Selection-pipeline intermediates (scripts/run_select.sh writes these).
export GOALS_SCORES_JSON="${GOALS_SCORES_JSON:-$WEASEL_DATA/goals_with_scores.json}"
export SELECTED_INDICES_JSON="${SELECTED_INDICES_JSON:-$WEASEL_DATA/selected_indices_T0_3.json}"
# Final SFT file fed to LLaMA-Factory. Default = the authors' pre-built 10K
# (scripts/download_data.sh grabs it); run_select.sh overwrites it if you
# re-run the pipeline from scratch.
export WEASEL_TRAIN_JSON="${WEASEL_TRAIN_JSON:-$WEASEL_DATA/weasel_agenttrek_train_10k.json}"
# Google-Drive file id from the repo README (pre-built WEASEL-selected 10K).
export WEASEL_TRAIN_GDRIVE_ID="${WEASEL_TRAIN_GDRIVE_ID:-175XAk5NyMxVDRhJUN8x72V7EOfNVWUp2}"

# -----------------------------------------------------------------------------
# Base models (downloaded into group-volume by scripts/download_models.sh).
# Point these at an existing cluster mirror instead if you already have them,
# e.g. export MODEL_QWEN25_7B=/group-volume/nait-models/Qwen2.5-7B-Instruct
# -----------------------------------------------------------------------------
export MODELS_DIR="${MODELS_DIR:-$WEASEL_WORK/models}"
export MODEL_QWEN25_7B="${MODEL_QWEN25_7B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
export MODEL_GEMMA3_4B="${MODEL_GEMMA3_4B:-$MODELS_DIR/gemma-3-4b-it}"
export MODEL_QWEN3_8B="${MODEL_QWEN3_8B:-$MODELS_DIR/Qwen3-8B}"
export MODEL_QWEN35_9B="${MODEL_QWEN35_9B:-$MODELS_DIR/Qwen3.5-9B}"
export QWEN35_9B_TEMPLATE="${QWEN35_9B_TEMPLATE:-qwen3}"          # Qwen3.x chat template
# HF repo ids used for download (only when the local dir above is missing).
export HFID_QWEN25_7B="${HFID_QWEN25_7B:-Qwen/Qwen2.5-7B-Instruct}"
export HFID_GEMMA3_4B="${HFID_GEMMA3_4B:-google/gemma-3-4b-it}"   # GATED on HF
export HFID_QWEN3_8B="${HFID_QWEN3_8B:-Qwen/Qwen3-8B}"
# Qwen3.5-9B repo id is a best guess — VERIFY/override (the model may be a local
# checkpoint or not yet public): export HFID_QWEN35_9B=<org/repo>  (or set MODEL_QWEN35_9B
# to an existing dir to skip the download entirely).
export HFID_QWEN35_9B="${HFID_QWEN35_9B:-Qwen/Qwen3.5-9B}"

# --- Experiment model (exp1 full-data vs exp2 weasel-subset) ---
# "qwen3.5-9b": set MODEL_QWEN35_9B to your /group-volume checkpoint dir.
# Confirm the chat template (Qwen3.x -> 'qwen3'); change if your model differs.
export MODEL_QWEN35_9B="${MODEL_QWEN35_9B:-$MODELS_DIR/Qwen3.5-9B}"
export QWEN35_9B_TEMPLATE="${QWEN35_9B_TEMPLATE:-qwen3}"

# --- NEW experiment dataset (NOT AgentTrek; you provide it) ---
export NEWDATA_RAW="${NEWDATA_RAW:-$WEASEL_DATA/newdata/raw.jsonl}"               # dataset as given
export NEWDATA_FULL_JSON="${NEWDATA_FULL_JSON:-$WEASEL_DATA/newdata/full.json}"   # converted (messages schema) — exp1 input
export NEWDATA_WEASEL_JSON="${NEWDATA_WEASEL_JSON:-$WEASEL_DATA/newdata/weasel_subset.json}"  # after selection — exp2 input
export EXP_OUTPUT_ROOT="${EXP_OUTPUT_ROOT:-$WEASEL_WORK/experiments}"             # exp checkpoints/merged/results

# -----------------------------------------------------------------------------
# Outputs (all on group-volume)
# -----------------------------------------------------------------------------
export OUTPUT_ROOT="${OUTPUT_ROOT:-$WEASEL_WORK/checkpoints}"      # LoRA adapters per model
export MERGED_ROOT="${MERGED_ROOT:-$WEASEL_WORK/merged}"           # merged fp16 models for serving
export EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-$WEASEL_WORK/eval-results}"

# -----------------------------------------------------------------------------
# HF cache redirect (user-volume protection — same rationale as tads)
# -----------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-$WEASEL_WORK/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
# Playwright (browsergym) browser binaries are ~400MB -> group-volume too.
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$WEASEL_WORK/cache/ms-playwright}"

# Extra shared libs for Chromium when `playwright install-deps` can't run (no root).
# Point WEASEL_XLIBS_DIR at a lib dir holding the missing libs (e.g. X11 libs you
# installed without root via `conda create -n weasel-xlibs -c conda-forge \
# xorg-libxcomposite xorg-libxdamage xorg-libxrandr`). If unset, auto-detect a
# conda env named 'weasel-xlibs'. Only applied when the dir actually exists, so on
# a VM that already has the libs this is a harmless no-op.
if [ -z "${WEASEL_XLIBS_DIR:-}" ] && command -v conda >/dev/null 2>&1; then
  _wx="$(conda env list 2>/dev/null | awk '/weasel-xlibs/{print $NF}')"
  [ -n "$_wx" ] && [ -d "$_wx/lib" ] && WEASEL_XLIBS_DIR="$_wx/lib"
  unset _wx
fi
if [ -n "${WEASEL_XLIBS_DIR:-}" ] && [ -d "$WEASEL_XLIBS_DIR" ]; then
  export LD_LIBRARY_PATH="$WEASEL_XLIBS_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# -----------------------------------------------------------------------------
# HF token (for gated google/gemma-3-4b-it). Pick ONE:
#   1) [BEST] huggingface-cli login   (writes ~/.huggingface/token; no env var)
#   2) export HF_TOKEN=hf_xxx in your shell rc BEFORE sourcing
#   3) echo 'export HF_TOKEN=hf_xxx' > ~/.weasel_secrets.sh && chmod 600 ~/.weasel_secrets.sh
# Never hardcode a token here — this file is git-tracked.
# -----------------------------------------------------------------------------
if [ -f "$HOME/.weasel_secrets.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.weasel_secrets.sh"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

# -----------------------------------------------------------------------------
# Offline by default (training/eval read only local disk). download/install
# scripts flip these to 0 in a subshell when they actually need the network.
# -----------------------------------------------------------------------------
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# -----------------------------------------------------------------------------
# Eval-time knobs (only matter for scripts/serve_vllm.sh + scripts/run_eval.sh)
# -----------------------------------------------------------------------------
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-weasel}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://$VLLM_HOST:$VLLM_PORT/v1}"   # AgentLab agent endpoint
# GPT judge key (WebArena-Lite=gpt-4.1-mini, WebArena=gpt-4-1106-preview). Remote API; no VRAM.
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
# MiniWob++ local HTML (set by scripts/install.sh eval after cloning miniwob-plusplus).
export MINIWOB_URL="${MINIWOB_URL:-file://$WEASEL_WORK/miniwob-plusplus/miniwob/html/miniwob/}"
# WorkArena (live ServiceNow developer instance — fill these in before WorkArena eval).
export SNOW_INSTANCE_URL="${SNOW_INSTANCE_URL:-}"
export SNOW_INSTANCE_UNAME="${SNOW_INSTANCE_UNAME:-}"
export SNOW_INSTANCE_PWD="${SNOW_INSTANCE_PWD:-}"

# -----------------------------------------------------------------------------
# Runtime hygiene
# -----------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

# -----------------------------------------------------------------------------
# Create output dirs (idempotent)
# -----------------------------------------------------------------------------
mkdir -p "$WEASEL_WORK" "$WEASEL_DATA" "$WEASEL_DATA/newdata" "$MODELS_DIR" \
         "$OUTPUT_ROOT" "$MERGED_ROOT" "$EXP_OUTPUT_ROOT" "$EVAL_RESULTS_ROOT" \
         "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
         "$PLAYWRIGHT_BROWSERS_PATH" "$WEASEL_REPO/logs" \
         2>/dev/null || true

# -----------------------------------------------------------------------------
# venv activation helper:  weasel_activate {select|train|eval}
# -----------------------------------------------------------------------------
weasel_activate() {
  local which="$1" venv=""
  case "$which" in
    select) venv="$WEASEL_VENV_SELECT" ;;
    train)  venv="$WEASEL_VENV_TRAIN"  ;;
    eval)   venv="$WEASEL_VENV_EVAL"   ;;
    *) echo "[weasel_activate] usage: weasel_activate {select|train|eval}" >&2; return 2 ;;
  esac
  if [ ! -f "$venv/bin/activate" ]; then
    echo "[weasel_activate] venv '$which' not found at $venv" >&2
    echo "[weasel_activate] create it:  bash scripts/install.sh $which" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
  echo "[weasel_activate] $which venv active: $(command -v python)"
}

# -----------------------------------------------------------------------------
# Existence checks (warn-only — never aborts)
# -----------------------------------------------------------------------------
_weasel_missing=0
_weasel_warn() {
  local var="$1" path="$2" fix="$3"
  if [ ! -e "$path" ]; then
    if [ "$_weasel_missing" = "0" ]; then
      echo ""
      echo "------------------------------------------------------------------"
      echo "[setup_env] WARNINGS: the following paths do not exist yet."
      echo "------------------------------------------------------------------"
    fi
    printf "  [missing] %-22s %s\n" "$var" "$path"
    printf "            fix:  %s\n" "$fix"
    _weasel_missing=$((_weasel_missing + 1))
  fi
}
_weasel_warn GROUP_VOLUME       "$GROUP_VOLUME"      "mount group-volume, or: export GROUP_VOLUME=/your/large/mount"
_weasel_warn WEASEL_VENV_SELECT "$WEASEL_VENV_SELECT" "bash scripts/install.sh select"
_weasel_warn WEASEL_VENV_TRAIN  "$WEASEL_VENV_TRAIN"  "bash scripts/install.sh train"
_weasel_warn WEASEL_VENV_EVAL   "$WEASEL_VENV_EVAL"   "bash scripts/install.sh eval"
_weasel_warn WEASEL_TRAIN_JSON  "$WEASEL_TRAIN_JSON"  "bash scripts/download_data.sh   (pre-built 10K) — or run scripts/run_select.sh"
_weasel_warn MODEL_QWEN25_7B    "$MODEL_QWEN25_7B"    "bash scripts/download_models.sh qwen25_7b"

if [ "$_weasel_missing" -gt 0 ]; then
  echo "------------------------------------------------------------------"
  echo "[setup_env] $_weasel_missing path(s) missing — env vars are still exported."
  echo "------------------------------------------------------------------"
else
  echo "[setup_env] All paths verified ✓"
fi
echo ""
echo "WEASEL env loaded."
echo "  WEASEL_WORK        = $WEASEL_WORK   (group-volume workspace)"
echo "  WEASEL_TRAIN_JSON  = $WEASEL_TRAIN_JSON"
echo "  OUTPUT_ROOT        = $OUTPUT_ROOT"
echo "  EVAL_RESULTS_ROOT  = $EVAL_RESULTS_ROOT"
echo "  venvs              = select:$WEASEL_VENV_SELECT  train:$WEASEL_VENV_TRAIN  eval:$WEASEL_VENV_EVAL"
echo "  activate a venv    = weasel_activate {select|train|eval}"
if [ -n "$HF_TOKEN" ]; then echo "  HF_TOKEN           = set (${#HF_TOKEN} chars)"
elif [ -f "$HOME/.huggingface/token" ]; then echo "  HF_TOKEN           = (~/.huggingface/token via huggingface-cli login)"
else echo "  HF_TOKEN           = (UNSET — needed for gated google/gemma-3-4b-it)"; fi

unset -f _weasel_warn
unset _weasel_missing
