#!/usr/bin/env bash
# One-time installer for the WEASEL full stack. Run on the VM (needs internet).
#
# Usage:
#   bash scripts/install.sh all              # select + train + eval (default)
#   bash scripts/install.sh select           # just the data-selection pipeline
#   bash scripts/install.sh train            # just LLaMA-Factory (LoRA SFT)
#   bash scripts/install.sh eval             # just vLLM + AgentLab/BrowserGym
#
# Everything lands on /group-volume (venvs, LLaMA-Factory clone, playwright
# browsers). Re-running is idempotent: an existing venv is reused/upgraded.
#
# Requires: python3.12 (AgentLab needs >=3.11,<3.13), git, internet.
# A100 80GB (sm_80) works with the default CUDA wheels of every tool below.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${WEASEL_WORK:-}" ]; then
  echo "Sourcing scripts/setup_env.sh..."
  source scripts/setup_env.sh
fi

WHAT="${1:-all}"
PY="${PYTHON_BIN:-python3.12}"
command -v "$PY" >/dev/null || PY="python3"
echo "[install] using interpreter: $($PY --version 2>&1)  ($PY)"
case "$($PY -c 'import sys;print(sys.version_info[:2]>=(3,11) and sys.version_info[:2]<(3,13))')" in
  True) ;;
  *) echo "[install][warn] AgentLab requires Python 3.11 or 3.12; '$PY' may break the eval venv." >&2 ;;
esac

# Network is required for installs; setup_env.sh defaults to offline, so flip
# it for THIS process only (children inherit; the parent shell is untouched).
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0

_mkvenv() {  # $1 = venv path
  if [ ! -f "$1/bin/activate" ]; then
    echo "[install] creating venv $1"
    "$PY" -m venv "$1"
  fi
  # shellcheck disable=SC1091
  source "$1/bin/activate"
  python -m pip install -U pip wheel setuptools
}

install_select() {
  echo "=============================================================="
  echo "[install] (1/3) SELECT venv -> $WEASEL_VENV_SELECT"
  echo "=============================================================="
  _mkvenv "$WEASEL_VENV_SELECT"
  # CUDA torch FIRST so bert-score doesn't drag in a CPU build. A100=sm_80.
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  pip install "bert-score==0.3.13" tqdm
  pip install -U "huggingface_hub[cli]" datasets gdown   # data download/convert
  python -c "import torch,bert_score;print('[select] torch',torch.__version__,'cuda',torch.cuda.is_available())"
  deactivate
}

install_train() {
  echo "=============================================================="
  echo "[install] (2/3) TRAIN venv -> $WEASEL_VENV_TRAIN  (LLaMA-Factory)"
  echo "=============================================================="
  _mkvenv "$WEASEL_VENV_TRAIN"
  if [ ! -d "$LLAMAFACTORY_DIR/.git" ]; then
    echo "[install] cloning LLaMA-Factory -> $LLAMAFACTORY_DIR"
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LLAMAFACTORY_DIR"
  else
    git -C "$LLAMAFACTORY_DIR" pull --ff-only || true
  fi
  # torch + metrics + 4-bit + multi-GPU sharding (deepspeed) for 8xA100.
  pip install -e "$LLAMAFACTORY_DIR[torch,metrics,bitsandbytes,deepspeed]"
  pip install "huggingface_hub[cli]"
  llamafactory-cli version || true
  deactivate
}

install_eval() {
  echo "=============================================================="
  echo "[install] (3/3) EVAL venv -> $WEASEL_VENV_EVAL  (vLLM + AgentLab)"
  echo "=============================================================="
  _mkvenv "$WEASEL_VENV_EVAL"
  # vLLM brings its own pinned torch (CUDA). Install it first, then the harness.
  pip install vllm
  pip install agentlab browsergym               # browsergym = core + all benchmark envs
  pip install browsergym-webarena browsergym-miniwob browsergym-workarena
  python -m playwright install chromium
  python -m playwright install-deps chromium 2>/dev/null || \
    echo "[install][note] 'playwright install-deps' needs root; if it failed, run it with sudo once."
  # MiniWob++ static HTML assets (offline-capable benchmark) -> group-volume.
  if [ ! -d "$WEASEL_WORK/miniwob-plusplus/.git" ]; then
    git clone https://github.com/Farama-Foundation/miniwob-plusplus.git "$WEASEL_WORK/miniwob-plusplus"
    git -C "$WEASEL_WORK/miniwob-plusplus" reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
  fi
  echo "[install] MiniWob HTML ready. setup_env.sh exports MINIWOB_URL=file://$WEASEL_WORK/miniwob-plusplus/miniwob/html/miniwob/"
  deactivate
}

case "$WHAT" in
  select) install_select ;;
  train)  install_train ;;
  eval)   install_eval ;;
  all)    install_select; install_train; install_eval ;;
  *) echo "usage: bash scripts/install.sh {all|select|train|eval}" >&2; exit 2 ;;
esac

echo ""
echo "[install] done: '$WHAT'. Next:"
echo "  1) bash scripts/download_models.sh all     # base checkpoints -> group-volume"
echo "  2) bash scripts/download_data.sh           # pre-built WEASEL 10K (or run_select.sh)"
echo "  3) bash scripts/run_train.sh --gpus 0,1,2,3,4,5,6,7"
