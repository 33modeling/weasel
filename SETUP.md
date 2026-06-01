# WEASEL — cluster setup (8×A100 80GB)

End-to-end setup for running WEASEL on a multi-GPU cloud VM. All heavy artefacts
(venvs, models, datasets, HF cache, checkpoints) live on **`/group-volume`**; the
small **`/user-volume`** (`$HOME`) holds only this code checkout + secrets.

> These scripts mirror the `tads/scripts` conventions: source `scripts/setup_env.sh`
> once per shell, then run stage scripts. `setup_env.sh` only *warns* about missing
> paths — it never aborts your shell.

## 0. Clone + env

```bash
git clone git@github.com:33modeling/weasel.git
cd weasel
source scripts/setup_env.sh          # exports paths, redirects HF cache to group-volume
```

Override volume/paths before sourcing if your mounts differ, e.g.:
```bash
export GROUP_VOLUME=/group-volume        # large mount
export WEASEL_WORK=/group-volume/$USER/weasel
```

HF token for the gated `google/gemma-3-4b-it` (pick one):
```bash
weasel_activate select && huggingface-cli login      # easiest
# or: echo 'export HF_TOKEN=hf_xxx' > ~/.weasel_secrets.sh && chmod 600 ~/.weasel_secrets.sh
```

## 1. Install (once, needs internet) — 3 isolated venvs on group-volume

```bash
bash scripts/install.sh all          # select + train + eval  (or run one at a time)
```
- **select** — torch (cu124) + bert-score → the data-selection pipeline (this repo)
- **train**  — clones + installs hiyouga/LLaMA-Factory (LoRA SFT)
- **eval**   — vLLM + AgentLab + BrowserGym + Playwright/Chromium + MiniWob++ HTML

(Why three? vLLM pins torch, AgentLab needs Python <3.13, bert-score/LLaMA-Factory
want different transformers — one env would conflict.)

## 2. Get models + data

```bash
bash scripts/download_models.sh all      # Qwen2.5-7B-Instruct, Qwen3-8B, gemma-3-4b-it (gated)
bash scripts/download_data.sh            # pre-built WEASEL-selected 10K (fast path, recommended)
```

**(Optional)** re-run selection from raw AgentTrek instead of the pre-built file:
```bash
bash scripts/download_data.sh agenttrek  # download + convert raw pool
bash scripts/run_select.sh --gpus 0      # prepare_scores(GPU) -> select_greedy -> postprocess
```
⚠️ `weasel/prune_axtree.py` is **missing from the upstream repo** (README: "added soon"),
so `run_select.sh` scores un-pruned AXTrees. Use the pre-built file for paper-faithful inputs.

## 3. Train (LoRA SFT — the 8×A100 step)

```bash
bash scripts/prepare_dataset.sh                      # register dataset with LLaMA-Factory
bash scripts/run_train.sh --gpus 0,1,2,3,4,5,6,7     # sequential DDP, all models
# or use the cluster fully (1 model per GPU, concurrent):
bash scripts/run_train.sh --gpus 0,1,2 --parallel
```
Recipe is paper-faithful (Table 9): LoRA rank 8 / alpha 8 / bf16; Qwen2.5-7B lr 2e-5 ×4ep,
Gemma3-4B lr 2e-5 ×2ep, Qwen3-8B lr 1e-6 ×2ep. Global batch is held constant across GPU counts.
Adapters → `$OUTPUT_ROOT/<model>/weasel`.

> Note: Qwen3-8B in the paper also uses **self-reasoning synthesis** (§2.5), which has
> **no code in this repo**. Training Qwen3 here uses the same selected data without that
> step, so it won't reproduce the Table-4 RS gain.

## 4. Merge + serve

```bash
bash scripts/run_merge.sh                  # LoRA -> merged fp16 under $MERGED_ROOT
bash scripts/serve_vllm.sh qwen25 --gpus 0 # OpenAI-compatible endpoint on :8000 (leave running)
```

## 5. Evaluate (zero-shot, in a second shell)

```bash
source scripts/setup_env.sh
bash scripts/run_eval.sh --bench miniwob              # fully local — start here
bash scripts/run_eval.sh --bench workarena_l1         # needs a free ServiceNow dev instance (SNOW_* vars)
HOST=<webarena-dns> bash scripts/setup_webarena.sh env && source $WEASEL_WORK/webarena_env.sh
bash scripts/run_eval.sh --bench webarena --n-jobs 4  # needs self-hosted WebArena (docker / AWS AMI)
```
Results → `$EVAL_RESULTS_ROOT`. The GPT success judges (WebArena) need `OPENAI_API_KEY`.

### Benchmark difficulty
| bench | local? | extra requirement |
|---|---|---|
| MiniWob++ | ✅ fully local | none |
| WorkArena L1/L2 | ⚠️ | free ServiceNow developer instance + `SNOW_*` |
| WebArena / -Lite | ❌ heavy | Docker-hosted sites (~100GB–1TB) — official AWS AMI recommended |

## Volume layout
```
/group-volume/$USER/weasel/        ← WEASEL_WORK (everything heavy)
├── venvs/{select,train,eval}/     ← the 3 venvs
├── LLaMA-Factory/                 ← training engine
├── models/                        ← base checkpoints
├── data/                          ← AgentTrek + weasel_agenttrek_train_10k.json
├── checkpoints/<model>/weasel/    ← OUTPUT_ROOT (LoRA adapters)
├── merged/<model>/weasel/         ← merged fp16 for serving
├── eval-results/                  ← AgentLab study outputs
└── cache/huggingface, cache/ms-playwright
~/weasel/                          ← this code checkout (user-volume)
```

## Caveat on `scripts/agentlab_eval.py`
The AgentLab agent/model-args class names shift between releases. If eval errors at
agent construction, adjust only the block marked `### AGENTLAB-VERSION-SENSITIVE`
to match your installed AgentLab (see github.com/ServiceNow/AgentLab examples).
Everything else (install/download/select/train/merge/serve) is version-stable.
