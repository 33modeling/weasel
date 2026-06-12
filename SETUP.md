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
bash scripts/download_models.sh all      # Qwen2.5-7B-Instruct, Qwen3-8B, gemma-3-4b-it (gated), Qwen3.5-9B
bash scripts/download_data.sh            # pre-built WEASEL-selected 10K (fast path, recommended)
```
Single model: `download_models.sh {qwen25_7b|gemma3_4b|qwen3_8b|qwen35_9b}`.

> **Qwen3.5-9B (`qwen35_9b`)** uses `HFID_QWEN35_9B` (default `Qwen/Qwen3.5-9B`,
> verified on HF) and `MODEL_QWEN35_9B` (default `$MODELS_DIR/Qwen3.5-9B`). To use
> an existing local checkpoint instead, point `MODEL_QWEN35_9B` at it (a present
> `config.json` makes the download a no-op). Chat template defaults to
> `QWEN35_9B_TEMPLATE=qwen3` — override if your checkpoint differs.

**(Optional)** re-run selection from raw AgentTrek instead of the pre-built file:
```bash
bash scripts/download_data.sh agenttrek  # download + convert raw pool
bash scripts/run_select.sh --gpus 0      # prune_axtree -> prepare_scores(GPU) -> select_greedy -> postprocess
```
Paper step 0 (`weasel.prune_axtree`, target-centered AXTree pruning) runs by default;
`PRUNE=0` skips it, `WINDOW`/`FALLBACK` override the pruning knobs (defaults 60/120).

### 2b. Use your own function-calling trajectories (Gemini / GPT exports)

Locally-generated trajectories under `train_data/*.jsonl` (OpenAI function-calling
schema: `{"tools":[...], "messages":[...]}`, optionally with `__source_*__` meta and
`content` as plain strings *or* `[{"type":"text","text":...}]` blocks) can feed WEASEL
via one converter that emits two shapes:

```bash
bash scripts/convert_traindata.sh        # converts every train_data/*.jsonl
# -> $WEASEL_DATA/gemini_steps.jsonl  (a) and  gemini_traj.jsonl  (b)
```

**Pick by purpose:**
* **Evaluating on MiniWob++ (or any AgentLab benchmark)** → use **(a) paper-style
  step selection**. The selection is the paper recipe, and the per-step ShareGPT
  output trains/serves cleanly into the existing eval path. This is the path to use
  when you want a benchmark success rate.
* **Just training a model to use it** (your own harness, native tool-calling) →
  use **(c) keep the original jsonl**: WEASEL still picks the subset, but the
  selected trajectories stay in their original function-calling schema, unchanged.
* **(b)** is the middle option — selection applied, re-emitted as LLaMA-Factory
  native function-calling ShareGPT (`weasel_gemini_traj`).

* **(a) paper-faithful** — each trajectory is exploded into per-step ShareGPT records
  (action serialized into assistant text, `## Goal/## AXTree/# Observation` markers
  injected), so the **default** `prepare_scores` settings and `select_greedy` t0-per-
  trajectory recipe apply unchanged:
  ```bash
  TRAIN_INPUT_JSON=$WEASEL_DATA/gemini_steps.jsonl bash scripts/run_select.sh --gpus 0
  bash scripts/prepare_dataset.sh && bash scripts/run_train.sh --gpus 0   # dataset weasel_agenttrek
  ```
* **(b) real use** — each trajectory stays a single native function-calling record
  (`function_call`/`observation` turns + `tools` column; registered as
  `weasel_gemini_traj`). Apply WEASEL by mapping the steps it selected back to whole
  trajectories:
  ```bash
  python -m weasel.select_trajectories \
    --selected-dataset $WEASEL_TRAIN_JSON --traj-dataset $WEASEL_DATA/gemini_traj.jsonl \
    --output $LLAMAFACTORY_DIR/data/weasel_gemini_traj.json
  DATASET_NAME=weasel_gemini_traj bash scripts/run_train.sh --gpus 0
  ```

* **(c) keep the original jsonl** — selection only, **no reformat**. WEASEL still
  scores the step projection (a), but the selected trajectories are re-emitted in
  their *original* function-calling schema (`tools`/`messages`/`tool_calls`/
  `reasoning_content`/`__source__`), so you can train them with your own harness:
  ```bash
  python -m weasel.select_trajectories \
    --selected-dataset $WEASEL_TRAIN_JSON \
    --original-input train_data/*.jsonl \
    --original-output $WEASEL_DATA/selected_original.jsonl
  ```
  `_traj_id` is the running record index convert_gemini assigned, so pass the **same
  files in the same order** here (and don't convert with `--limit`, which is debug-only).

Multiple input files are concatenated with globally-unique `_traj_id`, so the Gemini
and GPT-5.4-mini exports train as one pool. WEASEL groups steps by goal text; pass
`--unique-goal` to `weasel.convert_gemini` to force one group per input line instead.

## 3. Train (LoRA SFT — the 8×A100 step)

```bash
bash scripts/prepare_dataset.sh                      # register dataset with LLaMA-Factory
bash scripts/run_train.sh --gpus 0,1,2,3,4,5,6,7     # sequential DDP, all models
# or use the cluster fully (1 model per GPU, concurrent):
bash scripts/run_train.sh --gpus 0,1,2 --parallel
```
Recipe is paper-faithful (Table 9): LoRA rank 8 / alpha 8 / bf16; Qwen2.5-7B lr 2e-5 ×4ep,
Gemma3-4B lr 2e-5 ×2ep, Qwen3-8B lr 1e-6 ×2ep. Global batch is held constant across GPU counts.

**Model keys** (`MODELS=` selects which to run; default `qwen25 gemma3 qwen3`):

| key | base | template | lr / epochs / global-batch |
|---|---|---|---|
| `qwen25` | Qwen2.5-7B-Instruct | qwen | 2e-5 / 4 / 8 |
| `gemma3` | gemma-3-4b-it | gemma3 | 2e-5 / 2 / 16 |
| `qwen3` | Qwen3-8B | qwen3 | 1e-6 / 2 / 8 |
| `qwen35_9b` | Qwen3.5-9B | `$QWEN35_9B_TEMPLATE` (qwen3) | 1e-6 / 2 / 8 — inherits the Qwen3-8B recipe; tune in `model_spec` if needed |

`qwen35_9b` is **not** in the default set — run it explicitly (same for merge/serve):
```bash
MODELS="qwen35_9b" bash scripts/run_train.sh --gpus 0
MODELS="qwen35_9b" bash scripts/run_merge.sh
bash scripts/serve_vllm.sh qwen35_9b --gpus 0
# full-data vs WEASEL-subset for this model:
VARIANT=full DATASET_NAME=<full_dataset> MODELS="qwen35_9b" bash scripts/run_train.sh --gpus 0
```

**Where checkpoints go:** LoRA adapters (not full models) are written to
`$OUTPUT_ROOT/<model>/<variant>/` — e.g. `…/checkpoints/qwen25/weasel/` — with one
`checkpoint-*` per epoch (`save_strategy: epoch`), plus the rendered config
`train_config.rendered.yaml` and `logs/train_<model>.log`.

**Data variant (full vs WEASEL-subset):** `VARIANT` (default `weasel`) is the data
tag threaded through train → merge → serve → eval so the two never clobber each other.
The recipe is identical; **only the dataset differs** (paper's `+Full` vs `+Weasel`):
```bash
# WEASEL-subset (default)
bash scripts/run_train.sh --gpus 0                                  # dataset weasel_agenttrek
# Full data — register it first, then pass VARIANT + DATASET_NAME
VARIANT=full DATASET_NAME=<your_full_dataset> bash scripts/run_train.sh --gpus 0
```
→ adapters land in `…/<model>/weasel/` vs `…/<model>/full/`.

> Note: Qwen3-8B in the paper also uses **self-reasoning synthesis** (§2.5), which has
> **no code in this repo**. Training Qwen3 here uses the same selected data without that
> step, so it won't reproduce the Table-4 RS gain.

## 4. Merge + serve

`run_merge.sh` fuses the LoRA adapter back into the base model to produce a standalone
fp16 model (vLLM can't serve a bare adapter):
```
$OUTPUT_ROOT/<model>/<variant>  (adapter) + base model
   --(llamafactory-cli export)-->  $MERGED_ROOT/<model>/<variant>  (merged fp16, sharded)
```
```bash
bash scripts/run_merge.sh                  # VARIANT=weasel: …/merged/<model>/weasel
bash scripts/serve_vllm.sh qwen25 --gpus 0 # serves that merged model on :8000 (leave running)
# full-data variant — carry the SAME VARIANT:
VARIANT=full bash scripts/run_merge.sh && VARIANT=full bash scripts/serve_vllm.sh qwen25 --gpus 0
```

## 5. Evaluate (zero-shot, in a second shell)

```bash
source scripts/setup_env.sh
bash scripts/run_eval.sh --bench miniwob              # fully local — start here
bash scripts/run_eval.sh --bench workarena_l1         # needs a free ServiceNow dev instance (SNOW_* vars)
HOST=<webarena-dns> bash scripts/setup_webarena.sh env && source $WEASEL_WORK/webarena_env.sh
bash scripts/run_eval.sh --bench webarena --n-jobs 4  # needs self-hosted WebArena (docker / AWS AMI)
```
Results → `$EVAL_RESULTS_ROOT/<variant>/`, with a per-run success-rate summary
(`weasel_summary.json` + `logs/eval_<variant>_<bench>_summary.csv`). The GPT success
judges (WebArena) need `OPENAI_API_KEY`. **Pass the SAME `VARIANT` you served** so
full-data and WEASEL-subset success rates land in separate dirs:
```bash
VARIANT=full bash scripts/run_eval.sh --bench miniwob   # -> eval-results/full/
```

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
├── checkpoints/<model>/<variant>/ ← OUTPUT_ROOT (LoRA adapters; variant=weasel|full|…)
├── merged/<model>/<variant>/      ← merged fp16 for serving
├── eval-results/<variant>/        ← AgentLab study outputs + success-rate summary
└── cache/huggingface, cache/ms-playwright
~/weasel/                          ← this code checkout (user-volume)
```

## Caveat on `scripts/agentlab_eval.py`
The AgentLab agent/model-args class names shift between releases. If eval errors at
agent construction, adjust only the block marked `### AGENTLAB-VERSION-SENSITIVE`
to match your installed AgentLab (see github.com/ServiceNow/AgentLab examples).
Everything else (install/download/select/train/merge/serve) is version-stable.
