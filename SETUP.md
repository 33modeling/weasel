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
- **train**  — torch (cu124) + transformers + peft + bitsandbytes + liger-kernel
  → standalone LoRA SFT (`scripts/train_lora_sft.py` / `scripts/merge_lora.py` —
  no LLaMA-Factory)
- **eval**   — vLLM + AgentLab + BrowserGym + Playwright/Chromium + MiniWob++ HTML

(Why three? vLLM pins torch, AgentLab needs Python <3.13, bert-score and the
trainer want different transformers — one env would conflict.)

## 2. Get models + data

```bash
bash scripts/download_models.sh all      # Qwen2.5-7B-Instruct, Qwen3-8B, gemma-3-4b-it (gated), Qwen3.5-9B
bash scripts/download_data.sh            # pre-built WEASEL-selected 10K (fast path, recommended)
```
Single model: `download_models.sh {qwen25_7b|gemma3_4b|qwen3_8b|qwen35_9b}`.

> **Qwen3.5-9B (`qwen35_9b`)** uses `HFID_QWEN35_9B` (default `Qwen/Qwen3.5-9B`,
> verified on HF) and `MODEL_QWEN35_9B` (default `$MODELS_DIR/Qwen3.5-9B`). To use
> an existing local checkpoint instead, point `MODEL_QWEN35_9B` at it (a present
> `config.json` makes the download a no-op). The chat template comes from the
> checkpoint's tokenizer — no template setting needed.

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
* **(b)** is the middle option — selection applied, re-emitted as native
  function-calling ShareGPT (`function_call`/`observation` turns), which
  `scripts/train_lora_sft.py` trains directly.

* **(a) paper-faithful** — each trajectory is exploded into per-step ShareGPT records
  (action serialized into assistant text, `## Goal/## AXTree/# Observation` markers
  injected), so the **default** `prepare_scores` settings and `select_greedy` t0-per-
  trajectory recipe apply unchanged:
  ```bash
  TRAIN_INPUT_JSON=$WEASEL_DATA/gemini_steps.jsonl bash scripts/run_select.sh --gpus 0
  CUTOFF=32768 bash scripts/run_train.sh --gpus 0   # trains on $WEASEL_TRAIN_JSON
  ```
* **(b) real use** — each trajectory stays a single native function-calling record
  (`function_call`/`observation` turns + `tools` column). Apply WEASEL by mapping
  the steps it selected back to whole trajectories:
  ```bash
  python -m weasel.select_trajectories \
    --selected-dataset $WEASEL_TRAIN_JSON --traj-dataset $WEASEL_DATA/gemini_traj.jsonl \
    --output $WEASEL_DATA/gemini_traj_selected.jsonl
  DATA_FILE=$WEASEL_DATA/gemini_traj_selected.jsonl CUTOFF=32768 bash scripts/run_train.sh --gpus 0
  ```
  > `CUTOFF=32768` matters for these exports: their system prompt alone is
  > ~20K tokens, and the default 8192 would drop every conversation (loss is
  > assistant-only; examples whose assistant tokens are all truncated are dropped).

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

Training is `scripts/train_lora_sft.py` (transformers + peft; chat formatting
comes from each model's tokenizer template, loss on assistant turns only).
It reads the training FILE directly — no dataset registration step:

```bash
bash scripts/run_train.sh --gpus 0,1,2,3,4,5,6,7     # sequential DDP, all models
# or use the cluster fully (1 model per GPU, concurrent):
bash scripts/run_train.sh --gpus 0,1,2 --parallel
```
Recipe is paper-faithful (Table 9): LoRA rank 8 / alpha 8 / bf16; Qwen2.5-7B lr 2e-5 ×4ep,
Gemma3-4B lr 2e-5 ×2ep, Qwen3-8B lr 1e-6 ×2ep. Global batch is held constant across GPU counts.

**Model keys** (`MODELS=` selects which to run; default `qwen25 gemma3 qwen3`):

| key | base | lr / epochs / global-batch |
|---|---|---|
| `qwen25` | Qwen2.5-7B-Instruct | 2e-5 / 4 / 8 |
| `gemma3` | gemma-3-4b-it | 2e-5 / 2 / 16 |
| `qwen3` | Qwen3-8B | 1e-6 / 2 / 8 |
| `qwen35_9b` | Qwen3.5-9B | 1e-6 / 2 / 8 — inherits the Qwen3-8B recipe; tune in `model_spec` if needed |

`qwen35_9b` is **not** in the default set — run it explicitly (same for merge/serve):
```bash
MODELS="qwen35_9b" bash scripts/run_train.sh --gpus 0
MODELS="qwen35_9b" bash scripts/run_merge.sh
bash scripts/serve_vllm.sh qwen35_9b --gpus 0
# full-data vs WEASEL-subset for this model:
VARIANT=full DATA_FILE="$NEWDATA_FULL_JSON" MODELS="qwen35_9b" bash scripts/run_train.sh --gpus 0
```

**Low-VRAM / long-cutoff (`--qlora`, `--liger`):** the [seq, vocab] logits
tensor dominates memory at long cutoffs — at 32K tokens × 248K vocab
(Qwen3.5-9B) it is ~16GB bf16 plus a ~33GB fp32 copy inside the loss, so even
an 80GB A100 is borderline. `--liger` (Liger fused cross-entropy) never
materializes it. On 24GB cards (e.g. 2× RTX 4090) the bf16 weights alone
(~18GB for 9B) don't fit either — `--qlora` loads the base in 4-bit NF4
(~5.5GB) *and* enables Liger, which fits 9B @ 32K in roughly 17–19GB:
```bash
# 2x RTX 4090, Gemini exports:
QLORA=1 CUTOFF=32768 DATA_FILE=$WEASEL_DATA/gemini_traj_selected.jsonl \
  MODELS="qwen35_9b" bash scripts/run_train.sh --gpus 0,1
# A100 at CUTOFF>=32768 — fused CE alone is enough:
LIGER=1 CUTOFF=32768 MODELS="qwen35_9b" bash scripts/run_train.sh --gpus 0,1,2,3,4,5,6,7
```
The QLoRA adapter merges with the regular `run_merge.sh` (into the full-precision
base — standard practice; tiny numerical drift vs the quantized base is expected).

**Where checkpoints go:** LoRA adapters (not full models) are written to
`$OUTPUT_ROOT/<model>/<variant>/` — e.g. `…/checkpoints/qwen25/weasel/` — with one
`checkpoint-*` per epoch (plus the final adapter at the dir root) and
`logs/train_<model>.log`.

**Data variant (full vs WEASEL-subset):** `VARIANT` (default `weasel`) is the data
tag threaded through train → merge → serve → eval so the two never clobber each other.
The recipe is identical; **only the dataset differs** (paper's `+Full` vs `+Weasel`):
```bash
# WEASEL-subset (default)
bash scripts/run_train.sh --gpus 0                                  # data $WEASEL_TRAIN_JSON
# Full data — point DATA_FILE at the training file and tag the variant
VARIANT=full DATA_FILE="$NEWDATA_FULL_JSON" bash scripts/run_train.sh --gpus 0
```
→ adapters land in `…/<model>/weasel/` vs `…/<model>/full/`.

> Note: Qwen3-8B in the paper also uses **self-reasoning synthesis** (§2.5), which has
> **no code in this repo**. Training Qwen3 here uses the same selected data without that
> step, so it won't reproduce the Table-4 RS gain.

## 4. Merge + serve

`run_merge.sh` fuses the LoRA adapter back into the base model to produce a standalone
bf16 model (vLLM can't serve a bare adapter):
```
$OUTPUT_ROOT/<model>/<variant>  (adapter) + base model
   --(scripts/merge_lora.py)-->  $MERGED_ROOT/<model>/<variant>  (merged bf16, sharded)
```
```bash
bash scripts/run_merge.sh                  # VARIANT=weasel: …/merged/<model>/weasel
bash scripts/serve_vllm.sh qwen25 --gpus 0 # serves that merged model on :8000 (leave running)
# full-data variant — carry the SAME VARIANT:
VARIANT=full bash scripts/run_merge.sh && VARIANT=full bash scripts/serve_vllm.sh qwen25 --gpus 0
```

### Serving Qwen3.5-9B via vLLM (one-time overlay)

vLLM 0.22.1 only registers the multimodal `Qwen3_5ForConditionalGeneration`
arch; the standalone text path fails at hybrid (full + linear) KV-cache
unify. Compounding that, `scripts/merge_lora.py` writes the tuned weights
under the multimodal namespace (`model.language_model.*`), so they are
already in the shape the multimodal arch expects — they just need the base's
`visual.*` + `mtp.*` tensors next to them. The overlay helper writes that
combined checkpoint to `…/merged/qwen35_9b/weasel_mm/`:

```bash
/path/to/venvs/train/bin/python scripts/_overlay_text_into_mm.py \
   --base "$MODELS_DIR/Qwen3.5-9B" \
   --text "$MERGED_ROOT/qwen35_9b/weasel" \
   --out  "$MERGED_ROOT/qwen35_9b/weasel_mm"
```

Then serve `weasel_mm/` (not `weasel/`) with a few extra flags:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "$MERGED_ROOT/qwen35_9b/weasel_mm" \
   --served-model-name weasel --host 127.0.0.1 --port 8002 \
   --max-model-len 32768 --trust-remote-code --gpu-memory-utilization 0.83 \
   --enforce-eager --language-model-only --skip-mm-profiling
```

- `--language-model-only --skip-mm-profiling`: zero out the multimodal limits
  so the agent only sends text.
- `VLLM_USE_FLASHINFER_SAMPLER=0`: avoid FlashInfer's runtime nvcc JIT (no
  CUDA toolkit on the box); the PyTorch-native top-p/top-k sampler is fine.
- `--enforce-eager`: skip torch.compile/CUDA-graph capture — quicker boot
  and one less thing that depends on a specific toolchain.
- `Qwen3.5-9B` needs `transformers>=5.12.0` (model_type `qwen3_5_text` was
  added there). If the eval venv was provisioned earlier, upgrade with
  `pip install --upgrade 'transformers==5.12.0'`.

For the standalone text-only route (e.g. debugging without overlay), the
shim at `scripts/_vllm_qwen35.py` registers the missing
`Qwen3_5ForCausalLM` arch and a weight-name remap that strips the extra
`language_model.` segment — call it exactly like `vllm`. The multimodal
overlay path above is what actually serves cleanly today; the shim is kept
for reference.

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

`run_eval.sh` also writes an **HTML trajectory-analysis report**
(`logs/eval_<variant>_<bench>_report.html`) via `scripts/miniwob_report.py`.
Because it runs in the eval venv it reads the full per-step trajectories and
reports: overall + per-task success rate (worst tasks first), step-count
distribution (success vs failure), action-type frequency, a failure-mode
breakdown (agent/env error · out-of-steps · wrong-answer · action-exec errors),
token/time cost signals, and a per-episode drill-down of the action sequence
(with the agent's `think`, reward, and action errors). Re-generate or point it
at any past study standalone:
```bash
python scripts/miniwob_report.py --study-dir eval-results/weasel/<study> --out report.html
# (outside the eval venv it degrades to summary_info.json-only task stats)
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
├── models/                        ← base checkpoints
├── data/                          ← AgentTrek + weasel_agenttrek_train_10k.json
├── checkpoints/<model>/<variant>/ ← OUTPUT_ROOT (LoRA adapters; variant=weasel|full|…)
├── merged/<model>/<variant>/      ← merged bf16 for serving
├── eval-results/<variant>/        ← AgentLab study outputs + success-rate summary
└── cache/huggingface, cache/ms-playwright
~/weasel/                          ← this code checkout (user-volume)
```

## Caveat on `scripts/agentlab_eval.py`
The AgentLab agent/model-args class names shift between releases. If eval errors at
agent construction, adjust only the block marked `### AGENTLAB-VERSION-SENSITIVE`
to match your installed AgentLab (see github.com/ServiceNow/AgentLab examples).
Everything else (install/download/select/train/merge/serve) is version-stable.
