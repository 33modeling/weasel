# WEASEL

Official code for the ICML 2026 paper **WEASEL: Out-of-Domain Generalization for Web Agents via Importance-Diversity Data Selection**.

[[Paper](https://arxiv.org/abs/2605.20291)] [[Project Page](https://fatemehpesaran310.github.io/projects/weasel.html)]

WEASEL selects compact, goal-relevant, and diverse web-agent trajectory steps to improve out-of-domain generalization while reducing training cost.

![WEASEL overview](assets/weasel_overview.png)

> **Fork note (`33modeling/weasel`, `dev` branch).** This fork adds a fully
> scripted, end-to-end cluster setup (8×A100 80GB) on top of the upstream
> data-selection pipeline: install → download → select → LoRA-SFT (transformers+peft)
> → merge → serve (vLLM) → evaluate (AgentLab/BrowserGym). Active work lives on
> `dev`. See **[SETUP.md](SETUP.md)** for the full guide and the
> [Quickstart](#quickstart-cluster-dev-branch) below. Upstream:
> [fatemehpesaran310/weasel](https://github.com/fatemehpesaran310/weasel).

This repository contains the cleaned data-selection pipeline:

0. Prune AXTree states.
1. Compute goal-relevance and pairwise distance scores.
2. Run the WEASEL greedy subset-selection objective.
3. Build the final training subset, including length filtering and 10K subsampling.

We do not include the original training datasets in this repository. To download
AgentTrek, please refer to the official [xlang-ai/AgentTrek](https://github.com/xlang-ai/AgentTrek)
repository. In the commands below, replace `path/to/train.json` with the local
path to the downloaded training file.

If you want to skip the preprocessing steps and directly use our WEASEL-selected
training dataset, it will be available here:

- WEASEL-selected AgentTrek training dataset: [weasel_agenttrek_train_10k.json](https://drive.google.com/file/d/175XAk5NyMxVDRhJUN8x72V7EOfNVWUp2/view?usp=sharing)

## Quickstart (cluster, `dev` branch)

Scripted full stack for an 8×A100 80GB VM. Everything heavy goes to
`/group-volume`; `/user-volume` (`$HOME`) holds only this checkout. Full details
and per-benchmark notes are in **[SETUP.md](SETUP.md)**.

```bash
git clone -b dev git@github.com:33modeling/weasel.git
cd weasel
source scripts/setup_env.sh                       # paths, HF-cache redirect, venv helper
bash scripts/install.sh all                       # 3 isolated venvs: select / train / eval
weasel_activate select && huggingface-cli login   # for gated google/gemma-3-4b-it
bash scripts/download_models.sh all               # Qwen2.5-7B / Qwen3-8B / Qwen3.5-9B / Gemma3-4B
bash scripts/download_data.sh                     # pre-built WEASEL-selected 10K (fast path)
bash scripts/run_train.sh --gpus 0,1,2,3,4,5,6,7  # standalone LoRA SFT (paper Table-9 recipe)
bash scripts/run_merge.sh                         # LoRA -> merged bf16
bash scripts/serve_vllm.sh qwen25 --gpus 0        # OpenAI-compatible endpoint (leave running)
bash scripts/run_eval.sh --bench miniwob          # zero-shot eval (start with MiniWob)
```

Scripts (mirror the conventions of our `tads/scripts`):

| script | purpose |
|---|---|
| `scripts/setup_env.sh` | source once: group-volume workspace, HF-cache redirect, offline-by-default, `weasel_activate {select\|train\|eval}`, warn-only path checks |
| `scripts/install.sh` | create the 3 venvs (vLLM / AgentLab / bert-score / the trainer have conflicting deps) |
| `scripts/download_models.sh` · `download_data.sh` | base checkpoints + training data → group-volume |
| `scripts/run_select.sh` | re-run selection (`prune_axtree`→`prepare_scores`→`select_greedy`→`postprocess`) |
| `weasel/select_clean.py` | **alternative** one-pass curation for native function-calling exports (1 line = 1 trajectory): step-level BERTScore φ importance + trajectory-level fingerprint dedup → original schema, no convert round-trip |
| `scripts/run_train.sh` + `train_lora_sft.py` | 8×A100 standalone LoRA SFT — transformers+peft, no LLaMA-Factory (`--gpus`/`--parallel`, constant global batch) |
| `scripts/run_merge.sh` + `merge_lora.py` · `serve_vllm.sh` | merge LoRA + serve for eval |
| `scripts/run_eval.sh` + `agentlab_eval.py` | AgentLab/BrowserGym eval (`--bench miniwob\|webarena\|workarena_l1\|workarena_l2`) |
| `scripts/summarize_results.py` · `miniwob_report.py` | success-rate summary + HTML trajectory-analysis report (per-task SR, action freq, failure modes, per-episode action drill-down) |
| `scripts/setup_webarena.sh` | optional self-hosted WebArena sites (Docker) |

> The manual per-step commands below are still valid; the Quickstart just wraps
> them with cluster-aware paths and venvs.

## 0. AXTree Pruning

We use target-centered AXTree pruning before score computation, with a
threshold-based fallback when the action does not reference a valid bid.
On the cluster, `scripts/run_select.sh` runs this as step 0 by default
(`PRUNE=0` skips it; `WINDOW`/`FALLBACK` override the two knobs below).

```bash
python -m weasel.prune_axtree \
  --input path/to/train.json \
  --output path/to/train_pruned.json \
  --window-size 60 \
  --fallback-threshold 120
```

## 1. Prepare Scores

Run score preprocessing on the downloaded training data:

```bash
python -m weasel.prepare_scores \
  --input path/to/train_pruned.json \
  --output path/to/goals_with_scores.json \
  --augmented-dataset-output path/to/train_with_phi_scores.json
```

## 2. Greedy Selection

Run greedy subset selection using the precomputed scores:

```bash
python -m weasel.select_greedy \
  --input path/to/goals_with_scores.json \
  --output path/to/full_selected_dataset_indices_T0_3.json
```

## 3. Postprocess Dataset

Build the final WEASEL training subset:

```bash
python -m weasel.postprocess_dataset \
  --dataset path/to/train_pruned.json \
  --selected-indices path/to/full_selected_dataset_indices_T0_3.json \
  --output path/to/weasel_train_10k.json \
  --max-user-chars 40000 \
  --max-examples 10000 \
  --seed 0
```

## Alternative: one-pass curation for native FC data (`select_clean`)

Sections 0–3 above are the paper's **step-level** pipeline (AgentTrek-style data:
prune → score → select steps → postprocess). For applying WEASEL to **native
function-calling exports** where *one jsonl line is one whole multi-turn
trajectory* (e.g. Gemini/GPT tool-calling logs), `weasel.select_clean` does the
curation in **one pass on the original file** and re-emits the **original schema** —
no `convert_gemini` round-trip, so the output trains directly with `train_lora_sft.py`.

It keeps each WEASEL signal at the granularity that fits this data shape:

- **importance** — paper-faithful, *step level*: `r_t = BERTScore(obs_history_t, goal)`,
  `phi_t = max(0, r_t − r_{t-1})`, aggregated to a per-trajectory quality
  (`mean(phi)`, or final `r_T`). Read straight from the raw messages, so BERTScore
  stays on short step text (its valid regime).
- **dedup** — *trajectory level*: each trajectory → a short **fingerprint** (its action
  sequence + answer word-shingles); group by task, drop near-duplicate fingerprints
  (Jaccard ≥ threshold), keep the highest-quality representative. `O(N)` — no
  whole-trajectory BERTScore (which collapses on the shared 65k-char system prompt),
  no global all-pairs.

```bash
# importance (GPU/bert_score) + near-duplicate dedup, keep the top half per task
python -m weasel.select_clean \
  --input export.jsonl --output weasel_clean.jsonl --keep-frac 0.5

# dedup only — pure stdlib, no GPU
python -m weasel.select_clean --input export.jsonl --output weasel_clean.jsonl --no-importance
```

Key knobs: `--quality meanphi|final`, `--near-dup-threshold` (0.9), `--keep-frac` /
`--keep-k` (per task), `--min-steps`, `--task-field` (default `__source_task__`, else
the goal text). The output is original-schema records → trains directly via **Training** below.

## Training

The paper used [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
for supervised fine-tuning. This fork ships an equivalent standalone trainer —
`scripts/train_lora_sft.py` (transformers + peft, same recipe: LoRA rank 8 /
alpha 8 / bf16, loss on assistant turns only; per-model lr and epochs follow
the paper) — so the WEASEL-selected file trains with no extra framework:

```bash
python scripts/train_lora_sft.py \
  --model-path <base model> --data path/to/weasel_train_10k.json \
  --output-dir out/adapter --lr 1e-6 --epochs 2
python scripts/merge_lora.py --base <base model> --adapter out/adapter --output out/merged
```

On a cluster, `scripts/run_train.sh` wraps this (DDP via torchrun, per-model
paper recipes, `VARIANT`/`DATA_FILE` data routing). For 24GB GPUs or long
cutoffs, `--qlora` (4-bit base + Liger fused CE) / `--liger` keep memory in
check — see [SETUP.md](SETUP.md).

If you want to directly use our trained model checkpoints, they are available in
the [WEASEL Hugging Face collection](https://huggingface.co/collections/yeonjooooni/weasel):

- Qwen2.5-7B-Instruct WEASEL checkpoint
- Gemma3-4B-IT WEASEL checkpoint
- Qwen3-8B WEASEL checkpoint

## Evaluation

For WebArena evaluation, please refer to [web-arena-x/webarena](https://github.com/web-arena-x/webarena).

For MiniWob evaluation, please refer to the [MiniWob documentation](https://miniwob.farama.org/content/viewing/)
and [Farama-Foundation/miniwob-plusplus](https://github.com/Farama-Foundation/miniwob-plusplus).

For WorkArena evaluation, please refer to [ServiceNow/WorkArena](https://github.com/ServiceNow/WorkArena).

On a cluster, `scripts/serve_vllm.sh` + `scripts/run_eval.sh` drive these via the
[AgentLab](https://github.com/ServiceNow/AgentLab)/BrowserGym harness used in the paper.

## Citation

```bibtex
@inproceedings{pesaranzadeh2026weasel,
  title     = {{WEASEL}: Out-of-Domain Generalization for Web Agents via Importance-Diversity Data Selection},
  author    = {Pesaran Zadeh, Fatemeh and Choi, Seyeon and L\`u, Xing Han and Reddy, Siva and Kim, Gunhee},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
