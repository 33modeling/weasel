# Two experiments on a new dataset (qwen3.5-9b)

Goal: compare **full-data** vs **WEASEL-selected-subset** LoRA SFT under the
paper's recipe, measuring SR on a benchmark. Both experiments use the **same**
LoRA recipe (paper Qwen3-8B row: rank 8 / alpha 8 / lr 1e-6 / 2 epochs / bf16);
**only the training data differs** — exactly the paper's `+Full` vs `+Weasel` design.

| | Exp 1 (`full`) | Exp 2 (`weasel`) |
|---|---|---|
| data | full dataset (`$NEWDATA_FULL_JSON`) | WEASEL-selected subset (`$NEWDATA_WEASEL_JSON`) |
| finetune | LoRA (paper setup) | LoRA (identical) |
| trainer | `scripts/train_lora_sft.py` — recipe passed as flags by `run_experiment.sh` | same |
| eval | MiniWob (smoke) via existing harness | MiniWob (smoke) via existing harness |

Prereqs: complete `SETUP.md` steps 0–1 (`source scripts/setup_env.sh`, `bash scripts/install.sh all`).

## 1. Point at the model
`qwen3.5-9b` is a local group-volume checkpoint:
```bash
export MODEL_QWEN35_9B=/group-volume/<...>/qwen3.5-9b   # your path
```
The chat template comes from the checkpoint's tokenizer (the trainer aborts
with a clear error if the tokenizer has none).
No local checkpoint? `bash scripts/download_models.sh qwen35_9b` pulls
`Qwen/Qwen3.5-9B` (repo id verified on HF) into `$MODELS_DIR/Qwen3.5-9B`.

## 2. Prepare the NEW dataset  ← **needs the actual data**
The dataset is not in WEASEL schema, so convert it to a `messages` JSON first
(this is both the `train_lora_sft.py` training input and the WEASEL selection
input). If the data is already chat-shaped (`messages`/`conversations`, even with
`tools`/`tool_calls`), the trainer reads it directly and only WEASEL selection
needs the conversion:
```bash
python scripts/inspect_dataset.py /path/to/raw.jsonl          # discover its fields
# then convert (flags depend on the fields inspect shows):
python scripts/convert_dataset.py --in /path/to/raw.jsonl --out "$NEWDATA_FULL_JSON" \
    --user-field <...> --assistant-field <...> [--system-field <...>] \
    --goal-field <...>     # injects '## Goal:' so WEASEL can group trajectories (exp2)
```
> **Two things I still need from you to finalize:** (a) the dataset's exact fields
> (run `inspect_dataset.py` and share the output), and (b) how it defines a
> *goal / trajectory / step* — WEASEL importance = BERTScore(goal, state) and the
> greedy budget are per-trajectory, so a single-turn dataset needs a different
> mapping than a multi-step one. With that I'll lock `convert_dataset.py` and the
> selection field-mapping.

## 3. Run experiment 1 (full data)
```bash
bash scripts/run_experiment.sh --exp full --gpus 0,1,2,3,4,5,6,7
# -> trains LoRA on full data, merges, serves (vLLM), runs MiniWob via run_eval.sh
#    SR/results: $EXP_OUTPUT_ROOT/full/qwen35_9b/eval/full
#    (run_eval.sh routes per VARIANT; run_experiment.sh passes --variant full)
```

## 4. Run experiment 2 (WEASEL subset)
```bash
bash scripts/run_experiment.sh --exp weasel --gpus 0,1,2,3,4,5,6,7
# -> run_select on the full data -> subset -> LoRA -> merge -> serve -> MiniWob
#    SR/results: $EXP_OUTPUT_ROOT/weasel/qwen35_9b/eval/weasel
```
`run_experiment.sh` reuses `scripts/run_select.sh` (prune_axtree→prepare_scores→
select_greedy→postprocess) for selection and `scripts/run_eval.sh` +
`scripts/agentlab_eval.py` (the existing benchmark) for SR. Pruning is a no-op
for data without `## AXTree:` sections; `PRUNE=0` disables it explicitly.

## Caveats
- **MiniWob is a smoke test.** The benchmark for the new dataset still needs design
  (you flagged this). If the new data isn't web-agent data, MiniWob SR mainly
  validates the train→serve→eval plumbing, not task performance.
- **WEASEL selection assumes web-agent trajectories** (goal + AXTree states + steps).
  Applying it to arbitrary data requires the goal/step mapping in step 2.
- **Watch the cutoff.** Loss is on assistant turns only, and an example whose
  assistant tokens all fall beyond `--cutoff` is dropped. The Gemini
  function-calling exports carry a ~20K-token system prompt, so the default
  `CUTOFF=8192` would drop *every* conversation — run those with
  `CUTOFF=32768` (the trainer prints how many examples survive).
- Hyperparameters are the paper's Qwen3-8B values; tune for the new data via
  `run_experiment.sh`'s `--cutoff` flag and the `LR=` / `EPOCHS=` env overrides
  (e.g. `LR=2e-6 EPOCHS=3 bash scripts/run_experiment.sh --exp full ...`).
