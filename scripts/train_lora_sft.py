#!/usr/bin/env python3
"""Standalone LoRA SFT trainer (transformers + peft — no LLaMA-Factory).

Mirrors the paper recipe previously encoded in the LLaMA-Factory template:
LoRA rank 8 / alpha 8 / dropout 0, all linear targets, cosine LR with 0.1
warmup, AdamW, bf16, save per epoch. Launch single-GPU with `python`, or
multi-GPU DDP with `torchrun --nproc_per_node N` (scripts/run_train.sh wraps
both and holds the paper's global batch constant via --grad-accum).

Data: one --data file, JSON list or JSONL. Each record needs "messages"
(OpenAI schema: role system|user|assistant|tool, content; assistant turns may
carry "tool_calls") and may carry "tools" (function declarations, passed to
the chat template). ShareGPT spellings are normalized: "conversations"/
"from"/"value" records, human->user, gpt->assistant, and LLaMA-Factory's
function_call->assistant / observation->tool. So weasel_agenttrek_train_10k
.json, convert_gemini.py outputs, and raw tools+messages exports all train
without registration or conversion.

Loss is computed on assistant turns only. Per-message token spans come from
incremental chat-template renders; every prefix render is verified to be an
exact prefix of the full render, and a sample falls back to training on the
whole sequence when its template breaks that property (counted and reported;
--train-on-all forces the fallback for everything).
"""
from __future__ import annotations
import argparse
import json
import os
import sys

TRAINABLE_ROLES = {"assistant"}
ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "function_call": "assistant",  # LLaMA-Factory sharegpt: assistant tool call
    "observation": "tool",         # LLaMA-Factory sharegpt: tool result
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-path", required=True, help="Base model dir or HF id.")
    ap.add_argument("--data", required=True, help="Training file (.json list or .jsonl).")
    ap.add_argument("--output-dir", required=True, help="Adapter/checkpoint output dir.")
    ap.add_argument("--cutoff-len", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-targets", default="all-linear",
                    help="'all-linear' or comma-separated module names (q_proj,v_proj,...).")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-strategy", default="epoch", choices=["epoch", "steps", "no"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--num-workers", type=int, default=8, help="datasets.map processes.")
    ap.add_argument("--max-samples", type=int, default=None, help="Cap samples (debug).")
    ap.add_argument("--train-on-all", action="store_true",
                    help="Skip assistant-only masking; loss on every token.")
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the last checkpoint in --output-dir.")
    ap.add_argument("--save-state", action="store_true",
                    help="Also save optimizer/scheduler state in checkpoints "
                         "(needed for a faithful --resume; default saves model only).")
    return ap.parse_args()


# --------------------------------------------------------------------------
# Data loading / normalization
# --------------------------------------------------------------------------
def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        raise ValueError(f"empty data file: {path}")
    if text.lstrip().startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list or JSONL file: {path}")
    return data


def normalize_record(rec: dict) -> dict | None:
    """-> {"messages":[{role,content,...}], "tools": list|None} or None to drop."""
    msgs = rec.get("messages")
    if msgs is None and "conversations" in rec:  # ShareGPT spelling
        msgs = [{"role": m.get("from", ""), "content": m.get("value", "")}
                for m in rec["conversations"]]
    if not isinstance(msgs, list) or not msgs:
        return None
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role = ROLE_ALIASES.get(m.get("role", ""), m.get("role", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            return None
        nm = dict(m)
        nm["role"] = role
        if nm.get("content") is None:
            nm["content"] = ""
        out.append(nm)
    if not any(m["role"] in TRAINABLE_ROLES for m in out):
        return None
    tools = rec.get("tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            tools = None
    if isinstance(tools, dict):
        tools = [tools]
    if not tools:
        tools = None
    return {"messages": out, "tools": tools}


# --------------------------------------------------------------------------
# Tokenization with assistant-only loss masking
# --------------------------------------------------------------------------
def render(tok, msgs, tools, add_generation_prompt=False):
    if not msgs:
        return []
    ids = tok.apply_chat_template(
        msgs, tools=tools, tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,   # transformers >= 5 returns a dict by default
    )
    return list(ids)


def build_example(tok, msgs, tools, cutoff_len, train_on_all):
    """-> (input_ids, labels, used_fallback) or None to drop the sample."""
    full = render(tok, msgs, tools)
    if not full:
        return None
    full = full[:cutoff_len]

    if train_on_all:
        return full, list(full), False

    labels = [-100] * len(full)
    fallback = False
    prev = 0
    for k in range(1, len(msgs) + 1):
        ids = render(tok, msgs[:k], tools)
        # The masking math is only valid while each prefix render is an exact
        # prefix of the full render (templates that rewrite earlier turns —
        # e.g. strip <think> from non-final assistant messages — break this).
        n = min(len(ids), len(full))
        if ids[:n] != full[:n]:
            fallback = True
            break
        bound = min(len(ids), len(full))
        if msgs[k - 1]["role"] in TRAINABLE_ROLES:
            start = prev
            head = render(tok, msgs[: k - 1], tools, add_generation_prompt=True)
            # Skip the assistant header tokens when the generation prompt is a
            # verified prefix; otherwise train on the header too (harmless).
            if len(head) > prev and head == full[: len(head)]:
                start = min(len(head), bound)
            labels[start:bound] = full[start:bound]
        prev = bound
        if bound >= len(full):
            break

    if fallback:
        return full, list(full), True
    if all(l == -100 for l in labels):
        return None  # everything trainable was truncated away
    return full, labels, False


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    set_seed(args.seed)
    is_main = int(os.environ.get("RANK", "0")) == 0

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.chat_template is None:
        sys.exit("[train_lora_sft] tokenizer has no chat template; cannot format messages.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        sys.exit("[train_lora_sft] tokenizer has neither pad nor eos token; cannot pad batches.")

    raw = load_records(args.data)
    if args.max_samples:
        raw = raw[: args.max_samples]
    records = [r for r in (normalize_record(x) for x in raw) if r]
    if is_main:
        print(f"[train_lora_sft] {len(records)}/{len(raw)} usable records from {args.data}")
    if not records:
        sys.exit("[train_lora_sft] no usable records (need messages with an assistant turn).")

    def encode(row):
        rec = json.loads(row["payload"])
        out = build_example(tok, rec["messages"], rec["tools"],
                            args.cutoff_len, args.train_on_all)
        if out is None:
            return {"input_ids": [], "labels": [], "fallback": 0}
        ids, labels, fb = out
        return {"input_ids": ids, "labels": labels, "fallback": int(fb)}

    # Records go in as opaque JSON strings: Arrow's struct-schema unification
    # would otherwise union tool-parameter schemas across records and inject
    # null fields into the rendered prompts (or crash on mixed-type columns).
    ds = Dataset.from_list(
        [{"payload": json.dumps(r, ensure_ascii=False)} for r in records])
    ds = ds.map(encode, num_proc=max(1, args.num_workers),
                remove_columns=ds.column_names, desc="tokenize+mask")
    n_before = len(ds)
    n_fallback = sum(ds["fallback"])
    ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    ds = ds.remove_columns(["fallback"])
    if is_main:
        print(f"[train_lora_sft] {len(ds)}/{n_before} examples after tokenization "
              f"(cutoff {args.cutoff_len})")
        if n_fallback:
            pct = 100 * n_fallback / max(1, n_before)
            print(f"[train_lora_sft][warn] {n_fallback} examples ({pct:.0f}%) "
                  "fell back to full-sequence loss (template is not "
                  "prefix-stable for them)")
    if len(ds) == 0:
        sys.exit("[train_lora_sft] all examples were dropped; check --cutoff-len / data.")

    def collate(batch):
        maxlen = max(len(ex["input_ids"]) for ex in batch)
        pad_id = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for ex in batch:
            n = maxlen - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_id] * n)
            labels.append(ex["labels"] + [-100] * n)
            attn.append([1] * len(ex["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long)}

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True)
    model.config.use_cache = False

    targets = args.lora_targets
    if targets != "all-linear":
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=targets))
    if is_main:
        model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        bf16=args.dtype == "bf16",
        fp16=args.dtype == "fp16",
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_only_model=not args.save_state,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_timeout=180000000,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    if args.resume and not args.save_state and is_main:
        print("[train_lora_sft][warn] --resume without --save-state: checkpoints "
              "hold no optimizer/scheduler state, so resuming restarts AdamW "
              "moments and the LR schedule at the checkpoint step.")
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(args.output_dir)   # adapter -> output_dir root
    if trainer.is_world_process_zero():
        tok.save_pretrained(args.output_dir)
        print(f"[train_lora_sft] done. adapter saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
