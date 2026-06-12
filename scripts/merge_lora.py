#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model (standalone replacement for
`llamafactory-cli export`). The merged model serves directly with vLLM.

  python scripts/merge_lora.py --base <model dir> --adapter <adapter dir> \
      --output <merged dir> [--dtype bf16|fp16]

--adapter accepts either the training output dir (adapter at the root, as
scripts/train_lora_sft.py saves it) or a specific checkpoint-* dir; when the
root has no adapter_config.json the newest checkpoint-* is used.
"""
from __future__ import annotations
import argparse
import os
import re
import sys


def resolve_adapter(path: str) -> str:
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    ckpts = []
    for name in os.listdir(path) if os.path.isdir(path) else []:
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if m and os.path.isfile(os.path.join(path, name, "adapter_config.json")):
            ckpts.append((int(m.group(1)), os.path.join(path, name)))
    if not ckpts:
        sys.exit(f"[merge_lora] no adapter_config.json under {path} (or its checkpoint-* dirs)")
    ckpt = max(ckpts)[1]
    print(f"[merge_lora] using newest checkpoint: {ckpt}")
    return ckpt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="Base model dir or HF id.")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (or its parent).")
    ap.add_argument("--output", required=True, help="Merged model output dir.")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--max-shard-size", default="5GB")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = resolve_adapter(args.adapter)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"[merge_lora] base={args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True)
    print(f"[merge_lora] adapter={adapter}")
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True,
                          max_shard_size=args.max_shard_size)
    # Prefer the tokenizer saved with the adapter (train_lora_sft saves it);
    # fall back to the base model's.
    tok_src = adapter if os.path.isfile(os.path.join(adapter, "tokenizer_config.json")) \
        else args.base
    AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True) \
        .save_pretrained(args.output)
    print(f"[merge_lora] merged model saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
