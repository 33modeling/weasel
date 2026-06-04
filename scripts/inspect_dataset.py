#!/usr/bin/env python3
"""Inspect an arbitrary dataset so we can wire the WEASEL/LLaMA-Factory mapping.

  python scripts/inspect_dataset.py /path/to/your_dataset.{json,jsonl,parquet} [-n 3]

Prints: row count, top-level fields + value types, and a couple of pretty-printed
sample records (long strings truncated). Use this to tell me how the new dataset
encodes goal / state / step / response so I can finalize convert_dataset.py and the
selection field-mapping.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter


def load_rows(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            sys.exit("[inspect] need pandas for parquet: pip install pandas pyarrow")
        return pd.read_parquet(path).to_dict(orient="records")
    text = open(path, encoding="utf-8").read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(l) for l in text.splitlines() if l.strip()]


def shorten(v, limit=240):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + f" …(+{len(s)-limit} chars)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-n", "--samples", type=int, default=2)
    args = ap.parse_args()

    rows = load_rows(args.path)
    print(f"[inspect] {args.path}\n[inspect] rows: {len(rows)}")
    if not rows:
        return 0

    keys = Counter()
    types = {}
    for r in rows[:2000]:
        if isinstance(r, dict):
            for k, v in r.items():
                keys[k] += 1
                types.setdefault(k, type(v).__name__)
    print("\n[inspect] top-level fields (count / type):")
    for k, c in keys.most_common():
        print(f"  - {k:<24} {c:>5}  {types.get(k)}")

    # If there's a 'messages'/'conversations' field, summarise roles.
    for conv_key in ("messages", "conversations", "conversation"):
        if conv_key in keys:
            roles = Counter()
            for r in rows[:2000]:
                for m in (r.get(conv_key) or []):
                    if isinstance(m, dict):
                        roles[m.get("role") or m.get("from")] += 1
            print(f"\n[inspect] roles inside '{conv_key}': {dict(roles)}")

    print(f"\n[inspect] first {min(args.samples,len(rows))} record(s):")
    for i, r in enumerate(rows[: args.samples]):
        print(f"\n----- record[{i}] -----")
        if isinstance(r, dict):
            for k, v in r.items():
                if k in ("messages", "conversations", "conversation") and isinstance(v, list):
                    print(f"  {k}:")
                    for m in v:
                        role = (m.get("role") or m.get("from")) if isinstance(m, dict) else "?"
                        content = (m.get("content") or m.get("value")) if isinstance(m, dict) else m
                        print(f"    [{role}] {shorten(content)}")
                else:
                    print(f"  {k}: {shorten(v)}")
        else:
            print(f"  {shorten(r)}")
    print("\n[inspect] Share this output so the converter + selection mapping can be finalized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
