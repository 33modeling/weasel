#!/usr/bin/env python3
"""Convert the NEW dataset into the schema both stages need.

Output = a JSON list of records: {"messages": [{"role","content"}, ...]} with
roles system/user/assistant — which is BOTH:
  - LLaMA-Factory `sharegpt` (training input for exp1 full + exp2 subset), and
  - the WEASEL selection input, IF the user turn carries the headers the
    pipeline greps for: '## Goal:', '## AXTree:', '# Observation of current step:'.

Two paths:
  (A) Already chat-shaped (has 'messages' or 'conversations'): just normalise roles.
  (B) Flat instruction rows (e.g. {instruction,input,output}): map via --*-field flags.

For WEASEL selection on a NEW (non-web) dataset you must define a per-step GOAL so
trajectories can be grouped and importance = BERTScore(goal, state) is meaningful.
Use --goal-field to inject '## Goal: <text>' into the user turn. If your data has no
notion of multi-step trajectories/goal, selection won't match the paper — tell me the
structure and we'll adapt prepare_scores instead of forcing this schema.

Examples:
  # already messages-shaped:
  python scripts/convert_dataset.py --in raw.jsonl --out full.json
  # flat instruction rows -> messages, inject goal for selection:
  python scripts/convert_dataset.py --in raw.jsonl --out full.json \
      --system-field system --user-field instruction --assistant-field output \
      --goal-field instruction
"""
from __future__ import annotations
import argparse, json, os, sys


def load_rows(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    text = open(path, encoding="utf-8").read().strip()
    if not text:
        return []
    try:
        d = json.loads(text)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        return [json.loads(l) for l in text.splitlines() if l.strip()]


def normalise_messages(raw):
    """Path A: coerce existing 'messages'/'conversations' to [{role,content}]."""
    out = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from")
        content = m.get("content") if "content" in m else m.get("value", "")
        role = {"human": "user", "gpt": "assistant", "system": "system"}.get(role, role)
        if role in ("system", "user", "assistant"):
            out.append({"role": role, "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    # Path B field mapping (flat rows -> messages)
    ap.add_argument("--system-field", default=None)
    ap.add_argument("--user-field", default=None)
    ap.add_argument("--assistant-field", default=None)
    # Inject a WEASEL goal header into the user turn (enables selection grouping)
    ap.add_argument("--goal-field", default=None,
                    help="field whose text becomes '## Goal: ...' prepended to the user turn")
    args = ap.parse_args()

    rows = load_rows(args.inp)
    print(f"[convert] loaded {len(rows)} rows from {args.inp}")
    out_records, skipped = [], 0

    for r in rows:
        if isinstance(r, dict) and (r.get("messages") or r.get("conversations") or r.get("conversation")):
            msgs = normalise_messages(r.get("messages") or r.get("conversations") or r.get("conversation"))
        elif args.user_field and args.assistant_field and isinstance(r, dict):
            msgs = []
            if args.system_field and r.get(args.system_field):
                msgs.append({"role": "system", "content": str(r[args.system_field])})
            msgs.append({"role": "user", "content": str(r.get(args.user_field, ""))})
            msgs.append({"role": "assistant", "content": str(r.get(args.assistant_field, ""))})
        else:
            skipped += 1
            continue

        if args.goal_field and isinstance(r, dict) and r.get(args.goal_field):
            goal = str(r[args.goal_field]).strip().replace("\n", " ")
            for m in msgs:
                if m["role"] == "user":
                    m["content"] = f"## Goal: {goal}\n\n{m['content']}"
                    break
        if msgs:
            out_records.append({"messages": msgs})
        else:
            skipped += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out_records, open(args.out, "w"), ensure_ascii=False)
    print(f"[convert] wrote {len(out_records)} records -> {args.out}  (skipped {skipped})")
    if not out_records:
        sys.exit("[convert][error] produced 0 records — pass --user-field/--assistant-field "
                 "for flat rows, or share a sample so we can fix the mapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
