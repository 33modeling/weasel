"""Map WEASEL-selected steps back to whole trajectories.

WEASEL selects at the STEP level (weasel.select_greedy → weasel.postprocess_dataset
produce a per-step training file). To turn that into trajectory-level data you map
the selected steps' `_traj_id` back to whole trajectories. Two output styles:

  --traj-dataset      filter the CONVERTED native-FC ShareGPT file
                      (weasel.convert_gemini --mode traj) by its `_traj_id` field.

  --original-input    keep the ORIGINAL jsonl style: re-emit the selected
                      trajectories straight from the source export(s), unchanged
                      (tools / messages / tool_calls / reasoning_content / __source__).
                      `_traj_id` is the per-record running index that convert_gemini
                      assigned, so pass the SAME files in the SAME order here.

Provide at least one of the two. Both can be given to emit both styles.

Usage (keep original jsonl schema — selection only, no reformat):
  python -m weasel.select_trajectories \
    --selected-dataset $WEASEL_TRAIN_JSON \
    --original-input train_data/dit_task_0513_gemini_per_line.jsonl \
                     train_data/dit_task_0504_gpt54-mini_multi_harness_task_14k.jsonl \
    --original-output data/selected_original.jsonl

Usage (LLaMA-Factory native-FC ShareGPT):
  python -m weasel.select_trajectories \
    --selected-dataset $WEASEL_TRAIN_JSON \
    --traj-dataset data/gemini_traj.jsonl \
    --output data/gemini_traj_selected.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Set, Tuple


def iter_records(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (record_index, record), skipping blank/corrupt lines.

    Index counts only successfully-parsed, non-blank records — matching the
    `_traj_id` numbering in weasel.convert_gemini, so original-file line indices
    line up with the selected ids."""
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for idx, rec in enumerate(json.load(f)):
                yield idx, rec
            return
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[select_trajectories] skip corrupt line in {path.name}: {e}", file=sys.stderr)
                continue
            yield idx, rec
            idx += 1


def selected_traj_ids(path: Path) -> Set[int]:
    ids: Set[int] = set()
    missing = 0
    for _, rec in iter_records(path):
        tid = rec.get("_traj_id")
        if tid is None:
            missing += 1
        else:
            ids.add(int(tid))
    if missing:
        print(f"[select_trajectories] warning: {missing} selected step(s) had no _traj_id "
              "(were they produced by weasel.convert_gemini?)", file=sys.stderr)
    return ids


def filter_by_field(traj_path: Path, keep: Set[int], out_path: Path, strip_meta: bool) -> Tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = total = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for _, rec in iter_records(traj_path):
            total += 1
            tid = rec.get("_traj_id")
            if tid is None or int(tid) not in keep:
                continue
            if strip_meta:
                rec.pop("_traj_id", None)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written, total


def filter_originals(inputs, keep: Set[int], out_path: Path) -> Tuple[int, int]:
    """Re-emit selected records from the original export(s), schema untouched.

    Records are numbered with a single running counter across `inputs` in order,
    exactly as convert_gemini assigned `_traj_id`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gid = written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for in_path in inputs:
            for _, rec in iter_records(Path(in_path)):
                if gid in keep:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                gid += 1
    return written, gid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected-dataset", required=True,
                    help="WEASEL-selected step file (postprocess output) carrying _traj_id.")
    ap.add_argument("--traj-dataset", default=None,
                    help="Native-FC trajectory dataset from convert_gemini --mode traj to filter.")
    ap.add_argument("--output", default=None, help="Output for --traj-dataset (JSONL).")
    ap.add_argument("--original-input", nargs="+", default=None,
                    help="Original export jsonl(s), SAME files/order as the conversion, to "
                         "re-emit selected trajectories in their original schema.")
    ap.add_argument("--original-output", default=None, help="Output for --original-input (JSONL).")
    ap.add_argument("--strip-meta", action="store_true",
                    help="Drop the _traj_id field from --traj-dataset output.")
    args = ap.parse_args()

    sel_path = Path(args.selected_dataset)
    if not sel_path.exists():
        sys.exit(f"[select_trajectories] not found: {sel_path}")
    if not args.traj_dataset and not args.original_input:
        sys.exit("[select_trajectories] give --traj-dataset and/or --original-input")
    if args.traj_dataset and not args.output:
        sys.exit("[select_trajectories] --output required with --traj-dataset")
    if args.original_input and not args.original_output:
        sys.exit("[select_trajectories] --original-output required with --original-input")

    keep = selected_traj_ids(sel_path)
    print(f"[select_trajectories] {len(keep)} distinct trajectories selected by WEASEL")

    if args.traj_dataset:
        tp = Path(args.traj_dataset)
        if not tp.exists():
            sys.exit(f"[select_trajectories] not found: {tp}")
        w, t = filter_by_field(tp, keep, Path(args.output), args.strip_meta)
        print(f"[select_trajectories] native-FC: kept {w}/{t} -> {args.output}")

    if args.original_input:
        for p in args.original_input:
            if not Path(p).exists():
                sys.exit(f"[select_trajectories] not found: {p}")
        w, t = filter_originals(args.original_input, keep, Path(args.original_output))
        print(f"[select_trajectories] original-style: kept {w}/{t} -> {args.original_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
