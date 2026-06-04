"""Map WEASEL-selected steps back to whole trajectories (real-use bridge).

WEASEL selects at the STEP level (weasel.select_greedy → weasel.postprocess_dataset
produce a per-step training file). For native function-calling training you instead
want the FULL trajectories that those steps came from. Both files share `_traj_id`
(written by weasel.convert_gemini), so this script:

  1. reads the WEASEL-selected step dataset and collects its `_traj_id` set,
  2. filters the native-FC trajectory dataset down to those trajectories.

Usage:
  python -m weasel.select_trajectories \
    --selected-dataset data/weasel_agenttrek_train_10k.json \
    --traj-dataset     data/gemini_traj.jsonl \
    --output           data/gemini_traj_selected.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Set, Tuple


def iter_records(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for idx, rec in enumerate(json.load(f)):
                yield idx, rec
            return
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                yield idx, json.loads(line)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected-dataset", required=True,
                    help="WEASEL-selected step file (postprocess output) carrying _traj_id.")
    ap.add_argument("--traj-dataset", required=True,
                    help="Native-FC trajectory dataset from weasel.convert_gemini --mode traj.")
    ap.add_argument("--output", required=True, help="Filtered trajectory dataset (JSONL).")
    ap.add_argument("--strip-meta", action="store_true",
                    help="Drop the _traj_id field from the written trajectories.")
    args = ap.parse_args()

    sel_path, traj_path = Path(args.selected_dataset), Path(args.traj_dataset)
    if not sel_path.exists():
        sys.exit(f"[select_trajectories] not found: {sel_path}")
    if not traj_path.exists():
        sys.exit(f"[select_trajectories] not found: {traj_path}")

    keep = selected_traj_ids(sel_path)
    print(f"[select_trajectories] {len(keep)} distinct trajectories selected by WEASEL")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written = total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for _, rec in iter_records(traj_path):
            total += 1
            tid = rec.get("_traj_id")
            if tid is None or int(tid) not in keep:
                continue
            if args.strip_meta:
                rec.pop("_traj_id", None)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"[select_trajectories] kept {written}/{total} trajectories -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
