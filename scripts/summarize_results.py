#!/usr/bin/env python3
"""Summarize success rate from an AgentLab/BrowserGym study (MiniWob/WebArena/WorkArena).

AgentLab writes one `summary_info.json` per episode under each experiment dir,
containing at least `cum_reward` (and usually `n_steps`, `terminated`, `err_msg`).
This script walks those files and reports, with NO extra dependencies (stdlib only,
so it runs in any venv):

  * overall success rate      = mean(cum_reward > 0)
  * overall mean reward       = mean(cum_reward)        (partial credit, if any)
  * per-task breakdown        = success rate + mean reward + n episodes
  * errored / missing episodes

Success criterion: cum_reward > 0. MiniWob++ rewards are in [-1, 1] (1 on success,
often a small negative on failure), so >0 is the standard "solved" test; for the
0/1 benchmarks (WebArena/WorkArena) it reduces to reward == 1.

Usage:
  python scripts/summarize_results.py                       # newest study under EVAL_RESULTS_ROOT
  python scripts/summarize_results.py --root ./eval-results # newest study under a root
  python scripts/summarize_results.py --study-dir <path>    # an explicit study dir
  python scripts/summarize_results.py --study-dir <path> --csv out.csv
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path


# Task ids look like "miniwob.click-button", "webarena.123",
# "workarena.servicenow.create-incident" — dot/hyphen/alnum, no underscore
# (so a trailing "_<seed>_<hash>" in the dir name is NOT swallowed into the id).
TASK_RE = re.compile(r"(?:miniwob|webarena|workarena)\.[.A-Za-z0-9-]+")


def newest_study(root: Path) -> Path | None:
    """Newest dir under `root` that contains at least one summary_info.json."""
    candidates = []
    for d in root.iterdir() if root.is_dir() else []:
        if d.is_dir() and any(d.rglob("summary_info.json")):
            candidates.append(d)
    if not candidates:
        # root itself might be the study dir
        return root if any(root.rglob("summary_info.json")) else None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def task_name(exp_dir: Path) -> str:
    """Best-effort task id for an experiment dir (exp_args.pkl -> dir name regex)."""
    pkl = exp_dir / "exp_args.pkl"
    if pkl.exists():
        try:
            import pickle
            with open(pkl, "rb") as fh:
                ea = pickle.load(fh)
            t = getattr(getattr(ea, "env_args", None), "task_name", None)
            if t:
                return str(t)
        except Exception:
            pass
    m = TASK_RE.search(exp_dir.name)
    return m.group(0) if m else exp_dir.name


def collect(study_dir: Path):
    """Return (rows, n_errored, n_missing). rows = list of (task, reward, ok)."""
    rows, n_errored = [], 0
    for sinfo in sorted(study_dir.rglob("summary_info.json")):
        exp_dir = sinfo.parent
        try:
            info = json.loads(sinfo.read_text())
        except Exception:
            n_errored += 1
            continue
        reward = info.get("cum_reward")
        if reward is None or info.get("err_msg"):
            n_errored += 1
            continue
        rows.append((task_name(exp_dir), float(reward), float(reward) > 0))
    return rows, n_errored


def aggregate(rows):
    by_task: dict[str, list] = {}
    for task, reward, ok in rows:
        by_task.setdefault(task, []).append((reward, ok))
    table = []
    for task, vals in sorted(by_task.items()):
        n = len(vals)
        sr = sum(1 for _, ok in vals if ok) / n
        mr = sum(r for r, _ in vals) / n
        table.append((task, n, sr, mr))
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("EVAL_RESULTS_ROOT", "./eval-results"),
                    help="results root; the newest study under it is summarized")
    ap.add_argument("--study-dir", default=None, help="explicit study dir (overrides --root)")
    ap.add_argument("--csv", default=None, help="also write per-task table as CSV here")
    args = ap.parse_args()

    study_dir = Path(args.study_dir) if args.study_dir else newest_study(Path(args.root))
    if not study_dir or not study_dir.exists():
        print(f"[summarize] no study with summary_info.json found under {args.root}", file=sys.stderr)
        return 1

    rows, n_errored = collect(study_dir)
    if not rows:
        print(f"[summarize] no scored episodes in {study_dir} "
              f"({n_errored} errored/incomplete).", file=sys.stderr)
        return 1

    table = aggregate(rows)
    n = len(rows)
    overall_sr = sum(1 for _, _, ok in rows if ok) / n
    overall_mr = sum(r for _, r, _ in rows) / n

    print(f"\n=== study: {study_dir} ===")
    print(f"{'task':<45}{'n':>5}{'success':>10}{'mean_rwd':>10}")
    print("-" * 70)
    for task, cnt, sr, mr in table:
        print(f"{task:<45}{cnt:>5}{sr*100:>9.1f}%{mr:>10.3f}")
    print("-" * 70)
    print(f"{'OVERALL':<45}{n:>5}{overall_sr*100:>9.1f}%{overall_mr:>10.3f}")
    if n_errored:
        print(f"[note] {n_errored} episode(s) errored or had no cum_reward (excluded).")

    # machine-readable copy next to the study
    summary = {
        "study_dir": str(study_dir),
        "n_episodes": n,
        "success_rate": overall_sr,
        "mean_reward": overall_mr,
        "n_errored": n_errored,
        "per_task": [
            {"task": t, "n": c, "success_rate": sr, "mean_reward": mr}
            for t, c, sr, mr in table
        ],
    }
    out_json = study_dir / "weasel_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[summarize] wrote {out_json}")

    if args.csv:
        lines = ["task,n,success_rate,mean_reward"]
        lines += [f"{t},{c},{sr:.4f},{mr:.4f}" for t, c, sr, mr in table]
        lines.append(f"OVERALL,{n},{overall_sr:.4f},{overall_mr:.4f}")
        Path(args.csv).write_text("\n".join(lines) + "\n")
        print(f"[summarize] wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
