"""Resume an AgentLab study from disk against the currently-served model.

Why this exists: scripts/agentlab_eval.py always calls make_study() which
creates a fresh study directory. When an eval is interrupted (e.g. to relaunch
vLLM with different flags), we want to pick up the partially-finished study
instead of restarting from zero. Study.run() already has find_incomplete() +
n_relaunch logic, so loading the existing study and calling run() reuses the
completed task outputs and only re-executes incomplete ones.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"),
    )
    args = ap.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))
    os.environ.setdefault("OPENAI_BASE_URL", args.base_url)
    os.environ.setdefault("OPENAI_API_BASE", args.base_url)

    try:
        from agentlab.experiments.study import Study
    except Exception as e:
        sys.exit(f"[resume_eval] cannot import AgentLab ({e})")

    study_dir = Path(args.study_dir).resolve()
    if not study_dir.is_dir():
        sys.exit(f"[resume_eval] not a directory: {study_dir}")

    print(f"[resume_eval] loading study from {study_dir}")
    study = Study.load(study_dir)
    n_total = len(study.exp_args_list)
    print(f"[resume_eval] {n_total} tasks in study; running with n_jobs={args.n_jobs}")

    study.run(n_jobs=args.n_jobs)
    print("[resume_eval] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
