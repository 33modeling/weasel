#!/usr/bin/env python3
"""Drive an AgentLab study against a vLLM-served WEASEL model.

This is the ONE piece that is sensitive to the installed AgentLab version:
the agent/model-args classes were renamed across releases. The block marked
`### AGENTLAB-VERSION-SENSITIVE` builds a GenericAgent backed by an
OpenAI-compatible chat model (your vLLM endpoint). If your installed AgentLab
exposes different names, adjust only that block — see the upstream examples:
  https://github.com/ServiceNow/AgentLab  ->  main.py / README
The rest (benchmark selection, study run, results dir) is stable.

Run via scripts/run_eval.sh (which sets the env + benchmark preconditions).
"""
from __future__ import annotations
import argparse
import os
import sys


# BrowserGym benchmark name per --bench. AgentLab/BrowserGym register these ids.
BENCH_TO_BROWSERGYM = {
    "miniwob": "miniwob",
    "webarena": "webarena",
    "webarena_lite": "webarena",   # WebArena-Lite = official 165-task subset; filtered via _filter_webarena_lite
    "workarena_l1": "workarena_l1",
    "workarena_l2": "workarena_l2",
}


def build_agent_args(model_name: str, base_url: str):
    """### AGENTLAB-VERSION-SENSITIVE — build a GenericAgent on an OpenAI-compatible endpoint."""
    # vLLM speaks the OpenAI API; make sure the OpenAI client points at it.
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))
    os.environ.setdefault("OPENAI_BASE_URL", base_url)
    os.environ.setdefault("OPENAI_API_BASE", base_url)

    from agentlab.agents.generic_agent import GenericAgentArgs
    from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o  # default flag preset
    try:
        # Newer AgentLab: an explicit OpenAI-compatible/self-hosted model-args class.
        from agentlab.llm.chat_api import OpenAIModelArgs as ChatModelArgs  # type: ignore
        chat_model_args = ChatModelArgs(
            model_name=model_name,
            max_total_tokens=32_000,
            max_input_tokens=30_000,
            max_new_tokens=2_000,
        )
    except Exception:  # fall back to the generic self-hosted args
        from agentlab.llm.chat_api import SelfHostedModelArgs as ChatModelArgs  # type: ignore
        chat_model_args = ChatModelArgs(
            model_name=model_name,
            base_url=base_url,
            max_total_tokens=32_000,
            max_input_tokens=30_000,
            max_new_tokens=2_000,
        )
    return GenericAgentArgs(chat_model_args=chat_model_args, flags=FLAGS_GPT_4o)


def _filter_webarena_lite(study) -> None:
    """Keep only the official WebArena-Lite task subset (165 ids).

    Reads one task id per line ('123' or 'webarena.123', '#' comments allowed) from
    $WEBARENA_LITE_TASKS or configs/webarena_lite_tasks.txt. Aborts when the list is
    missing or nothing matches — running full WebArena under a 'lite' label would
    silently misreport the SR.
    """
    list_path = os.environ.get("WEBARENA_LITE_TASKS", "configs/webarena_lite_tasks.txt")
    if not os.path.isfile(list_path):
        sys.exit(
            "[agentlab_eval] --bench webarena_lite needs the official 165-task id list.\n"
            f"  Put one task id per line in {list_path} (or set WEBARENA_LITE_TASKS=<file>);\n"
            "  the list ships with the WebArena-Lite release (e.g. THUDM/WebRL).\n"
            "  Use --bench webarena to run the full 812-task set instead."
        )
    wanted = set()
    with open(list_path) as fh:
        for line in fh:
            tid = line.split("#", 1)[0].strip()
            if tid:
                wanted.add(tid if "." in tid else f"webarena.{tid}")
    try:
        before = len(study.exp_args_list)
        study.exp_args_list = [
            ea for ea in study.exp_args_list
            if getattr(ea.env_args, "task_name", "") in wanted
        ]
    except Exception as e:
        sys.exit(f"[agentlab_eval] cannot filter tasks on this AgentLab version ({e}); "
                 "use --bench webarena or adapt the AGENTLAB-VERSION-SENSITIVE block.")
    if not study.exp_args_list:
        sys.exit(f"[agentlab_eval] webarena_lite: 0 of {before} tasks matched {list_path} — "
                 "check the id format (expected '123' or 'webarena.123').")
    print(f"[agentlab_eval] webarena_lite: kept {len(study.exp_args_list)}/{before} tasks")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(BENCH_TO_BROWSERGYM))
    ap.add_argument("--model-name", default=os.environ.get("VLLM_SERVED_NAME", "weasel"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out-root", default=os.environ.get("EVAL_RESULTS_ROOT", "./eval-results"))
    ap.add_argument("--limit", type=int, default=None, help="cap number of tasks (debug)")
    args = ap.parse_args()

    os.environ.setdefault("AGENTLAB_EXP_ROOT", args.out_root)

    try:
        from agentlab.experiments.study import make_study
    except Exception as e:  # pragma: no cover
        sys.exit(f"[agentlab_eval] cannot import AgentLab ({e}). Did you `bash scripts/install.sh eval`?")

    agent_args = build_agent_args(args.model_name, args.base_url)
    benchmark = BENCH_TO_BROWSERGYM[args.bench]

    print(f"[agentlab_eval] bench={args.bench} (browsergym='{benchmark}') "
          f"model={args.model_name} endpoint={args.base_url} n_jobs={args.n_jobs}")
    study = make_study(benchmark=benchmark, agent_args=[agent_args], comment=f"weasel-{args.bench}")

    if args.bench == "webarena_lite":
        _filter_webarena_lite(study)

    # Optional task cap for a smoke test (attribute name varies; best-effort).
    if args.limit is not None:
        try:
            study.exp_args_list = study.exp_args_list[: args.limit]
            print(f"[agentlab_eval] limited to {len(study.exp_args_list)} tasks")
        except Exception:
            print("[agentlab_eval][warn] could not apply --limit on this AgentLab version; running full set.")

    study.run(n_jobs=args.n_jobs)
    print(f"[agentlab_eval] done. results under {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
