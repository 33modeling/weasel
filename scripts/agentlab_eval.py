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
    "webarena_lite": "webarena",   # WebArena-Lite = the 165-task subset; filtered below if supported
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
