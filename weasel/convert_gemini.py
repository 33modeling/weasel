"""Convert Gemini-style function-calling trajectories into WEASEL-ingestible data.

The Gemini export (train_data/*.jsonl) stores ONE full multi-turn trajectory per
line in the OpenAI function-calling schema:

    {"tools": [ {"type":"function","function":{...}}, ... ],
     "messages": [ {"role","content","reasoning_content","tool_calls","tool_call_id"}, ... ]}

WEASEL was built for *web-agent* per-step records whose user prompt embeds
`## Goal:`, `## AXTree:`, `# Observation of current step: ... # Action space:`
markers, grouped into trajectories by goal, with the action in the assistant
*text*. This converter bridges the two, in two modes:

  --mode step   (a) PAPER-FAITHFUL.  Explode each trajectory into one record per
                assistant action. Every step of a line shares the same `## Goal:`
                so prepare_scores groups them into one trajectory and select_greedy
                picks the t0 best steps — exactly the paper recipe, with the
                environment observation standing in for the AXTree. Output is plain
                ShareGPT (action serialized into assistant text), so it flows
                through the EXISTING weasel_agenttrek train path unchanged.

  --mode traj   (b) REAL USE.  Keep each line as one native function-calling
                ShareGPT record (tool_calls preserved as function_call/observation
                turns + a tools column) so training keeps the true agent signal.
                Each record carries `_traj_id`; after running WEASEL on the step
                data, weasel.select_trajectories maps the selected steps back to
                their trajectories to subset this file.

  --mode both   write both outputs from a single pass (shared _traj_id).

Both outputs stream line-by-line, so the multi-GB export never loads fully into RAM.

Examples
--------
  python -m weasel.convert_gemini \
    --input train_data/dit_task_0513_gemini_per_line.jsonl \
    --mode both \
    --steps-output data/gemini_steps.jsonl \
    --traj-output  data/gemini_traj.jsonl \
    --stats-output data/gemini_convert_stats.json

Then, paper-faithful selection (mode a):
  TRAIN_INPUT_JSON=data/gemini_steps.jsonl bash scripts/run_select.sh --gpus 0
  bash scripts/prepare_dataset.sh        # registers as weasel_agenttrek
  bash scripts/run_train.sh --gpus 0

Real-use trajectory training (mode b), optionally WEASEL-filtered:
  python -m weasel.select_trajectories \
    --selected-dataset data/weasel_agenttrek_train_10k.json \
    --traj-dataset data/gemini_traj.jsonl \
    --output data/gemini_traj_selected.jsonl
  cp data/gemini_traj_selected.jsonl $LLAMAFACTORY_DIR/data/weasel_gemini_traj.json
  DATASET_NAME=weasel_gemini_traj bash scripts/run_train.sh --gpus 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def to_text(content: Any) -> str:
    """Normalize a message `content` (str | None | list-of-parts | dict) to text.

    GPT/Gemini exports mix plain strings, null (assistant + tool_calls), and
    OpenAI structured content blocks like [{"type":"text","text":...}, ...]."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for el in content:
            if isinstance(el, str):
                parts.append(el)
            elif isinstance(el, dict):
                parts.append(el.get("text") or el.get("content") or "")
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    return str(content)


# --------------------------------------------------------------------------- io
def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (line_index, record). Tolerates a JSON-array file too."""
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":  # whole-file JSON array fallback
            for idx, rec in enumerate(json.load(f)):
                yield idx, rec
            return
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError as e:  # skip a corrupt line, keep going
                print(f"[convert_gemini] skip line {idx}: {e}", file=sys.stderr)


# ------------------------------------------------------------------ serializing
def _args_to_text(arguments: Any) -> str:
    """Compact one tool-call's arguments to a stable string."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments.strip()
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)


def _args_to_obj(arguments: Any) -> Any:
    """Parse a tool-call's arguments into an object for native FC output."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def serialize_action(msg: Dict[str, Any]) -> str:
    """Render an assistant message (text and/or tool_calls) as WEASEL action text."""
    parts: List[str] = []
    reasoning = to_text(msg.get("reasoning_content")).strip()
    if reasoning:
        parts.append(f"<think>{reasoning}</think>")
    text = to_text(msg.get("content")).strip()
    if text:
        parts.append(text)
    calls = msg.get("tool_calls") or []
    rendered = []
    for tc in calls:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name", "unknown")
        rendered.append(f"{name}({_args_to_text(fn.get('arguments', {}))})")
    if rendered:
        parts.append("<action>" + "\n".join(rendered) + "</action>")
    return "\n".join(parts).strip()


def first_role(messages: List[Dict[str, Any]], role: str) -> str:
    for m in messages:
        if m.get("role") == role:
            return to_text(m.get("content"))
    return ""


def tool_names(record: Dict[str, Any]) -> str:
    names = []
    for t in record.get("tools") or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        if fn.get("name"):
            names.append(fn["name"])
    return ", ".join(names)


def clip(text: str, limit: int) -> str:
    if limit and limit > 0 and len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
    return text


# Placeholder so an empty observation keeps the section non-blank — otherwise the
# `\s*` in prepare_scores' AXTree/Observation regexes swallows the boundary
# newline and the next marker leaks into the captured field.
EMPTY_OBS = "(no observation)"


def source_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    """Carry provenance fields (e.g. __source_task__/__source_agent__) if present."""
    meta = {}
    if record.get("__source_task__") is not None:
        meta["_source_task"] = record["__source_task__"]
    if record.get("__source_agent__") is not None:
        meta["_source_agent"] = record["__source_agent__"]
    return meta


# ----------------------------------------------------------------- mode: step
def build_steps(
    record: Dict[str, Any],
    traj_id: int,
    *,
    unique_goal: bool,
    max_obs_chars: int,
    max_history_chars: int,
    max_system_chars: int,
) -> List[Dict[str, Any]]:
    """Explode one trajectory into per-action ShareGPT step records."""
    messages = record.get("messages") or []
    goal = (first_role(messages, "user") or "").strip()
    if not goal:
        return []
    goal_line = f"{goal} (traj#{traj_id})" if unique_goal else goal
    system = clip((first_role(messages, "system") or "").strip(), max_system_chars)
    actions = tool_names(record)

    steps: List[Dict[str, Any]] = []
    history: List[str] = []          # all prior observations (running history)
    last_obs = ""                    # most recent observation = current "state"
    step_no = 0

    for m in messages:
        role = m.get("role")
        if role == "tool":
            obs = to_text(m.get("content")).strip()
            last_obs = obs
            history.append(obs)
            continue
        if role != "assistant":
            continue
        action_text = serialize_action(m)
        if not action_text:
            continue

        state = clip(last_obs, max_obs_chars) or EMPTY_OBS
        obs_history = clip("\n\n".join(history), max_history_chars) or EMPTY_OBS
        user_content = (
            f"## Goal: {goal_line}\n\n"
            f"## AXTree:\n{state}\n\n"
            f"# Observation of current step:\n{obs_history}\n"
            f"# Action space:\n{actions}"
        )
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant", "content": action_text})
        rec = {"messages": msgs, "_traj_id": traj_id, "_step": step_no}
        rec.update(source_meta(record))
        steps.append(rec)
        step_no += 1

    return steps


# ----------------------------------------------------------------- mode: traj
def build_traj(record: Dict[str, Any], traj_id: int, *, max_obs_chars: int) -> Optional[Dict[str, Any]]:
    """Convert one trajectory to native function-calling ShareGPT for LLaMA-Factory."""
    messages = record.get("messages") or []
    out_msgs: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        text = to_text(m.get("content"))
        if role == "system":
            out_msgs.append({"role": "system", "content": text})
        elif role == "user":
            out_msgs.append({"role": "user", "content": text})
        elif role == "tool":
            out_msgs.append({"role": "observation", "content": clip(text, max_obs_chars)})
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                payload = [
                    {
                        "name": (tc.get("function", {}) or {}).get("name", "unknown"),
                        "arguments": _args_to_obj((tc.get("function", {}) or {}).get("arguments", {})),
                    }
                    for tc in calls
                ]
                # LLaMA-Factory accepts a single object or a list of calls.
                fc = payload[0] if len(payload) == 1 else payload
                out_msgs.append({"role": "function_call", "content": json.dumps(fc, ensure_ascii=False)})
            elif text.strip():
                out_msgs.append({"role": "assistant", "content": text})
    if not any(m["role"] == "user" for m in out_msgs):
        return None
    # Drop prompt-only trajectories (no assistant/function_call target to train on).
    if not any(m["role"] in ("assistant", "function_call") for m in out_msgs):
        return None
    tools = record.get("tools") or []
    out = {
        "messages": out_msgs,
        "tools": json.dumps(tools, ensure_ascii=False),
        "_traj_id": traj_id,
    }
    out.update(source_meta(record))
    return out


# ----------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, nargs="+",
                    help="One or more Gemini/GPT-style JSONL files (one trajectory per line). "
                         "Multiple files are concatenated into the outputs with globally unique _traj_id.")
    ap.add_argument("--mode", choices=["step", "traj", "both"], default="both")
    ap.add_argument("--steps-output", default=None, help="JSONL output for --mode step/both.")
    ap.add_argument("--traj-output", default=None, help="JSONL output for --mode traj/both.")
    ap.add_argument("--unique-goal", action="store_true",
                    help="Append '(traj#<id>)' to each step's goal so every input line is its own "
                         "trajectory group (default: group by raw task text, paper-style).")
    ap.add_argument("--max-obs-chars", type=int, default=4000,
                    help="Per-observation cap for the step state/AXTree (0 = no cap).")
    ap.add_argument("--max-history-chars", type=int, default=8000,
                    help="Cap for the running observation history in step records (0 = no cap).")
    ap.add_argument("--max-system-chars", type=int, default=0,
                    help="Cap for the system prompt copied into step records (0 = no cap).")
    ap.add_argument("--traj-max-obs-chars", type=int, default=0,
                    help="Per-observation cap in native-FC traj output (0 = keep full, recommended).")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N lines (debug).")
    ap.add_argument("--stats-output", default=None, help="Optional JSON with conversion counts.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    in_paths = [Path(p) for p in args.input]
    for p in in_paths:
        if not p.exists():
            sys.exit(f"[convert_gemini] input not found: {p}")

    want_step = args.mode in ("step", "both")
    want_traj = args.mode in ("traj", "both")
    if want_step and not args.steps_output:
        sys.exit("[convert_gemini] --steps-output required for mode step/both")
    if want_traj and not args.traj_output:
        sys.exit("[convert_gemini] --traj-output required for mode traj/both")

    steps_f = traj_f = None
    if want_step:
        Path(args.steps_output).parent.mkdir(parents=True, exist_ok=True)
        steps_f = open(args.steps_output, "w", encoding="utf-8")
    if want_traj:
        Path(args.traj_output).parent.mkdir(parents=True, exist_ok=True)
        traj_f = open(args.traj_output, "w", encoding="utf-8")

    gid = 0  # global trajectory id, unique across all input files
    n_traj = n_steps = n_skipped = n_traj_written = 0
    per_file = []
    try:
        for in_path in in_paths:
            f_read = f_steps = f_traj = 0
            for local_idx, rec in iter_jsonl(in_path):
                if args.limit is not None and local_idx >= args.limit:
                    break
                tid = gid
                gid += 1
                n_traj += 1
                f_read += 1
                if want_step:
                    steps = build_steps(
                        rec, tid,
                        unique_goal=args.unique_goal,
                        max_obs_chars=args.max_obs_chars,
                        max_history_chars=args.max_history_chars,
                        max_system_chars=args.max_system_chars,
                    )
                    if not steps:
                        n_skipped += 1
                    for s in steps:
                        steps_f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    n_steps += len(steps)
                    f_steps += len(steps)
                if want_traj:
                    t = build_traj(rec, tid, max_obs_chars=args.traj_max_obs_chars)
                    if t is None:
                        if not want_step:
                            n_skipped += 1
                    else:
                        traj_f.write(json.dumps(t, ensure_ascii=False) + "\n")
                        n_traj_written += 1
                        f_traj += 1
                if n_traj % 1000 == 0:
                    print(f"[convert_gemini] {n_traj} trajectories...", file=sys.stderr)
            per_file.append({"input": str(in_path), "trajectories_read": f_read,
                             "step_records": f_steps if want_step else None,
                             "traj_records": f_traj if want_traj else None})
            print(f"[convert_gemini] done {in_path.name}: read={f_read} "
                  f"steps={f_steps if want_step else '-'} traj={f_traj if want_traj else '-'}",
                  file=sys.stderr)
    finally:
        if steps_f:
            steps_f.close()
        if traj_f:
            traj_f.close()

    stats = {
        "inputs": [str(p) for p in in_paths],
        "per_file": per_file,
        "mode": args.mode,
        "trajectories_read": n_traj,
        "step_records_written": n_steps if want_step else None,
        "traj_records_written": n_traj_written if want_traj else None,
        "skipped_empty": n_skipped,
        "steps_output": args.steps_output if want_step else None,
        "traj_output": args.traj_output if want_traj else None,
    }
    print("[convert_gemini] " + json.dumps(stats, ensure_ascii=False))
    if args.stats_output:
        Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_output).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"[convert_gemini] wrote {args.stats_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
