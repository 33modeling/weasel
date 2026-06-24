"""WEASEL-style curation directly on raw function-calling trajectories.

Unlike the step-level paper pipeline (convert_gemini -> prepare_scores ->
select_greedy -> postprocess -> select_trajectories), this runs in ONE pass over
the ORIGINAL export and emits the ORIGINAL schema -- no format round-trip.

Designed for multi-turn FC data where **one jsonl line == one whole trajectory**
(e.g. the dit_* Gemini/GPT exports). Two signals, each at the granularity it fits:

  importance  [STEP level, paper-faithful, optional/GPU]
      r_t  = BERTScore(observation-history-up-to-t, goal)
      phi_t = max(0, r_t - r_{t-1})                 (marginal goal-relevance gain)
      trajectory quality = mean(phi)                (or final r_T with --quality final)
      Inputs are read straight from the raw messages (user = goal, tool =
      observation), so BERTScore stays on short step-level text -- its valid
      regime. No synthesized "## AXTree" markers, no conversion.

  dedup       [TRAJECTORY level, cheap/CPU]
      Each trajectory -> a short "fingerprint" (its action sequence + answer
      word-shingles). Group by task (--task-field, default __source_task__, else
      the goal text); within a group, greedily drop near-duplicate fingerprints
      (Jaccard >= --near-dup-threshold), keeping the highest-quality representative.
      O(N) read + O(group^2) cheap set ops -- no global all-pairs, no BERTScore here.

  select      keep the survivors (optionally only the top --keep-frac / --keep-k by
      quality per task), re-emitted in their ORIGINAL schema so they train directly
      with scripts/train_lora_sft.py.

Usage:
  # importance (GPU/bert_score) + near-dup dedup
  python -m weasel.select_clean --input export.jsonl --output clean.jsonl

  # dedup only -- no GPU, no bert_score (pure stdlib, fast):
  python -m weasel.select_clean --input export.jsonl --output clean.jsonl --no-importance

  # several source files, keep the top half by quality per task:
  python -m weasel.select_clean --input a.jsonl b.jsonl --output clean.jsonl --keep-frac 0.5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple


# ----------------------------------------------------------------------------
# Raw FC reading
# ----------------------------------------------------------------------------
def to_text(content: Any) -> str:
    """Normalize an OpenAI message 'content' (str | list-of-blocks | None) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def iter_records(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (record_index, record) from a JSONL or JSON-array file.

    record_index counts only parsed, non-blank records, so the same files re-read
    for output line up with the indices used during scoring."""
    with path.open("r", encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            for idx, rec in enumerate(json.load(fh)):
                yield idx, rec
            return
        idx = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[select_clean] skip corrupt line in {path.name}: {exc}", file=sys.stderr)
                continue
            yield idx, rec
            idx += 1


def first_user_text(messages: Sequence[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return to_text(m.get("content")).strip()
    return ""


def _clip(text: str, limit: int) -> str:
    return text if limit <= 0 or len(text) <= limit else text[:limit]


# ----------------------------------------------------------------------------
# Answer-language filter (script-ratio based, no deps)
# ----------------------------------------------------------------------------
_HANGUL = ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F))
_CJK = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))


def script_counts(text: str) -> Tuple[int, int]:
    """(hangul_chars, cjk_ideograph_chars) in text."""
    ko = cjk = 0
    for ch in text:
        o = ord(ch)
        if any(lo <= o <= hi for lo, hi in _HANGUL):
            ko += 1
        elif any(lo <= o <= hi for lo, hi in _CJK):
            cjk += 1
    return ko, cjk


def passes_answer_lang(answer: str, lang: str) -> bool:
    """True if `answer` is acceptable for the target-language filter.

    'ko' keeps Korean-dominant answers (drops Chinese-dominant); 'zh' the reverse.
    Pure-Latin / empty answers have neither script and always pass."""
    ko, cjk = script_counts(answer)
    if lang == "ko":
        return ko >= cjk
    if lang == "zh":
        return cjk >= ko
    return True


# ----------------------------------------------------------------------------
# Step extraction (importance) + fingerprint (dedup)
# ----------------------------------------------------------------------------
def _action_repr(msg: Dict[str, Any], arg_chars: int) -> str:
    """Compact 'tool(args)' string for one assistant action (empty if no tool_call)."""
    calls = msg.get("tool_calls") or []
    reprs: List[str] = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") if isinstance(c.get("function"), dict) else {}
        name = fn.get("name") or c.get("name") or "fn"
        args = fn.get("arguments", c.get("arguments"))
        if isinstance(args, str):
            arg_s = args
        else:
            try:
                arg_s = json.dumps(args, ensure_ascii=False, sort_keys=True)
            except Exception:
                arg_s = str(args)
        reprs.append(f"{name}({_clip(arg_s, arg_chars)})")
    return " ".join(reprs)


def walk_trajectory(record: Dict[str, Any], *, max_obs_chars: int, max_history_chars: int,
                    arg_chars: int) -> Tuple[str, List[str], List[str], str, int]:
    """Return (goal, obs_histories, action_tokens, final_answer, n_steps).

    obs_histories[t] = accumulated tool-observation text the agent had seen *before*
    action t (mirrors the paper's per-step obs_history). action_tokens feed the dedup
    fingerprint; final_answer is the last assistant text turn."""
    messages = record.get("messages") or []
    goal = first_user_text(messages)

    history: List[str] = []
    obs_histories: List[str] = []
    action_tokens: List[str] = []
    final_answer = ""

    for m in messages:
        role = m.get("role")
        if role == "tool":
            history.append(_clip(to_text(m.get("content")).strip(), max_obs_chars))
        elif role == "assistant":
            act = _action_repr(m, arg_chars)
            content = to_text(m.get("content")).strip()
            if act:                                   # tool-call action == an importance step
                obs_histories.append(_clip("\n\n".join(history), max_history_chars))
                action_tokens.append(act)
            if content:                               # text turn == (usually final) answer
                final_answer = content
    return goal, obs_histories, action_tokens, final_answer, len(action_tokens)


def fingerprint(action_tokens: Sequence[str], final_answer: str, *, answer_shingle: int) -> FrozenSet[str]:
    """A trajectory's dedup fingerprint: its action tokens + answer word-shingles."""
    tokens = set(action_tokens)
    words = re.findall(r"\w+", final_answer.lower())
    for i in range(len(words) - answer_shingle + 1):
        tokens.add("ans:" + " ".join(words[i:i + answer_shingle]))
    if not tokens:
        tokens.add("<empty>")
    return frozenset(tokens)


def jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


# ----------------------------------------------------------------------------
# Importance (BERTScore phi) -- optional, GPU
# ----------------------------------------------------------------------------
def phi_from_relevance(r_values: Sequence[float]) -> List[float]:
    phi, prev = [], 0.0
    for v in r_values:
        phi.append(max(0.0, v - prev))
        prev = v
    return phi


def load_bert_scorer(model_type: str, device: Optional[str]):
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        sys.exit(f"[select_clean] bert_score not installed ({exc}); "
                 "use --no-importance, or `pip install bert-score` in the select venv.")
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    return BERTScorer(model_type=model_type, rescale_with_baseline=False, lang="en", device=device)


def quality_from_steps(scorer, goals: Sequence[str], histories_per_traj: Sequence[Sequence[str]],
                       *, batch_size: int, quality: str) -> List[float]:
    """One quality score per trajectory from step-level BERTScore phi.

    Empty obs-history (every trajectory's first step, before any tool result) is scored
    r=0 directly instead of calling BERTScore on an empty string. Non-empty step pairs
    are flattened and scored in one batched pass."""
    candidates: List[str] = []
    references: List[str] = []
    plans: List[List[Optional[int]]] = []   # per traj: f1 index per step, or None (empty obs -> r=0)
    for goal, histories in zip(goals, histories_per_traj):
        plan: List[Optional[int]] = []
        for h in histories:
            if h:
                plan.append(len(candidates))
                candidates.append(h)
                references.append(goal)
            else:
                plan.append(None)
        plans.append(plan)

    f1_all: List[float] = []
    for s in range(0, len(candidates), batch_size):
        _, _, f1 = scorer.score(candidates[s:s + batch_size], references[s:s + batch_size],
                                batch_size=batch_size, verbose=False)
        f1_all.extend(float(x) for x in f1.cpu().tolist())

    out: List[float] = []
    for plan in plans:
        r = [0.0 if idx is None else f1_all[idx] for idx in plan]
        if not r:
            out.append(0.0)
        elif quality == "final":
            out.append(r[-1])
        else:  # mean phi
            phi = phi_from_relevance(r)
            out.append(sum(phi) / len(phi))
    return out


# ----------------------------------------------------------------------------
# Dedup + selection
# ----------------------------------------------------------------------------
def dedup_group(group: List[Dict[str, Any]], threshold: float,
                max_jaccard_group: int) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Greedily keep the highest-quality representative of each near-dup cluster.

    Returns (kept, jaccard_samples); each sample is a candidate's max Jaccard to the
    already-kept set (feeds the stats histogram). Empty for the exact-dedup fallback."""
    ordered = sorted(group, key=lambda m: m["quality"], reverse=True)
    if len(ordered) > max_jaccard_group:
        # Too large for O(n^2) Jaccard: fall back to exact-fingerprint dedup (O(n)).
        seen: set = set()
        kept = []
        for m in ordered:
            if m["fp"] in seen:
                continue
            seen.add(m["fp"])
            kept.append(m)
        return kept, []
    kept: List[Dict[str, Any]] = []
    samples: List[float] = []
    for m in ordered:
        maxj = max((jaccard(m["fp"], k["fp"]) for k in kept), default=0.0)
        if kept:
            samples.append(maxj)
        if maxj >= threshold:
            continue
        kept.append(m)
    return kept, samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="Raw FC export jsonl(s) (1 line == 1 trajectory). Concatenated in order.")
    ap.add_argument("--output", required=True, help="Selected trajectories (original schema), JSONL.")
    ap.add_argument("--no-importance", action="store_true",
                    help="Skip BERTScore phi (no GPU); quality falls back to trajectory length.")
    ap.add_argument("--quality", choices=["meanphi", "final"], default="meanphi",
                    help="Trajectory quality from step relevance: mean(phi) [default] or final r_T.")
    ap.add_argument("--near-dup-threshold", type=float, default=0.9,
                    help="Signature Jaccard >= this == near-duplicate (1.0 = exact). Default 0.9.")
    ap.add_argument("--keep-frac", type=float, default=None,
                    help="After dedup, keep this fraction of survivors per task (by quality).")
    ap.add_argument("--keep-k", type=int, default=None,
                    help="After dedup, keep at most this many survivors per task (by quality).")
    ap.add_argument("--min-steps", type=int, default=1, help="Drop trajectories with fewer actions.")
    ap.add_argument("--answer-lang", choices=["ko", "zh"], default=None,
                    help="Keep only trajectories whose final answer is dominant in this language "
                         "(drops the other CJK script; e.g. --answer-lang ko drops Chinese answers).")
    ap.add_argument("--task-field", default="__source_task__",
                    help="Record field that identifies the task; falls back to goal text if absent.")
    ap.add_argument("--model-type", default="roberta-large", help="BERTScore model (importance).")
    ap.add_argument("--batch-size", type=int, default=64, help="BERTScore batch size.")
    ap.add_argument("--score-chunk", type=int, default=64, help="Trajectories scored per GPU flush.")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto).")
    ap.add_argument("--max-obs-chars", type=int, default=4000)
    ap.add_argument("--max-history-chars", type=int, default=8000)
    ap.add_argument("--arg-chars", type=int, default=60, help="Per-action arg chars kept in the fingerprint.")
    ap.add_argument("--answer-shingle", type=int, default=5, help="Answer word-shingle size for the fingerprint.")
    ap.add_argument("--max-jaccard-group", type=int, default=3000,
                    help="Groups larger than this fall back to exact-fingerprint dedup.")
    args = ap.parse_args()

    if args.keep_frac is not None and args.keep_k is not None:
        ap.error("give at most one of --keep-frac / --keep-k")
    inputs = [Path(p) for p in args.input]
    for p in inputs:
        if not p.exists():
            sys.exit(f"[select_clean] not found: {p}")

    scorer = None if args.no_importance else load_bert_scorer(args.model_type, args.device)

    # --- Pass 1: extract per-trajectory quality + fingerprint (lightweight metadata only) ---
    metas: List[Dict[str, Any]] = []
    n_read = n_dropped_steps = n_dropped_lang = 0
    pending: List[Dict[str, Any]] = []   # buffered trajectories awaiting a GPU flush

    def flush_pending() -> None:
        if not pending:
            return
        if scorer is not None:
            quals = quality_from_steps(
                scorer, [p["goal"] for p in pending], [p["histories"] for p in pending],
                batch_size=args.batch_size, quality=args.quality,
            )
        else:
            quals = [float(p["n_steps"]) for p in pending]   # length proxy when importance is off
        for p, q in zip(pending, quals):
            metas.append({"gid": p["gid"], "task": p["task"], "fp": p["fp"], "quality": q})
        pending.clear()

    gid = 0
    for path in inputs:
        for _, rec in iter_records(path):
            this_gid = gid
            gid += 1
            goal, histories, actions, answer, n_steps = walk_trajectory(
                rec, max_obs_chars=args.max_obs_chars, max_history_chars=args.max_history_chars,
                arg_chars=args.arg_chars,
            )
            n_read += 1
            if n_steps < args.min_steps:
                n_dropped_steps += 1
                continue
            if args.answer_lang and not passes_answer_lang(answer, args.answer_lang):
                n_dropped_lang += 1
                continue
            task = str(rec.get(args.task_field) or goal or "<no_task>")
            fp = fingerprint(actions, answer, answer_shingle=args.answer_shingle)
            pending.append({"gid": this_gid, "task": task, "fp": fp, "goal": goal,
                            "histories": histories, "n_steps": n_steps})
            if len(pending) >= args.score_chunk:
                flush_pending()
            if n_read % 2000 == 0:
                print(f"[select_clean] scored {n_read} trajectories...", file=sys.stderr)
    flush_pending()

    # --- Dedup + per-task selection ---
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in metas:
        by_task[m["task"]].append(m)

    keep_gids: set = set()
    n_after_dedup = 0
    jaccard_samples: List[float] = []
    for task, group in by_task.items():
        survivors, samples = dedup_group(group, args.near_dup_threshold, args.max_jaccard_group)
        jaccard_samples.extend(samples)
        n_after_dedup += len(survivors)
        survivors.sort(key=lambda m: m["quality"], reverse=True)
        if args.keep_k is not None:
            survivors = survivors[: args.keep_k]
        elif args.keep_frac is not None:
            keep_n = max(1, round(len(survivors) * args.keep_frac))
            survivors = survivors[:keep_n]
        keep_gids.update(m["gid"] for m in survivors)

    # --- Pass 2: re-emit selected records in their ORIGINAL schema ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    gid = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in inputs:
            for _, rec in iter_records(path):
                if gid in keep_gids:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                gid += 1

    # --- Report ---
    rollout_sizes = sorted((len(g) for g in by_task.values()), reverse=True)
    print("\n=== select_clean ===")
    print(f"  read trajectories         : {n_read}")
    print(f"  dropped (< {args.min_steps} step)        : {n_dropped_steps}")
    if args.answer_lang:
        print(f"  dropped (answer != {args.answer_lang})    : {n_dropped_lang}")
    print(f"  near-duplicates removed   : {n_read - n_dropped_steps - n_dropped_lang - n_after_dedup}")
    print(f"  after dedup               : {n_after_dedup}")
    print(f"  selected (written)        : {written}  -> {out_path}")
    print(f"  importance                : {'OFF (length proxy)' if scorer is None else args.quality}")
    print(f"  near-dup threshold        : {args.near_dup_threshold}")
    if args.keep_frac is not None:
        print(f"  per-task keep-frac        : {args.keep_frac}")
    if args.keep_k is not None:
        print(f"  per-task keep-k           : {args.keep_k}")

    # tasks / rollouts-per-task (shows how much cross-rollout redundancy exists)
    if rollout_sizes:
        med = rollout_sizes[len(rollout_sizes) // 2]
        buckets = {"1": 0, "2-5": 0, "6-20": 0, "21+": 0}
        for s in rollout_sizes:
            key = "1" if s == 1 else "2-5" if s <= 5 else "6-20" if s <= 20 else "21+"
            buckets[key] += 1
        print(f"  tasks                     : {len(rollout_sizes)}  "
              f"(rollouts/task: max {rollout_sizes[0]}, median {med})")
        print("  rollouts/task histogram   : " + "  ".join(f"{k}={v}" for k, v in buckets.items()))

    # Jaccard distribution of dedup decisions (to calibrate --near-dup-threshold)
    if jaccard_samples:
        counts = [0] * 10
        for v in jaccard_samples:
            counts[min(9, int(v * 10))] += 1
        mx = max(counts) or 1
        thr_bucket = min(9, int(args.near_dup_threshold * 10))
        print(f"  Jaccard dist (n={len(jaccard_samples)}; bars = dedup decisions):")
        for i in range(10):
            bar = "#" * round(counts[i] / mx * 26)
            mark = "  <-- threshold here" if i == thr_bucket else ""
            print(f"    {i/10:.1f}-{(i+1)/10:.1f}  {counts[i]:>6}  {bar}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
