#!/usr/bin/env python3
"""Generate a self-contained HTML trajectory-analysis report from an
AgentLab/BrowserGym study (MiniWob++ / WebArena / WorkArena).

What it shows: overall success rate, per-task breakdown, step-count
distribution (success vs failure), action-type frequency, a failure-mode
breakdown (agent/env error / out-of-steps / wrong-answer / action-execution
errors), token & time stats when present, and a per-episode drill-down of the
actual action sequence (with the agent's `think`, reward, and action errors).

Trajectory detail needs the EVAL venv (the per-step files are gzipped pickles of
AgentLab `StepInfo` objects). Run it there — `scripts/run_eval.sh` does this
automatically after each eval. Without agentlab importable, it falls back to
`summary_info.json`-only stats (task-level success/steps/errors, no actions) and
says so in the report.

  python scripts/miniwob_report.py                                # newest study under EVAL_RESULTS_ROOT
  python scripts/miniwob_report.py --root eval-results/weasel     # newest study under a root
  python scripts/miniwob_report.py --study-dir <study> --out report.html
"""
from __future__ import annotations
import argparse
import gzip
import html
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ACTION_FN_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")


# --------------------------------------------------------------------------
# Locate the study directory
# --------------------------------------------------------------------------
def newest_study(root: Path) -> Path | None:
    """Newest dir under `root` containing a summary_info.json (root itself counts)."""
    cands = [d for d in (root.iterdir() if root.is_dir() else [])
             if d.is_dir() and any(d.rglob("summary_info.json"))]
    if cands:
        return max(cands, key=lambda p: p.stat().st_mtime)
    return root if (root.is_dir() and any(root.rglob("summary_info.json"))) else None


# --------------------------------------------------------------------------
# Episode extraction
# --------------------------------------------------------------------------
def _action_type(action: str | None) -> str:
    if not action:
        return "(none)"
    m = ACTION_FN_RE.match(action)
    return m.group(1) if m else "(other)"


def _think_of(agent_info) -> str:
    if not isinstance(agent_info, dict):
        return ""
    for k in ("think", "thought", "reasoning"):
        v = agent_info.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _exp_dirs(study_dir: Path):
    """All experiment dirs (those that hold a summary_info.json)."""
    return sorted({p.parent for p in study_dir.rglob("summary_info.json")})


def _episode_from_dir_stdlib(exp_dir: Path) -> dict | None:
    """summary_info.json-only episode (no per-step actions)."""
    sfile = exp_dir / "summary_info.json"
    try:
        info = json.loads(sfile.read_text())
    except Exception:
        return None
    reward = info.get("cum_reward")
    return {
        "task": _task_name_from_dir(exp_dir, None),
        "dir": exp_dir.name,
        "reward": reward,
        "success": (reward is not None) and float(reward) > 0,
        "n_steps": info.get("n_steps"),
        "terminated": bool(info.get("terminated")),
        "truncated": bool(info.get("truncated")),
        "err_msg": (info.get("err_msg") or "").strip(),
        "stats": {k: v for k, v in info.items() if k.startswith("stats.")},
        "steps": [],          # no trajectory in the stdlib path
        "has_steps": False,
    }


TASK_RE = re.compile(r"(?:miniwob|webarena|workarena)[.\w-]+")


def _task_name_from_dir(exp_dir: Path, exp_args) -> str:
    t = getattr(getattr(exp_args, "env_args", None), "task_name", None)
    if t:
        return str(t)
    # fall back to exp_args.pkl, then the dir name
    pkl = exp_dir / "exp_args.pkl"
    if pkl.exists():
        try:
            with open(pkl, "rb") as fh:
                ea = pickle.load(fh)
            t = getattr(getattr(ea, "env_args", None), "task_name", None)
            if t:
                return str(t)
        except Exception:
            pass
    m = TASK_RE.search(exp_dir.name)
    return m.group(0) if m else exp_dir.name


def extract_episodes(study_dir: Path):
    """Return (episodes, mode) where mode is 'full' (with actions) or 'summary'."""
    try:
        from agentlab.experiments.loop import yield_all_exp_results  # type: ignore
    except Exception:
        # No agentlab -> summary_info.json only.
        eps = [e for e in (_episode_from_dir_stdlib(d) for d in _exp_dirs(study_dir)) if e]
        return eps, "summary"

    episodes = []
    for res in yield_all_exp_results(str(study_dir), progress_fn=None):
        try:
            info = res.summary_info or {}
        except Exception:
            info = {}
        exp_dir = Path(getattr(res, "exp_dir", study_dir))
        try:
            exp_args = res.exp_args
        except Exception:
            exp_args = None
        reward = info.get("cum_reward")
        ep = {
            "task": _task_name_from_dir(exp_dir, exp_args),
            "dir": exp_dir.name,
            "reward": reward,
            "success": (reward is not None) and float(reward) > 0,
            "n_steps": info.get("n_steps"),
            "terminated": bool(info.get("terminated")),
            "truncated": bool(info.get("truncated")),
            "err_msg": (info.get("err_msg") or "").strip(),
            "stats": {k: v for k, v in info.items() if k.startswith("stats.")},
            "steps": [],
            "has_steps": False,
        }
        try:
            steps = res.steps_info or []
            for st in steps:
                action = getattr(st, "action", None)
                obs = getattr(st, "obs", None) or {}
                ainfo = getattr(st, "agent_info", None) or {}
                ep["steps"].append({
                    "i": getattr(st, "step", len(ep["steps"])),
                    "action": action,
                    "type": _action_type(action),
                    "think": _think_of(ainfo),
                    "reward": getattr(st, "reward", None),
                    "action_error": (obs.get("last_action_error") or "").strip()
                                    if isinstance(obs, dict) else "",
                    "axtree_chars": len(obs.get("axtree_txt", "")) if isinstance(obs, dict) else 0,
                })
            ep["has_steps"] = bool(ep["steps"])
            # n_steps fallback from the actual step count
            if ep["n_steps"] is None and ep["steps"]:
                ep["n_steps"] = sum(1 for s in ep["steps"] if s["action"] is not None)
        except Exception as e:
            ep["steps_error"] = str(e)
        episodes.append(ep)
    mode = "full" if any(e["has_steps"] for e in episodes) else "summary"
    return episodes, mode


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def compute_stats(episodes) -> dict:
    n = len(episodes)
    succ = [e for e in episodes if e["success"]]
    fail = [e for e in episodes if not e["success"]]

    # per-task
    by_task = defaultdict(list)
    for e in episodes:
        by_task[e["task"]].append(e)
    task_rows = []
    for task, eps in by_task.items():
        ns = len(eps)
        ok = sum(1 for e in eps if e["success"])
        task_rows.append({
            "task": task, "n": ns, "ok": ok,
            "sr": ok / ns if ns else 0.0,
            "mean_reward": _mean([e["reward"] for e in eps]),
            "mean_steps": _mean([e["n_steps"] for e in eps]),
        })
    task_rows.sort(key=lambda r: (r["sr"], -r["n"]))   # worst first

    # action frequency (full mode only)
    action_freq = Counter()
    action_err = 0
    for e in episodes:
        for s in e["steps"]:
            if s["action"] is not None:
                action_freq[s["type"]] += 1
            if s["action_error"]:
                action_err += 1

    # failure-mode buckets
    buckets = Counter()
    for e in fail:
        if e["err_msg"]:
            buckets["agent/env error (err_msg)"] += 1
        elif e["truncated"]:
            buckets["out of steps (truncated)"] += 1
        elif e["terminated"]:
            buckets["finished, no reward (wrong/gave up)"] += 1
        else:
            buckets["incomplete / unknown"] += 1

    # token & time stats: scan stats.* keys for token / time-ish numbers
    tok_keys, time_keys = Counter(), Counter()
    tok_sum, time_sum = defaultdict(float), defaultdict(float)
    for e in episodes:
        for k, v in e["stats"].items():
            if not isinstance(v, (int, float)):
                continue
            kl = k.lower()
            if "token" in kl:
                tok_keys[k] += 1; tok_sum[k] += v
            elif "time" in kl or "elapsed" in kl or "duration" in kl:
                time_keys[k] += 1; time_sum[k] += v

    return {
        "n": n,
        "n_success": len(succ),
        "sr": len(succ) / n if n else 0.0,
        "mean_reward": _mean([e["reward"] for e in episodes]),
        "mean_steps_all": _mean([e["n_steps"] for e in episodes]),
        "mean_steps_succ": _mean([e["n_steps"] for e in succ]),
        "mean_steps_fail": _mean([e["n_steps"] for e in fail]),
        "median_steps_all": _median([e["n_steps"] for e in episodes]),
        "task_rows": task_rows,
        "action_freq": action_freq.most_common(),
        "total_actions": sum(action_freq.values()),
        "action_errors": action_err,
        "failure_buckets": buckets.most_common(),
        "n_fail": len(fail),
        "tok_avg": {k: tok_sum[k] / tok_keys[k] for k in tok_keys},
        "time_avg": {k: time_sum[k] / time_keys[k] for k in time_keys},
    }


# --------------------------------------------------------------------------
# HTML rendering (self-contained: inline CSS + a little vanilla JS)
# --------------------------------------------------------------------------
def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _fmt(x, pct=False, nd=1):
    if x is None:
        return "—"
    if pct:
        return f"{x * 100:.1f}%"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _bar(frac, bad=False, w=160):
    frac = max(0.0, min(1.0, frac))
    cls = "bar bad" if bad else "bar"
    return (f'<span class="{cls}"><span class="barfill" style="width:{frac * w:.0f}px">'
            f'</span></span>')


def render_html(stats, episodes, meta) -> str:
    css = """
    body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
      margin:0;background:#0d1117;color:#c9d1d9}
    .wrap{max-width:1100px;margin:0 auto;padding:24px}
    h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 10px;
      border-bottom:1px solid #21262d;padding-bottom:6px}
    .sub{color:#8b949e;font-size:13px;margin-bottom:18px}
    .cards{display:flex;flex-wrap:wrap;gap:12px}
    .card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px 18px;min-width:120px}
    .card .v{font-size:24px;font-weight:600} .card .l{color:#8b949e;font-size:12px}
    .card.good .v{color:#3fb950} .card.bad .v{color:#f85149} .card.warn .v{color:#d29922}
    table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}
    th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #21262d}
    th{color:#8b949e;font-weight:600} tr:hover td{background:#161b22}
    td.num{text-align:right;font-variant-numeric:tabular-nums}
    .bar{display:inline-block;width:160px;height:9px;background:#21262d;border-radius:5px;vertical-align:middle}
    .barfill{display:inline-block;height:9px;background:#3fb950;border-radius:5px}
    .bad .barfill{background:#f85149}
    details{background:#161b22;border:1px solid #21262d;border-radius:8px;margin:8px 0;padding:4px 12px}
    details summary{cursor:pointer;font-weight:600;padding:6px 0}
    summary .ok{color:#3fb950} summary .fa{color:#f85149}
    .steps{font-size:12px;margin:6px 0}
    .steps td{vertical-align:top;border-bottom:1px solid #1b2026}
    .act{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#79c0ff;white-space:pre-wrap}
    .think{color:#8b949e;white-space:pre-wrap}
    .err{color:#f85149;white-space:pre-wrap}
    .pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;background:#21262d}
    .note{background:#1c2128;border-left:3px solid #d29922;padding:10px 14px;border-radius:4px;color:#d29922;margin:12px 0}
    code{background:#21262d;padding:1px 5px;border-radius:4px}
    """
    js = """
    function filterEp(){var q=document.getElementById('q').value.toLowerCase();
      document.querySelectorAll('details.ep').forEach(function(d){
        d.style.display=d.dataset.k.indexOf(q)>=0?'':'none';});}
    """
    out = []
    A = out.append
    A(f"<!doctype html><html><head><meta charset='utf-8'><title>{_esc(meta['title'])}</title>")
    A(f"<style>{css}</style></head><body><div class='wrap'>")
    A(f"<h1>{_esc(meta['title'])}</h1>")
    A(f"<div class='sub'>study: <code>{_esc(meta['study'])}</code> &nbsp;·&nbsp; "
      f"variant: <b>{_esc(meta['variant'])}</b> &nbsp;·&nbsp; mode: <b>{_esc(meta['mode'])}</b>"
      f" &nbsp;·&nbsp; generated {_esc(meta['ts'])}</div>")

    if meta["mode"] == "summary":
        A("<div class='note'>Trajectory detail unavailable — agentlab is not importable "
          "in this environment, so only <code>summary_info.json</code> task-level stats are "
          "shown (no per-action breakdown). Run this in the <b>eval</b> venv for full "
          "trajectories.</div>")

    # ---- summary cards ----
    sr = stats["sr"]
    A("<div class='cards'>")
    A(f"<div class='card'><div class='v'>{stats['n']}</div><div class='l'>episodes</div></div>")
    A(f"<div class='card {'good' if sr>=0.5 else 'bad'}'><div class='v'>{_fmt(sr,pct=True)}</div>"
      f"<div class='l'>success rate ({stats['n_success']}/{stats['n']})</div></div>")
    A(f"<div class='card'><div class='v'>{_fmt(stats['mean_reward'],nd=3)}</div><div class='l'>mean reward</div></div>")
    A(f"<div class='card'><div class='v'>{_fmt(stats['mean_steps_all'])}</div><div class='l'>mean steps</div></div>")
    if stats["total_actions"]:
        A(f"<div class='card {'warn' if stats['action_errors'] else ''}'><div class='v'>{stats['action_errors']}</div>"
          f"<div class='l'>action-exec errors</div></div>")
    A("</div>")

    # ---- step distribution ----
    A("<h2>Steps: success vs failure</h2><table>")
    A("<tr><th>group</th><th class='num'>episodes</th><th class='num'>mean steps</th></tr>")
    A(f"<tr><td><span class='ok'>success</span></td><td class='num'>{stats['n_success']}</td>"
      f"<td class='num'>{_fmt(stats['mean_steps_succ'])}</td></tr>")
    A(f"<tr><td><span class='fa'>failure</span></td><td class='num'>{stats['n_fail']}</td>"
      f"<td class='num'>{_fmt(stats['mean_steps_fail'])}</td></tr>")
    A(f"<tr><td>all (median {_fmt(stats['median_steps_all'])})</td><td class='num'>{stats['n']}</td>"
      f"<td class='num'>{_fmt(stats['mean_steps_all'])}</td></tr>")
    A("</table>")

    # ---- failure modes ----
    if stats["failure_buckets"]:
        A("<h2>Failure modes</h2><table>")
        A("<tr><th>reason</th><th class='num'>count</th><th>share of failures</th></tr>")
        for reason, c in stats["failure_buckets"]:
            frac = c / stats["n_fail"] if stats["n_fail"] else 0
            A(f"<tr><td>{_esc(reason)}</td><td class='num'>{c}</td>"
              f"<td>{_bar(frac, bad=True)}</td></tr>")
        A("</table>")

    # ---- action frequency ----
    if stats["action_freq"]:
        mx = stats["action_freq"][0][1]
        A(f"<h2>Action types <span class='pill'>{stats['total_actions']} actions</span></h2><table>")
        A("<tr><th>action</th><th class='num'>count</th><th>frequency</th></tr>")
        for name, c in stats["action_freq"]:
            A(f"<tr><td class='act'>{_esc(name)}</td><td class='num'>{c}</td>"
              f"<td>{_bar(c / mx)}</td></tr>")
        A("</table>")

    # ---- per-task ----
    A("<h2>Per-task success rate <span class='pill'>worst first</span></h2><table>")
    A("<tr><th>task</th><th class='num'>n</th><th class='num'>SR</th><th></th>"
      "<th class='num'>mean reward</th><th class='num'>mean steps</th></tr>")
    for r in stats["task_rows"]:
        A(f"<tr><td class='act'>{_esc(r['task'])}</td><td class='num'>{r['n']}</td>"
          f"<td class='num'>{_fmt(r['sr'],pct=True)}</td>"
          f"<td>{_bar(r['sr'], bad=r['sr'] < 0.5)}</td>"
          f"<td class='num'>{_fmt(r['mean_reward'],nd=3)}</td>"
          f"<td class='num'>{_fmt(r['mean_steps'])}</td></tr>")
    A("</table>")

    # ---- token / time ----
    if stats["tok_avg"] or stats["time_avg"]:
        A("<h2>Cost signals (avg per episode)</h2><table><tr><th>metric</th><th class='num'>avg</th></tr>")
        for k, v in sorted(stats["tok_avg"].items()):
            A(f"<tr><td>{_esc(k)}</td><td class='num'>{_fmt(v)}</td></tr>")
        for k, v in sorted(stats["time_avg"].items()):
            A(f"<tr><td>{_esc(k)}</td><td class='num'>{_fmt(v,nd=2)}</td></tr>")
        A("</table>")

    # ---- per-episode drill-down (full mode) ----
    has_traj = any(e["has_steps"] for e in episodes)
    if has_traj:
        A("<h2>Trajectories</h2>")
        A("<input id='q' oninput='filterEp()' placeholder='filter by task / success / fail…' "
          "style='width:100%;padding:8px;background:#0d1117;border:1px solid #21262d;"
          "color:#c9d1d9;border-radius:6px;margin-bottom:8px'>")
        # show failures first, then successes; cap to keep the file sane
        ordered = sorted(episodes, key=lambda e: (e["success"], e["task"]))
        cap = meta["max_traj"]
        for e in ordered[:cap]:
            tag = "<span class='ok'>✓</span>" if e["success"] else "<span class='fa'>✗</span>"
            key = f"{e['task']} {'success' if e['success'] else 'fail'} {e['dir']}".lower()
            A(f"<details class='ep' data-k='{_esc(key)}'>")
            A(f"<summary>{tag} <span class='act'>{_esc(e['task'])}</span> "
              f"<span class='pill'>r={_fmt(e['reward'],nd=2)}</span> "
              f"<span class='pill'>{_fmt(e['n_steps'])} steps</span>"
              + (f" <span class='pill'>truncated</span>" if e['truncated'] else "")
              + (f" <span class='err'>err</span>" if e['err_msg'] else "")
              + "</summary>")
            if e["err_msg"]:
                A(f"<div class='err'>err_msg: {_esc(e['err_msg'][:500])}</div>")
            A("<table class='steps'><tr><th>#</th><th>action</th><th>think</th>"
              "<th class='num'>r</th><th>error</th></tr>")
            for s in e["steps"]:
                if s["action"] is None and not s["action_error"]:
                    continue
                think = s["think"][:280] + ("…" if len(s["think"]) > 280 else "")
                A(f"<tr><td class='num'>{_esc(s['i'])}</td>"
                  f"<td class='act'>{_esc(s['action'])}</td>"
                  f"<td class='think'>{_esc(think)}</td>"
                  f"<td class='num'>{_fmt(s['reward'],nd=2)}</td>"
                  f"<td class='err'>{_esc(s['action_error'][:200])}</td></tr>")
            A("</table></details>")
        if len(ordered) > cap:
            A(f"<div class='sub'>… showing {cap} of {len(ordered)} episodes "
              f"(raise with --max-traj).</div>")

    A(f"<script>{js}</script>")
    A("</div></body></html>")
    return "\n".join(out)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("EVAL_RESULTS_ROOT", "./eval-results"),
                    help="dir holding studies; the newest is used (ignored if --study-dir).")
    ap.add_argument("--study-dir", default=None, help="an explicit AgentLab study dir.")
    ap.add_argument("--out", default=None, help="output HTML path (default: <study>/trajectory_report.html).")
    ap.add_argument("--variant", default=os.environ.get("VARIANT", ""), help="label for the report header.")
    ap.add_argument("--max-traj", type=int, default=400, help="cap episodes in the drill-down.")
    args = ap.parse_args()

    if args.study_dir:
        study = Path(args.study_dir)
    else:
        study = newest_study(Path(args.root))
    if not study or not study.exists():
        sys.exit(f"[miniwob_report] no study with summary_info.json under {args.root!r} "
                 "(run an eval first, or pass --study-dir).")

    episodes, mode = extract_episodes(study)
    if not episodes:
        sys.exit(f"[miniwob_report] no episodes found under {study}")
    stats = compute_stats(episodes)
    meta = {
        "title": "WEASEL eval — trajectory report",
        "study": study.name,
        "variant": args.variant or "—",
        "mode": mode,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "max_traj": args.max_traj,
    }
    out_path = Path(args.out) if args.out else (study / "trajectory_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(stats, episodes, meta), encoding="utf-8")

    print(f"[miniwob_report] {stats['n']} episodes, SR {stats['sr']*100:.1f}% "
          f"({stats['n_success']}/{stats['n']}), mode={mode}")
    print(f"[miniwob_report] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
