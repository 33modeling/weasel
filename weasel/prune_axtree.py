"""Prune AXTree sections before WEASEL score computation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


AXTREE_SECTION_RE = re.compile(
    r"(##\s*AXTree:\s*)(.*?)(# History of interaction with the task:?[\t ]*)",
    re.S,
)
ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.S)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune the AXTree in each training example using target-centered "
            "pruning when an action bid is available, otherwise fall back to "
            "a prefix threshold."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL dataset.")
    parser.add_argument("--output", required=True, help="Pruned dataset output path.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Number of bid entries to keep on each side of the target bid.",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=int,
        default=120,
        help="Number of bid entries kept when centered pruning cannot be used.",
    )
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Optional JSON file with pruning summary statistics.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact JSON.",
    )
    return parser.parse_args()


def load_jsonl_or_json(path: Path) -> List[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON list or JSONL file: {path}")
    except json.JSONDecodeError:
        pass

    records: List[Any] = []
    try:
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        if records:
            return records
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Expected a JSON list or JSONL file: {path}")


def write_json(data: Any, path: Path, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_indent = None if indent == 0 else indent
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=actual_indent)


def get_message_content(
    item: Dict[str, Any],
    role: str,
    fallback_index: Optional[int] = None,
    last: bool = False,
) -> str:
    messages = item.get("messages", [])
    if isinstance(messages, list):
        matches = [
            message.get("content", "")
            for message in messages
            if isinstance(message, dict) and message.get("role") == role
        ]
        if matches:
            return matches[-1] if last else matches[0]

    if fallback_index is not None:
        try:
            return messages[fallback_index].get("content", "")
        except Exception:
            return ""
    return ""


def set_user_prompt(item: Dict[str, Any], content: str) -> bool:
    messages = item.get("messages", [])
    if not isinstance(messages, list):
        return False

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            message["content"] = content
            return True

    try:
        messages[1]["content"] = content
        return True
    except Exception:
        return False


def extract_bid_from_action(text: str) -> Tuple[Optional[str], Optional[str], str]:
    action_match = ACTION_RE.search(text or "")
    if not action_match:
        return None, None, ""

    think_match = THINK_RE.search(text or "")
    think = think_match.group(1) if think_match else ""

    action_text = action_match.group(1)
    action_name_match = re.match(r"([a-zA-Z_]+)\(", action_text)
    if not action_name_match:
        return None, None, think

    action_name = action_name_match.group(1)

    if action_name in {"click", "dblclick", "hover", "press", "focus", "clear"}:
        match = re.search(rf"{action_name}\('([\w\d]+)'", action_text)
        return action_name, match.group(1) if match else None, think

    if action_name == "fill":
        match = re.search(r"fill\('([\w\d]+)',", action_text)
        return action_name, match.group(1) if match else None, think

    if action_name == "select_option":
        match = re.search(r"select_option\('([\w\d]+)',", action_text)
        return action_name, match.group(1) if match else None, think

    if action_name == "drag_and_drop":
        match = re.search(r"drag_and_drop\('([\w\d]+)',\s*'([\w\d]+)'\)", action_text)
        if not match:
            return action_name, None, think
        bid1, bid2 = match.group(1), match.group(2)
        try:
            bid = max(bid1, bid2, key=lambda value: int(re.search(r"\d+", value).group()))
        except Exception:
            bid = bid1
        return action_name, bid, think

    if action_name == "upload_file":
        match = re.search(r"upload_file\('([\w\d]+)'", action_text)
        return action_name, match.group(1) if match else None, think

    if action_name in {"noop", "scroll", "send_msg_to_user", "go_back", "go_forward", "goto"}:
        return action_name, None, think

    return action_name, None, think


def extract_axtree_stats(text: str) -> Optional[Tuple[str, int, int]]:
    match = AXTREE_SECTION_RE.search(text or "")
    if not match:
        return None

    full_axtree = match.group(2).strip()
    bid_list = re.findall(r"\[([^\]]+)\]", full_axtree)
    tokens = re.findall(r"\S+|\s+", full_axtree)
    return full_axtree, len(bid_list), len(tokens)


def prune_threshold(axtree_text: str, threshold: int) -> str:
    match = AXTREE_SECTION_RE.search(axtree_text or "")
    if not match:
        return axtree_text

    axtree_body = match.group(2)
    before = axtree_text[:match.start(2)]
    after = axtree_text[match.end(2):]

    lines = axtree_body.strip().splitlines()
    pruned_lines: List[str] = []
    bid_count = 0

    for line in lines:
        pruned_lines.append(line)
        if re.search(r"\[[^\]]+\]", line):
            bid_count += 1
        if bid_count >= threshold:
            break

    pruned_axtree = "\n".join(pruned_lines)
    return f"{before}{pruned_axtree}\n{after}"


def prune_gold_centered(axtree_text: str, bid: str, window_size: int) -> str:
    match = AXTREE_SECTION_RE.search(axtree_text or "")
    if not match:
        return axtree_text

    full_axtree = match.group(2)
    before = axtree_text[:match.start(2)]
    after = axtree_text[match.end(2):]

    bid_list = re.findall(r"\[([^\]]+)\]", full_axtree)
    if bid not in bid_list:
        return axtree_text

    index = bid_list.index(bid)
    start_i = max(index - window_size, 0)
    end_i = min(index + window_size + 1, len(bid_list))
    bid_subset = set(bid_list[start_i:end_i])

    lines = full_axtree.splitlines()
    pruned_lines: List[str] = []
    started = False

    for line in lines:
        line_bid_match = re.search(r"\[([^\]]+)\]", line)

        if line_bid_match and line_bid_match.group(1) in bid_subset:
            started = True
            pruned_lines.append(line)
            continue

        if started and not line_bid_match:
            pruned_lines.append(line)
            continue

        if line_bid_match and started and line_bid_match.group(1) not in bid_subset:
            break

    if not pruned_lines:
        return axtree_text

    pruned_axtree = "\n".join(pruned_lines)
    return before + lines[0] + "\n" + pruned_axtree + after


def prune_item(
    item: Dict[str, Any],
    window_size: int,
    fallback_threshold: int,
) -> Tuple[bool, str, Optional[int], Optional[int]]:
    action_content = get_message_content(item, "assistant", fallback_index=2, last=True)
    user_prompt = get_message_content(item, "user", fallback_index=1)
    original_stats = extract_axtree_stats(user_prompt)
    if original_stats is None:
        return False, "missing_axtree", None, None

    _, bid, _ = extract_bid_from_action(action_content)

    if bid:
        pruned_content = prune_gold_centered(user_prompt, bid, window_size)
        centered_stats = extract_axtree_stats(pruned_content)
        if centered_stats is not None and centered_stats[0] != original_stats[0]:
            method = "centered"
        else:
            pruned_content = prune_threshold(user_prompt, fallback_threshold)
            method = "fallback_threshold"
    else:
        pruned_content = prune_threshold(user_prompt, fallback_threshold)
        method = "threshold"

    pruned_stats = extract_axtree_stats(pruned_content)
    if pruned_stats is None:
        return False, "missing_axtree_after_prune", None, None

    updated = set_user_prompt(item, pruned_content)
    if not updated:
        return False, "missing_user_message", None, None

    return True, method, original_stats[2], pruned_stats[2]


def main() -> None:
    args = parse_args()
    dataset = load_jsonl_or_json(Path(args.input))

    stats: Dict[str, Any] = {
        "total_examples": len(dataset),
        "pruned_examples": 0,
        "missing_axtree": 0,
        "missing_user_message": 0,
        "missing_axtree_after_prune": 0,
        "centered": 0,
        "threshold": 0,
        "fallback_threshold_count": 0,
        "original_tokens_sum": 0,
        "pruned_tokens_sum": 0,
        "window_size": args.window_size,
        "fallback_threshold": args.fallback_threshold,
    }

    for item in dataset:
        if not isinstance(item, dict):
            continue
        success, method, original_tokens, pruned_tokens = prune_item(
            item,
            window_size=args.window_size,
            fallback_threshold=args.fallback_threshold,
        )
        if not success:
            stats[method] += 1
            continue

        stats["pruned_examples"] += 1
        stats[method] += 1
        if original_tokens is not None:
            stats["original_tokens_sum"] += original_tokens
        if pruned_tokens is not None:
            stats["pruned_tokens_sum"] += pruned_tokens

    if stats["pruned_examples"]:
        stats["avg_original_axtree_tokens"] = (
            stats["original_tokens_sum"] / stats["pruned_examples"]
        )
        stats["avg_pruned_axtree_tokens"] = (
            stats["pruned_tokens_sum"] / stats["pruned_examples"]
        )
    else:
        stats["avg_original_axtree_tokens"] = 0.0
        stats["avg_pruned_axtree_tokens"] = 0.0

    write_json(dataset, Path(args.output), args.indent)
    if args.stats_output:
        write_json(stats, Path(args.stats_output), args.indent)

    print(
        "Pruned AXTree examples: "
        f"{stats['pruned_examples']}/{stats['total_examples']} "
        f"(centered={stats['centered']}, "
        f"threshold={stats['threshold']}, "
        f"fallback_threshold={stats['fallback_threshold_count']})"
    )
    print(
        "Average AXTree tokens: "
        f"{stats['avg_original_axtree_tokens']:.1f} -> "
        f"{stats['avg_pruned_axtree_tokens']:.1f}"
    )


if __name__ == "__main__":
    main()
