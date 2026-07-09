#!/usr/bin/env python3
"""Replay fixed policy inputs with a policy model and search_subagent.

This is a policy-only counterpart to generate_sft_rollout.py.  It is useful
when the backbone trajectory should be held fixed: each input row contains one
policy_input, and this script runs only the policy tool loop on that text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from generate_sft_rollout import (
    DEFAULT_MODEL,
    DEFAULT_POLICY_API_URL,
    DEFAULT_RETRIEVAL_URL,
    POLICY_SYSTEM_PROMPT,
    compact_record,
    extract_answer_evidence_blocks,
    load_env_file,
    load_records,
    new_token_usage,
    resolve_api_key,
    run_policy_tool_loop,
    write_jsonl_record,
)


NO_INFO_PATTERNS = [
    r"do(?:es)? not (?:mention|contain|include|provide|reference)",
    r"don.?t (?:mention|contain|include|provide|show|state)",
    r"not (?:mentioned|provided|specified|available|found|identified|enough|clear)",
    r"no (?:information|evidence|document|documents|source|sources|individual|relevant)",
    r"insufficient",
    r"cannot (?:determine|be determined|identify|answer)",
    r"could not determine",
    r"unable to determine",
    r"provided documents do not",
    r"retrieved documents do not",
    r"does not contain evidence",
    r"not contain evidence",
    r"evidence unavailable",
]

STOPWORDS = set(
    "the a an of in on at to by for with and or as from according was were is are "
    "be based provided retrieved documents document evidence states state that this "
    "those it its he she they his her has have had into about"
    .split()
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)


def has_no_info_answer(text: str) -> bool:
    lower = str(text or "").lower()
    return any(re.search(pattern, lower) for pattern in NO_INFO_PATTERNS)


def has_valid_answer(text: str) -> bool:
    lower = str(text or "").lower()
    return "<answer>" in lower and "</answer>" in lower and not has_no_info_answer(lower)


def extract_answer(text: str) -> str:
    match = ANSWER_RE.search(str(text or ""))
    return match.group(1).strip() if match else ""


def assistant_messages(policy_round: dict[str, Any]) -> list[dict[str, Any]]:
    messages = policy_round.get("assistant_message_trace") or []
    if messages:
        return messages
    stats = [
        item
        for item in (policy_round.get("policy_api_call_stats") or [])
        if item.get("stage") == "policy" and item.get("assistant_turn") is not None
    ]
    return [
        {
            "turn": item.get("assistant_turn"),
            "content": item.get("content", ""),
            "reasoning_content": item.get("reasoning_content", ""),
            "reasoning_details": item.get("reasoning_details"),
        }
        for item in stats
    ]


def assistant_turn_count(policy_round: dict[str, Any]) -> int:
    messages = assistant_messages(policy_round)
    if messages:
        return len(messages)
    tools = len(policy_round.get("tool_trace") or [])
    final_output = policy_round.get("final_output") or policy_round.get("final_content")
    return tools + 1 if tools and final_output else 0


def normalize_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return [
        token
        for token in re.sub(r"\s+", " ", text).split()
        if token and token not in STOPWORDS
    ]


def equivalent_answer(left: str, right: str) -> bool:
    left_tokens = normalize_tokens(left)
    right_tokens = normalize_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    left_norm = " ".join(left_tokens)
    right_norm = " ".join(right_tokens)
    if left_norm == right_norm:
        return True
    if len(left_norm) > 3 and left_norm in right_norm:
        return True
    if len(right_norm) > 3 and right_norm in left_norm:
        return True

    left_nums = set(re.findall(r"\d+", str(left)))
    right_nums = set(re.findall(r"\d+", str(right)))
    overlap = set(left_tokens) & set(right_tokens)
    if left_nums and right_nums and (left_nums <= right_nums or right_nums <= left_nums):
        if len(overlap) >= 2 or len(left_tokens) <= 4 or len(right_tokens) <= 4:
            return True

    return False


def iter_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def command_extract(args: argparse.Namespace) -> None:
    seen_ids: set[str] = set()
    count = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        with open(args.deepseek_trace, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                trace = json.loads(line)
                for policy_round in trace.get("policy_rounds") or []:
                    if args.require_assistant_turns is not None:
                        if assistant_turn_count(policy_round) != args.require_assistant_turns:
                            continue
                    if args.valid_answer_only:
                        final_output = str(policy_round.get("final_content") or policy_round.get("final_output") or "")
                        if not has_valid_answer(final_output):
                            continue
                    policy_input = str(policy_round.get("policy_input") or "").strip()
                    if not policy_input:
                        continue
                    policy_round_id = str(policy_round.get("policy_round_id") or "")
                    replay_id = f"{trace.get('source_index')}:{policy_round_id}"
                    if replay_id in seen_ids:
                        continue
                    seen_ids.add(replay_id)
                    row = {
                        "record_type": "fixed_policy_input",
                        "replay_id": replay_id,
                        "source_index": trace.get("source_index"),
                        "uid": trace.get("uid"),
                        "question": trace.get("question"),
                        "ground_truth": trace.get("ground_truth"),
                        "ground_truth_targets": trace.get("ground_truth_targets", []),
                        "policy_round_id": policy_round_id,
                        "round": policy_round.get("round"),
                        "query_index": policy_round.get("query_index"),
                        "query_count": policy_round.get("query_count"),
                        "policy_input": policy_input,
                        "backbone_search_block": policy_round.get("backbone_search_block"),
                        "deepseek_final_output": policy_round.get("final_content") or policy_round.get("final_output"),
                        "deepseek_final_answer": extract_answer(
                            str(policy_round.get("final_content") or policy_round.get("final_output") or "")
                        ),
                        "deepseek_has_valid_answer": has_valid_answer(
                            str(policy_round.get("final_content") or policy_round.get("final_output") or "")
                        ),
                        "deepseek_assistant_turns": assistant_turn_count(policy_round),
                        "deepseek_tool_call_count": len(policy_round.get("tool_trace") or []),
                        "deepseek_tool_trace": policy_round.get("tool_trace", []),
                        "deepseek_assistant_message_trace": assistant_messages(policy_round),
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    if args.limit and count >= args.limit:
                        print(f"wrote={count} output={args.output}")
                        return
    print(f"wrote={count} output={args.output}")


def load_done_replay_ids(path: str) -> set[str]:
    done: set[str] = set()
    if not path or not os.path.exists(path):
        return done
    for row in iter_jsonl(path):
        replay_id = str(row.get("replay_id") or "")
        if replay_id:
            done.add(replay_id)
    return done


def build_replay_record(row: dict[str, Any], result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    final_output = str(result.get("final_content") or result.get("final_output") or "")
    deepseek_answer = str(row.get("deepseek_final_answer") or "")
    naive_answer = extract_answer(final_output)
    if not has_valid_answer(final_output):
        comparison = "naive_no_valid_answer"
    elif equivalent_answer(naive_answer, deepseek_answer):
        comparison = "naive_same_or_equiv_answer"
    else:
        comparison = "naive_different_answer"

    return compact_record(
        {
            "record_type": "fixed_policy_replay",
            "replay_id": row.get("replay_id"),
            "source_index": row.get("source_index"),
            "uid": row.get("uid"),
            "question": row.get("question"),
            "ground_truth": row.get("ground_truth"),
            "ground_truth_targets": row.get("ground_truth_targets", []),
            "policy_round_id": row.get("policy_round_id"),
            "round": row.get("round"),
            "query_index": row.get("query_index"),
            "query_count": row.get("query_count"),
            "policy_input": row.get("policy_input"),
            "backbone_search_block": row.get("backbone_search_block"),
            "deepseek_final_output": row.get("deepseek_final_output"),
            "deepseek_final_answer": deepseek_answer,
            "deepseek_has_valid_answer": row.get("deepseek_has_valid_answer"),
            "deepseek_assistant_turns": row.get("deepseek_assistant_turns"),
            "deepseek_tool_call_count": row.get("deepseek_tool_call_count"),
            "deepseek_tool_trace": row.get("deepseek_tool_trace", []),
            "deepseek_assistant_message_trace": row.get("deepseek_assistant_message_trace", []),
            "naive_final_output": final_output,
            "naive_final_answer": naive_answer,
            "naive_has_valid_answer": has_valid_answer(final_output),
            "naive_assistant_turns": result.get("assistant_turns", 0),
            "naive_tool_call_count": result.get("tool_call_count", 0),
            "naive_tool_trace": result.get("tool_trace", []),
            "naive_assistant_message_trace": result.get("assistant_message_trace", []),
            "naive_messages": result.get("messages", []),
            "naive_sft_messages": result.get("sft_messages", []),
            "naive_policy_elapsed_seconds": result.get("elapsed_seconds"),
            "naive_policy_token_usage": result.get("token_usage", {}),
            "naive_policy_api_call_stats": result.get("api_call_stats", []),
            "comparison_to_deepseek": comparison,
            "api_model": args.policy_model,
            "policy_model": args.policy_model,
            "error": result.get("error"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def command_replay(args: argparse.Namespace) -> None:
    load_env_file(args.policy_env_file)
    args.policy_api_key = resolve_api_key(
        args.policy_api_key,
        args.policy_api_key_env_var,
        ("POLICY_API_KEY", "ZAI_API_KEY", "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"),
    )
    if not args.policy_model:
        args.policy_model = args.model
    if not args.policy_api_key and not args.policy_api_url.startswith("http://127.0.0.1"):
        if not args.policy_api_url.startswith("http://localhost"):
            raise ValueError("Missing policy API key for non-local policy API.")

    rows = load_records(args.input, limit=args.limit, offset=args.offset)
    done = load_done_replay_ids(args.output) if args.resume else set()
    rows = [row for row in rows if str(row.get("replay_id") or "") not in done]
    print(f"Loaded replay inputs={len(rows) + len(done)} done={len(done)} processing={len(rows)}")

    write_lock = threading.Lock()
    semaphore = (
        threading.BoundedSemaphore(args.retrieval_max_concurrent)
        if args.retrieval_max_concurrent and args.retrieval_max_concurrent > 0
        else None
    )

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        result = run_policy_tool_loop(search_query=str(row.get("policy_input") or ""), args=args, semaphore=semaphore)
        return build_replay_record(row, result, args)

    started_at = time.perf_counter()
    if args.num_workers <= 1:
        for index, row in enumerate(rows, start=1):
            write_jsonl_record(args.output, run_one(row), write_lock)
            if index % 10 == 0 or index == len(rows):
                print(f"processed={index}/{len(rows)} output={args.output}")
        return

    completed = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(run_one, row) for row in rows]
        for future in as_completed(futures):
            write_jsonl_record(args.output, future.result(), write_lock)
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                elapsed = time.perf_counter() - started_at
                print(f"processed={completed}/{len(futures)} elapsed={elapsed:.1f}s output={args.output}")


def command_summarize(args: argparse.Namespace) -> None:
    rows = iter_jsonl(args.input)
    counts = Counter()
    turn_dist = Counter()
    tool_dist = Counter()
    elapsed = []
    for row in rows:
        counts["total"] += 1
        if row.get("error"):
            counts["error"] += 1
        counts[str(row.get("comparison_to_deepseek") or "<none>")] += 1
        if row.get("naive_has_valid_answer"):
            counts["_naive_valid_answer_total"] += 1
        else:
            counts["_naive_no_valid_answer_total"] += 1
        turn_dist[int(row.get("naive_assistant_turns") or 0)] += 1
        tool_dist[int(row.get("naive_tool_call_count") or 0)] += 1
        if row.get("naive_policy_elapsed_seconds") is not None:
            elapsed.append(float(row["naive_policy_elapsed_seconds"]))

    summary = {
        "input": args.input,
        "record_count": counts["total"],
        "comparison_counts": {
            key: value
            for key, value in counts.items()
            if key
            not in {
                "total",
                "error",
                "_naive_valid_answer_total",
                "_naive_no_valid_answer_total",
            }
        },
        "naive_valid_answer_count": counts["_naive_valid_answer_total"],
        "naive_no_valid_answer_count": counts["_naive_no_valid_answer_total"],
        "error_count": counts["error"],
        "naive_assistant_turn_dist": dict(sorted(turn_dist.items())),
        "naive_tool_call_count_dist": dict(sorted(tool_dist.items())),
    }
    if elapsed:
        summary["naive_policy_elapsed_seconds_mean"] = sum(elapsed) / len(elapsed)
        summary["naive_policy_elapsed_seconds_median"] = statistics.median(elapsed)
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract fixed policy_input rows from DeepSeek traces.")
    extract.add_argument("--deepseek_trace", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--require_assistant_turns", type=int, default=3)
    extract.add_argument("--valid_answer_only", action=argparse.BooleanOptionalAction, default=True)
    extract.add_argument("--limit", type=int, default=None)
    extract.set_defaults(func=command_extract)

    replay = subparsers.add_parser("replay", help="Replay fixed policy_input rows with a policy API.")
    replay.add_argument("--input", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--model", default=DEFAULT_MODEL)
    replay.add_argument("--policy_model", default=None)
    replay.add_argument("--policy_api_url", default=DEFAULT_POLICY_API_URL)
    replay.add_argument("--policy_api_key", default=os.environ.get("POLICY_API_KEY", ""))
    replay.add_argument("--policy_api_key_env_var", default="POLICY_API_KEY")
    replay.add_argument("--policy_env_file", default="")
    replay.add_argument("--policy_enable_thinking", action=argparse.BooleanOptionalAction, default=False)
    replay.add_argument("--policy_thinking_field", default="thinking")
    replay.add_argument("--policy_thinking_type", default="enabled")
    replay.add_argument("--policy_preserve_reasoning_content", action=argparse.BooleanOptionalAction, default=False)
    replay.add_argument("--policy_extra_body_json", default="")
    replay.add_argument("--system_prompt", default=POLICY_SYSTEM_PROMPT)
    replay.add_argument("--retrieval_url", default=DEFAULT_RETRIEVAL_URL)
    replay.add_argument("--topk", type=int, default=3)
    replay.add_argument("--retrieval_timeout", type=int, default=180)
    replay.add_argument("--retrieval_max_concurrent", type=int, default=64)
    replay.add_argument("--save_raw_retrieval_response", action="store_true")
    replay.add_argument("--api_timeout", type=float, default=120.0)
    replay.add_argument("--api_max_retries", type=int, default=3)
    replay.add_argument("--temperature", type=float, default=0.2)
    replay.add_argument("--max_tokens", type=int, default=4096)
    replay.add_argument("--max_assistant_turns", type=int, default=3)
    replay.add_argument("--max_parallel_calls", type=int, default=1)
    replay.add_argument("--num_workers", type=int, default=8)
    replay.add_argument("--limit", type=int, default=None)
    replay.add_argument("--offset", type=int, default=0)
    replay.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    replay.add_argument("--save_raw_api_response", action="store_true")
    replay.add_argument("--no_proxy", action=argparse.BooleanOptionalAction, default=True)
    replay.set_defaults(func=command_replay)

    summarize = subparsers.add_parser("summarize", help="Summarize replay output.")
    summarize.add_argument("--input", required=True)
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(func=command_summarize)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
