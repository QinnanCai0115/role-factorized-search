#!/usr/bin/env python3
"""Replay policy holes while keeping the backbone search plan fixed.

This script supports a controlled ablation:
1. Extract every policy hole from a reference orchestrator trace.
2. Fill those holes with another policy model via replay_fixed_policy_inputs.py.
3. Rebuild the original backbone conversation using the fixed backbone search
   turns, replace policy outputs, and ask the backbone for the final answer.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from generate_sft_rollout import (
    add_api_response_usage,
    add_token_usage,
    build_final_backbone_message,
    build_initial_backbone_messages,
    build_next_backbone_message,
    build_parallel_policy_results_for_backbone,
    call_chat_completion,
    compact_record,
    compute_final_scores,
    extract_backbone_final_answer,
    extract_final_answer,
    judge_backbone_final_answer,
    load_env_file,
    merge_token_usage,
    new_token_usage,
    resolve_api_key,
    round_seconds,
    write_jsonl_record,
)


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
    count = 0
    seen: set[str] = set()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for trace in iter_jsonl(args.trace):
            if trace.get("record_type") != "orchestrator_trace":
                continue
            for policy_round in trace.get("policy_rounds") or []:
                replay_id = str(policy_round.get("policy_round_id") or "")
                if not replay_id or replay_id in seen:
                    continue
                seen.add(replay_id)
                row = {
                    "record_type": "fixed_backbone_policy_hole",
                    "replay_id": replay_id,
                    "source_index": trace.get("source_index"),
                    "uid": trace.get("uid"),
                    "question": trace.get("question"),
                    "ground_truth": trace.get("ground_truth"),
                    "ground_truth_targets": trace.get("ground_truth_targets", []),
                    "policy_round_id": replay_id,
                    "round": policy_round.get("round"),
                    "query_index": policy_round.get("query_index"),
                    "query_count": policy_round.get("query_count"),
                    "policy_input": policy_round.get("policy_input"),
                    "backbone_search_block": policy_round.get("backbone_search_block"),
                    "reference_policy_final_output": policy_round.get("final_output"),
                    "reference_policy_final_content": policy_round.get("final_content"),
                    "reference_policy_model": (trace.get("model_pair") or {}).get("policy_model"),
                }
                out.write(json.dumps(compact_record(row), ensure_ascii=False) + "\n")
                count += 1
                if args.limit and count >= args.limit:
                    print(f"wrote={count} output={args.output}")
                    return
    print(f"wrote={count} output={args.output}")


def load_replay_results(path: str) -> dict[str, dict[str, Any]]:
    results = {}
    for row in iter_jsonl(path):
        replay_id = str(row.get("replay_id") or row.get("policy_round_id") or "")
        if replay_id:
            results[replay_id] = row
    return results


def replay_row_to_policy_result(row: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not row:
        return {
            "final_output": "",
            "final_content": "",
            "has_answer_evidence": False,
            "elapsed_seconds": None,
            "token_usage": {},
            "api_call_stats": [],
            "error": "missing replay result",
        }
    return {
        "final_output": row.get("naive_final_output") or row.get("final_output") or "",
        "final_content": row.get("naive_final_output") or row.get("final_content") or "",
        "final_reasoning_content": row.get("naive_final_reasoning_content", ""),
        "has_answer_evidence": bool(row.get("naive_has_valid_answer") or row.get("has_answer_evidence")),
        "assistant_turns": row.get("naive_assistant_turns"),
        "tool_call_count": row.get("naive_tool_call_count"),
        "tool_trace": row.get("naive_tool_trace", []),
        "assistant_message_trace": row.get("naive_assistant_message_trace", []),
        "elapsed_seconds": row.get("naive_policy_elapsed_seconds"),
        "token_usage": row.get("naive_policy_token_usage", {}),
        "api_call_stats": row.get("naive_policy_api_call_stats", []),
        "error": row.get("error"),
        "policy_model": row.get("policy_model") or row.get("api_model"),
    }


def build_replaced_policy_event(
    event: dict[str, Any],
    replay_results: dict[str, dict[str, Any]],
    trace: dict[str, Any],
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, Any]]:
    round_idx = int(event.get("round") or 0)
    source_index = trace.get("source_index")
    policy_runs = []
    token_usage = new_token_usage()
    api_call_stats = []
    errors = []
    for result in event.get("results") or []:
        query_index = int(result.get("query_index") or 0)
        replay_id = f"{source_index}:r{round_idx}"
        if int(event.get("parallel_query_count") or len(event.get("results") or [])) != 1:
            replay_id = f"{source_index}:r{round_idx}:q{query_index}"
        search_query = str(result.get("search_query") or "")
        replay_row = replay_results.get(replay_id)
        policy_result = replay_row_to_policy_result(replay_row)
        if policy_result.get("error"):
            errors.append(f"q{query_index}: {policy_result.get('error')}")
        merge_token_usage(token_usage, policy_result.get("token_usage", {}))
        for call in policy_result.get("api_call_stats", []) or []:
            if isinstance(call, dict):
                api_call_stats.append(
                    {
                        "round": round_idx,
                        "query_index": query_index,
                        "search_query": search_query,
                        **call,
                    }
                )
        policy_runs.append(
            {
                "query_index": query_index,
                "search_query": search_query,
                "policy_result": policy_result,
                "elapsed_seconds": policy_result.get("elapsed_seconds"),
            }
        )

    backbone_input = build_parallel_policy_results_for_backbone(policy_runs)
    replaced_results = []
    for run in policy_runs:
        policy_result = run["policy_result"]
        replaced_results.append(
            {
                "query_index": run.get("query_index"),
                "search_query": run.get("search_query"),
                "final_output": policy_result.get("final_output", ""),
                "final_content": policy_result.get("final_content", ""),
                "final_reasoning_content": policy_result.get("final_reasoning_content", ""),
                "backbone_input": build_parallel_policy_results_for_backbone([run]),
                "has_answer_evidence": policy_result.get("has_answer_evidence", False),
                "elapsed_seconds": policy_result.get("elapsed_seconds"),
                "error": policy_result.get("error"),
            }
        )

    replaced_event = {
        **event,
        "stage": "policy_outputs",
        "results": replaced_results,
        "final_output": backbone_input,
        "backbone_input": backbone_input,
        "replaced_policy": True,
        "reference_policy_results": event.get("results", []),
        "token_usage": token_usage,
        "error": "; ".join(errors) if errors else None,
    }
    return replaced_event, backbone_input, api_call_stats, token_usage


def process_trace(trace: dict[str, Any], replay_results: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    question = str(trace.get("question") or "")
    ground_truth = trace.get("ground_truth")
    ground_truth_targets = trace.get("ground_truth_targets", [])
    token_usage = new_token_usage()
    api_call_stats = []
    new_chain = []
    backbone_messages = build_initial_backbone_messages(question)
    last_policy_backbone_output = ""
    sample_error = None

    chain = trace.get("orchestrator_chain") or []
    index = 0
    while index < len(chain):
        event = chain[index]
        stage = event.get("stage")
        if stage == "backbone_output":
            search_count = int(event.get("search_query_count") or 0)
            if search_count <= 0:
                break
            backbone_messages.append({"role": "assistant", "content": str(event.get("response") or "")})
            fixed_event = {**event, "fixed_from_reference": True}
            new_chain.append(fixed_event)
            if index + 1 < len(chain) and chain[index + 1].get("stage") == "policy_outputs":
                replaced_event, last_policy_backbone_output, policy_stats, policy_usage = build_replaced_policy_event(
                    chain[index + 1], replay_results, trace
                )
                new_chain.append(compact_record(replaced_event))
                merge_token_usage(token_usage, policy_usage)
                api_call_stats.extend(policy_stats)
                backbone_messages.append(build_next_backbone_message(last_policy_backbone_output))
                index += 2
                continue
        index += 1

    final_answer = ""
    final_output = ""
    final_answer_source = None
    backbone_response = ""
    backbone_error = None
    backbone_elapsed_seconds = None
    backbone_usage = {}
    try:
        if last_policy_backbone_output:
            backbone_messages.append(build_final_backbone_message(last_policy_backbone_output))
        api_started_at = time.perf_counter()
        backbone_response, raw_backbone = call_chat_completion(
            messages=backbone_messages,
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.backbone_model,
            timeout=args.api_timeout,
            max_retries=args.api_max_retries,
            temperature=args.backbone_temperature,
            max_tokens=args.backbone_max_tokens,
            no_proxy=args.no_proxy,
            retry_on_empty_content=True,
        )
        backbone_elapsed_seconds = round_seconds(time.perf_counter() - api_started_at)
        backbone_usage = add_api_response_usage(token_usage, raw_backbone, args.backbone_model)
        api_call_stats.append(
            {
                "stage": "backbone_final_reanswer",
                "model": args.backbone_model,
                "elapsed_seconds": backbone_elapsed_seconds,
                "usage": backbone_usage,
            }
        )
    except Exception as exc:
        backbone_elapsed_seconds = round_seconds(time.perf_counter() - api_started_at)
        backbone_error = f"{type(exc).__name__}: {exc}"
        sample_error = backbone_error

    final_answer = extract_backbone_final_answer(backbone_response) or ""
    if final_answer:
        final_output = backbone_response
        final_answer_source = "backbone"
    elif backbone_response.strip() and not backbone_error:
        final_output = backbone_response
        final_answer = backbone_response.strip()
        final_answer_source = "backbone"

    final_em, final_f1, computed_targets = compute_final_scores(final_answer, ground_truth)
    if not ground_truth_targets:
        ground_truth_targets = computed_targets

    judge = {
        "score": None,
        "is_correct": None,
        "response": "",
        "error": "final answer was not produced by backbone",
        "usage": {},
        "elapsed_seconds": None,
    }
    if final_answer_source == "backbone":
        judge = judge_backbone_final_answer(
            question=question,
            ground_truth_targets=ground_truth_targets,
            predicted_answer=final_answer,
            args=args,
        )
        add_token_usage(token_usage, judge.get("usage", {}), args.backbone_judge_model)
        api_call_stats.append(
            {
                "stage": "backbone_judge",
                "model": args.backbone_judge_model,
                "temperature": 0.0,
                "elapsed_seconds": judge.get("elapsed_seconds"),
                "usage": judge.get("usage", {}),
                "response": judge.get("response", ""),
                "score": judge.get("score"),
                "is_correct": judge.get("is_correct"),
                "error": judge.get("error"),
            }
        )

    new_chain.append(
        {
            "stage": "backbone_final_reanswer",
            "model": args.backbone_model,
            "response": backbone_response,
            "final_answer": final_answer,
            "elapsed_seconds": backbone_elapsed_seconds,
            "token_usage": backbone_usage,
            "error": backbone_error,
        }
    )
    new_chain.append(
        {
            "stage": "backbone_final_answer_llm_judge",
            "model": args.backbone_judge_model,
            "temperature": 0.0,
            "predicted_answer": final_answer,
            "ground_truth_targets": ground_truth_targets,
            "score": judge.get("score"),
            "is_correct": judge.get("is_correct"),
            "response": judge.get("response", ""),
            "elapsed_seconds": judge.get("elapsed_seconds"),
            "token_usage": judge.get("usage", {}),
            "error": judge.get("error"),
        }
    )

    return compact_record(
        {
            "record_type": "fixed_backbone_hole_replay_trace",
            "source_index": trace.get("source_index"),
            "uid": trace.get("uid"),
            "data_source": trace.get("data_source"),
            "question": question,
            "ground_truth": ground_truth,
            "ground_truth_targets": ground_truth_targets,
            "reference_trace_path": args.trace,
            "reference_final_answer": trace.get("final_answer"),
            "reference_final_answer_em": trace.get("final_answer_em"),
            "reference_final_answer_f1": trace.get("final_answer_f1"),
            "policy_round_count": sum(1 for item in new_chain if item.get("stage") == "policy_outputs"),
            "orchestrator_chain": new_chain,
            "final_output": final_output,
            "final_answer": final_answer,
            "final_answer_source": final_answer_source,
            "final_answer_em": final_em,
            "final_answer_f1": final_f1,
            "final_em": final_em,
            "final_f1": final_f1,
            "backbone_final_answer_llm_judge_score": judge.get("score"),
            "backbone_final_answer_llm_judge_is_correct": judge.get("is_correct"),
            "backbone_final_answer_llm_judge_response": judge.get("response", ""),
            "backbone_final_answer_llm_judge_error": judge.get("error"),
            "backbone_final_answer_llm_judge_elapsed_seconds": judge.get("elapsed_seconds"),
            "elapsed_seconds": round_seconds(time.perf_counter() - started_at),
            "token_usage": token_usage,
            "api_call_stats": api_call_stats,
            "model_pair": {"backbone_model": args.backbone_model, "policy_model": args.policy_model},
            "error": sample_error,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def load_done_source_indices(path: str) -> set[int]:
    done = set()
    if not path or not os.path.exists(path):
        return done
    for row in iter_jsonl(path):
        try:
            done.add(int(row["source_index"]))
        except Exception:
            pass
    return done


def command_answer(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    load_env_file(args.backbone_judge_env_file)
    args.api_key = resolve_api_key(args.api_key, args.api_key_env_var, ("DEEPSEEK_API_KEY",))
    args.backbone_judge_api_key = resolve_api_key(
        args.backbone_judge_api_key,
        args.backbone_judge_api_key_env_var,
        ("DEEPSEEK_API_KEY", "BACKBONE_API_KEY"),
    )
    if not args.api_key and not args.api_url.startswith(("http://127.0.0.1", "http://localhost")):
        raise ValueError("Missing backbone API key for non-local API.")

    traces = [row for row in iter_jsonl(args.trace) if row.get("record_type") == "orchestrator_trace"]
    if args.offset:
        traces = traces[args.offset :]
    if args.limit:
        traces = traces[: args.limit]
    done = load_done_source_indices(args.output) if args.resume else set()
    traces = [row for row in traces if int(row.get("source_index") or -1) not in done]
    replay_results = load_replay_results(args.replay)
    print(f"Loaded traces={len(traces) + len(done)} done={len(done)} processing={len(traces)} replay_rows={len(replay_results)}")

    lock = threading.Lock()
    if args.num_workers <= 1:
        for idx, trace in enumerate(traces, start=1):
            write_jsonl_record(args.output, process_trace(trace, replay_results, args), lock)
            if idx % 10 == 0 or idx == len(traces):
                print(f"processed={idx}/{len(traces)} output={args.output}")
        return

    completed = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_trace, trace, replay_results, args) for trace in traces]
        for future in as_completed(futures):
            write_jsonl_record(args.output, future.result(), lock)
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                print(f"processed={completed}/{len(futures)} output={args.output}")


def command_summarize(args: argparse.Namespace) -> None:
    rows = iter_jsonl(args.input)
    values = defaultdict(list)
    source_counts = Counter(str(row.get("final_answer_source") or "<none>") for row in rows)
    errors = [row.get("error") for row in rows if row.get("error")]
    for row in rows:
        for key in (
            "final_answer_em",
            "final_answer_f1",
            "final_em",
            "final_f1",
            "backbone_final_answer_llm_judge_score",
            "elapsed_seconds",
        ):
            if row.get(key) is not None:
                values[key].append(float(row[key]))
    summary = {
        "input": args.input,
        "sample_count": len(rows),
        "error_count": len(errors),
        "final_answer_source_counts": dict(source_counts),
    }
    for key, nums in values.items():
        summary[f"{key}_mean"] = statistics.fmean(nums) if nums else None
        summary[f"{key}_count"] = len(nums)
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract")
    extract.add_argument("--trace", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--limit", type=int, default=None)
    extract.set_defaults(func=command_extract)

    answer = sub.add_parser("answer")
    answer.add_argument("--trace", required=True)
    answer.add_argument("--replay", required=True)
    answer.add_argument("--output", required=True)
    answer.add_argument("--env_file", default=".secrets/deepseek.env")
    answer.add_argument("--api_url", default="https://api.deepseek.com/v1")
    answer.add_argument("--api_key", default=os.environ.get("BACKBONE_API_KEY", ""))
    answer.add_argument("--api_key_env_var", default="BACKBONE_API_KEY")
    answer.add_argument("--backbone_model", default="deepseek-reasoner")
    answer.add_argument("--backbone_temperature", type=float, default=0.0)
    answer.add_argument("--backbone_max_tokens", type=int, default=8192)
    answer.add_argument("--backbone_judge_api_url", default="https://api.deepseek.com/v1")
    answer.add_argument("--backbone_judge_model", default="deepseek-reasoner")
    answer.add_argument("--backbone_judge_env_file", default=".secrets/deepseek.env")
    answer.add_argument("--backbone_judge_api_key", default=os.environ.get("BACKBONE_JUDGE_API_KEY", ""))
    answer.add_argument("--backbone_judge_api_key_env_var", default="BACKBONE_JUDGE_API_KEY")
    answer.add_argument("--backbone_judge_max_tokens", type=int, default=4096)
    answer.add_argument("--policy_model", default="")
    answer.add_argument("--api_timeout", type=float, default=300.0)
    answer.add_argument("--api_max_retries", type=int, default=4)
    answer.add_argument("--num_workers", type=int, default=8)
    answer.add_argument("--limit", type=int, default=None)
    answer.add_argument("--offset", type=int, default=0)
    answer.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    answer.add_argument("--no_proxy", action=argparse.BooleanOptionalAction, default=True)
    answer.set_defaults(func=command_answer)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--input", required=True)
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(func=command_summarize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
