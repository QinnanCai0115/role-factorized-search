#!/usr/bin/env python3
"""DeepSeek Reasoner no-retrieval direct QA baseline.

This runner evaluates an OpenAI-compatible chat model on QA records without
exposing any search/retrieval/tool protocol. It is intended to be comparable
with direct_search_budget_matched.py outputs while keeping total search calls
strictly at zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from scripts.baselines.direct_search_budget_matched import (
    add_token_usage,
    call_chat_completion,
    compute_final_scores,
    extract_final_answer,
    extract_ground_truth,
    extract_question,
    extract_token_usage,
    load_env_file,
    load_records,
    merge_token_usage,
    new_token_usage,
    round_seconds,
    save_json,
    to_jsonable,
)


BASELINE_NAME = "DeepSeekReasoner-NoSearch"
DEFAULT_API_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_INPUT = "/ai/cqn/datacon/data/hotpotqa_2wiki_musique_train/test_all.parquet"
DEFAULT_OUTPUT = "/ai/cqn/datacon/data/deepseek_reasoner_no_search_test_all/predictions.json"

NO_SEARCH_SYSTEM_PROMPT = """You are a precise QA assistant.

Answer the question using only your existing knowledge and reasoning. You do
not have access to search, retrieval, browsing, databases, or tools, and you
must not ask to call any tool.

Output only the final answer, with no explanation, citations, reasoning steps,
or extra text. For yes/no questions, start with "Yes" or "No". For entity,
date, place, or country questions, output only the answer value when possible.
"""


def build_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": NO_SEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": f"<question>\n{question}\n</question>"},
    ]


def default_jsonl_path(output_path: str) -> str:
    path = Path(output_path)
    if path.suffix:
        return str(path.with_suffix(".jsonl"))
    return f"{output_path}.jsonl"


def iter_jsonl(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def done_source_indices(path: str) -> set[int]:
    done: set[int] = set()
    for row in iter_jsonl(path):
        try:
            done.add(int(row.get("source_index")))
        except Exception:
            continue
    return done


def append_jsonl(path: str, row: dict[str, Any], lock: threading.Lock) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def process_one(index: int, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    baseline_name = getattr(args, "baseline_name", BASELINE_NAME)
    source_index = int(row.get("__source_index", index))
    uid = str(row.get("uid") or row.get("id") or uuid4().hex)
    question = extract_question(row)
    ground_truth = extract_ground_truth(row)
    started_at = time.perf_counter()

    token_usage = new_token_usage()
    api_call_stats: list[dict[str, Any]] = []
    messages = build_messages(question)
    final_output = ""
    final_answer = ""
    raw_response: dict[str, Any] = {}
    error: Optional[str] = None

    if not question:
        final_em, final_f1, targets = compute_final_scores("", ground_truth)
        return {
            "baseline_name": baseline_name,
            "source_index": source_index,
            "uid": uid,
            "data_source": row.get("data_source") or row.get("dataset"),
            "question": "",
            "ground_truth": ground_truth,
            "ground_truth_targets": targets,
            "final_output": "",
            "final_answer": "",
            "final_answer_source": None,
            "final_em": final_em,
            "final_f1": final_f1,
            "final_answer_em": final_em,
            "final_answer_f1": final_f1,
            "total_search_calls": 0,
            "total_retrieval_queries": 0,
            "search_calls_per_outer_round": [],
            "fallback_triggered": False,
            "natural_final_answer": False,
            "tool_calls_allowed": False,
            "token_usage": token_usage,
            "api_call_stats": api_call_stats,
            "trajectory": [],
            "evidence_bank": [],
            "error": "empty question",
            "elapsed_seconds": round_seconds(time.perf_counter() - started_at),
        }

    api_started_at = time.perf_counter()
    try:
        final_output, raw_response = call_chat_completion(
            messages=messages,
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            timeout=args.api_timeout,
            max_retries=args.api_max_retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            no_proxy=args.no_proxy,
            extra_body=args.extra_body,
        )
        usage = extract_token_usage(raw_response)
        add_token_usage(token_usage, usage, args.model)
        api_call_stats.append(
            {
                "stage": "no_search_direct_answer",
                "model": args.model,
                "elapsed_seconds": round_seconds(time.perf_counter() - api_started_at),
                "usage": usage,
            }
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        api_call_stats.append(
            {
                "stage": "no_search_direct_answer",
                "model": args.model,
                "elapsed_seconds": round_seconds(time.perf_counter() - api_started_at),
                "usage": {},
                "error": error,
            }
        )

    final_answer = extract_final_answer(final_output)
    final_em, final_f1, targets = compute_final_scores(final_answer, ground_truth)
    trajectory: list[dict[str, Any]] = [
        {
            "stage": "no_search_direct_answer",
            "response": final_output,
            "error": error,
        }
    ]
    if args.save_raw_api_response:
        trajectory[0]["raw_api_response"] = raw_response

    result = {
        "baseline_name": baseline_name,
        "source_index": source_index,
        "uid": uid,
        "data_source": row.get("data_source") or row.get("dataset"),
        "split": row.get("split"),
        "question": question,
        "ground_truth": ground_truth,
        "ground_truth_targets": targets,
        "final_output": final_output,
        "final_answer": final_answer,
        "final_answer_source": "no_search_direct",
        "final_em": final_em,
        "final_f1": final_f1,
        "final_answer_em": final_em,
        "final_answer_f1": final_f1,
        "total_search_calls": 0,
        "total_retrieval_queries": 0,
        "search_calls_per_outer_round": [],
        "fallback_triggered": False,
        "natural_final_answer": bool(final_answer),
        "tool_calls_allowed": False,
        "token_usage": token_usage,
        "api_call_stats": api_call_stats,
        "trajectory": trajectory,
        "evidence_bank": [],
        "final_evidence_ids": [],
        "available_evidence_ids": [],
        "invalid_evidence_ids": [],
        "final_evidence_refs_valid": True,
        "final_answer_cites_existing_evidence": False,
        "error": error,
        "elapsed_seconds": round_seconds(time.perf_counter() - started_at),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.save_messages:
        result["messages"] = messages
    return to_jsonable(result)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    if count == 0:
        return {
            "count": 0,
            "final_em": None,
            "final_f1": None,
            "error_count": 0,
            "avg_total_search_calls": 0.0,
            "max_total_search_calls": 0,
            "avg_total_retrieval_queries": 0.0,
            "max_total_retrieval_queries": 0,
            "data_source_counts": {},
            "token_usage": new_token_usage(),
        }

    em_values = [r.get("final_em") for r in results if isinstance(r.get("final_em"), (int, float))]
    f1_values = [r.get("final_f1") for r in results if isinstance(r.get("final_f1"), (int, float))]
    aggregate_usage = new_token_usage()
    for result in results:
        merge_token_usage(aggregate_usage, result.get("token_usage", {}))
    data_sources = Counter(str(r.get("data_source") or "<none>") for r in results)

    return {
        "count": count,
        "final_em": sum(em_values) / len(em_values) if em_values else None,
        "final_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "error_count": sum(1 for r in results if r.get("error")),
        "avg_total_search_calls": 0.0,
        "max_total_search_calls": 0,
        "avg_total_retrieval_queries": 0.0,
        "max_total_retrieval_queries": 0,
        "data_source_counts": dict(data_sources),
        "token_usage": aggregate_usage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{BASELINE_NAME} evaluation")
    parser.add_argument("--baseline_name", default=BASELINE_NAME, help="Baseline name written to outputs.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input parquet/jsonl/json file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Final output JSON file.")
    parser.add_argument("--output_jsonl", default="", help="Incremental per-sample JSONL path.")
    parser.add_argument("--env_file", default=".secrets/deepseek.env", help="Optional env file with API key.")
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--api_timeout", type=float, default=300.0)
    parser.add_argument("--api_max_retries", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--no_proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_messages", action="store_true", help="Include full chat messages per sample.")
    parser.add_argument("--save_raw_api_response", action="store_true", help="Include raw API responses in trajectory.")
    parser.add_argument(
        "--extra_body_json",
        default="",
        help="Optional JSON object merged into each chat completion request payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.extra_body = {}
    if args.extra_body_json:
        args.extra_body.update(json.loads(args.extra_body_json))
    load_env_file(args.env_file)
    if not args.api_key:
        args.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not args.api_key:
        raise ValueError("Missing API key: set DEEPSEEK_API_KEY or pass --api_key.")

    args.output_jsonl = args.output_jsonl or default_jsonl_path(args.output)
    rows = load_records(args.input, limit=args.limit, offset=args.offset)
    for local_idx, row in enumerate(rows):
        row["__source_index"] = args.offset + local_idx

    completed = done_source_indices(args.output_jsonl) if args.resume else set()
    pending = [(i, row) for i, row in enumerate(rows) if int(row["__source_index"]) not in completed]
    write_lock = threading.Lock()

    if pending:
        if args.num_workers <= 1:
            for i, row in tqdm(pending, desc=args.baseline_name):
                append_jsonl(args.output_jsonl, process_one(i, row, args), write_lock)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {executor.submit(process_one, i, row, args): i for i, row in pending}
                for future in tqdm(as_completed(futures), total=len(futures), desc=args.baseline_name):
                    append_jsonl(args.output_jsonl, future.result(), write_lock)

    results = iter_jsonl(args.output_jsonl)
    requested_indices = {int(row["__source_index"]) for _, row in enumerate(rows)}
    results = [r for r in results if int(r.get("source_index", -1)) in requested_indices]
    results.sort(key=lambda r: int(r.get("source_index", -1)))

    summary = summarize_results(results)
    output = {
        "baseline_name": args.baseline_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "model": args.model,
        "api_url": args.api_url,
        "extra_body": args.extra_body,
        "no_search": {
            "tool_calls_allowed": False,
            "retrieval_url": None,
            "search_protocol": None,
            "prompt": NO_SEARCH_SYSTEM_PROMPT,
        },
        "summary": summary,
        "results_jsonl": args.output_jsonl,
        "results": results,
    }
    save_json(output, args.output)
    print(
        f"[{args.baseline_name}] count={summary['count']} "
        f"EM={summary['final_em'] if summary['final_em'] is not None else 'NA'} "
        f"F1={summary['final_f1'] if summary['final_f1'] is not None else 'NA'} "
        f"errors={summary['error_count']} max_search_calls=0"
    )
    print(f"Saved predictions to {args.output}")
    print(f"Incremental results at {args.output_jsonl}")


if __name__ == "__main__":
    main()
