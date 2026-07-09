#!/usr/bin/env python3
"""Judge final answers in DirectSearch-BudgetMatched prediction JSON files.

The script reuses the existing final-answer judge from generate_sft_rollout.py:
question + golden answer + predicted final answer -> strict True/False.

It writes incremental JSONL records for resume safety, then writes a full JSON
copy of the input with per-result llm_final_judge_score fields.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.examples.search_r1_like.generate_sft_rollout import (  # noqa: E402
    judge_backbone_final_answer,
    load_env_file,
)


DEFAULT_INPUT = (
    "/ai/cqn/s3/ckpt/untrained_qwen_direct_search_baseline/"
    "qwen3_1p7b_direct_search_no_think_test_all_4125_mt128/predictions.json"
)


def default_output_path(input_path: str) -> str:
    path = Path(input_path)
    if path.suffix:
        return str(path.with_name(path.stem + ".llm_final_judge" + path.suffix))
    return str(path) + ".llm_final_judge.json"


def default_jsonl_path(output_path: str) -> str:
    path = Path(output_path)
    if path.suffix:
        return str(path.with_suffix(".jsonl"))
    return str(path) + ".jsonl"


def load_prediction_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Expected input JSON with a top-level results list.")
    return data


def collect_ground_truth_targets(row: dict[str, Any]) -> list[str]:
    targets = row.get("ground_truth_targets")
    if isinstance(targets, list):
        values = [str(item).strip() for item in targets if str(item).strip()]
        if values:
            return values

    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict):
        for key in ("target", "answer", "answers", "golden_answers", "gt", "gts"):
            value = ground_truth.get(key)
            values = flatten_targets(value)
            if values:
                return values
    return flatten_targets(ground_truth)


def flatten_targets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        values: list[str] = []
        for key in ("target", "answer", "answers", "golden_answers", "ground_truth", "gt", "gts"):
            values.extend(flatten_targets(value.get(key)))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(flatten_targets(item))
        return values
    text = str(value).strip()
    return [text] if text else []


def load_existing_jsonl(path: str) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    jsonl = Path(path)
    if not jsonl.exists():
        return existing
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = result_key(row)
            if key:
                existing[key] = row
    return existing


def result_key(row: dict[str, Any]) -> str:
    source_index = row.get("source_index")
    if source_index is not None:
        return str(source_index)
    uid = row.get("uid")
    return str(uid) if uid is not None else ""


def build_judge_items(results: list[dict[str, Any]], *, limit: Optional[int], offset: int) -> list[dict[str, Any]]:
    sliced = results[offset:]
    if limit is not None:
        sliced = sliced[:limit]

    items: list[dict[str, Any]] = []
    for row in sliced:
        if not isinstance(row, dict):
            continue
        item = {
            "source_index": row.get("source_index"),
            "uid": row.get("uid"),
            "data_source": row.get("data_source"),
            "question": str(row.get("question") or "").strip(),
            "ground_truth_targets": collect_ground_truth_targets(row),
            "final_answer": str(row.get("final_answer") or "").strip(),
            "final_em": row.get("final_em"),
            "final_f1": row.get("final_f1"),
            "final_answer_source": row.get("final_answer_source"),
            "total_search_calls": row.get("total_search_calls"),
            "total_retrieval_queries": row.get("total_retrieval_queries"),
        }
        items.append(item)
    return items


def apply_judge_result(result: dict[str, Any], judge_row: dict[str, Any]) -> None:
    result["llm_final_judge_score"] = judge_row.get("llm_final_judge_score")
    result["llm_final_judge_is_correct"] = judge_row.get("llm_final_judge_is_correct")
    result["llm_final_judge_response"] = judge_row.get("llm_final_judge_response")
    result["llm_final_judge_error"] = judge_row.get("llm_final_judge_error")
    result["llm_final_judge_model"] = judge_row.get("llm_final_judge_model")
    result["llm_final_judge_usage"] = judge_row.get("llm_final_judge_usage", {})
    result["llm_final_judge_elapsed_seconds"] = judge_row.get("llm_final_judge_elapsed_seconds")


def summarize(judge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        float(row["llm_final_judge_score"])
        for row in judge_rows
        if isinstance(row.get("llm_final_judge_score"), (int, float))
    ]
    errors = [str(row.get("llm_final_judge_error")) for row in judge_rows if row.get("llm_final_judge_error")]
    return {
        "llm_final_judged_count": len(scores),
        "llm_final_judge_score_mean": statistics.fmean(scores) if scores else None,
        "llm_final_judge_score_sum": sum(scores),
        "llm_final_judge_error_count": len(errors),
        "llm_final_judge_error_counts": dict(Counter(errors)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge final answers and add llm_final_judge_score.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--jsonl_output", default=None, help="Incremental judge records; used for resume.")
    parser.add_argument(
        "--env_file",
        action="append",
        default=[],
        help="Env file with BACKBONE_JUDGE_API_KEY or DEEPSEEK_API_KEY. Can be passed multiple times.",
    )
    parser.add_argument("--judge_api_url", default=os.environ.get("BACKBONE_JUDGE_API_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--judge_api_key", default=os.environ.get("BACKBONE_JUDGE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "")
    parser.add_argument("--judge_model", default=os.environ.get("BACKBONE_JUDGE_MODEL", "deepseek-reasoner"))
    parser.add_argument("--judge_max_tokens", type=int, default=16)
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("API_TIMEOUT", "180")))
    parser.add_argument("--api_max_retries", type=int, default=int(os.environ.get("API_MAX_RETRIES", "4")))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Ignore existing JSONL records and rejudge.")
    parser.add_argument("--no_proxy", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.env_file:
        args.env_file = [
            "/ai/cqn/datacon/.secrets/deepseek.env",
            "/ai/cqn/s3/.secrets/deepseek.env",
        ]
    for env_file in args.env_file:
        load_env_file(env_file)

    if not args.judge_api_key:
        args.judge_api_key = (
            os.environ.get("BACKBONE_JUDGE_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or ""
        )
    if not args.judge_api_key and not args.judge_api_url.startswith(("http://127.0.0.1", "http://localhost")):
        raise SystemExit("Missing judge API key: set BACKBONE_JUDGE_API_KEY or DEEPSEEK_API_KEY.")

    output_path = args.output or default_output_path(args.input)
    jsonl_path = args.jsonl_output or default_jsonl_path(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    data = load_prediction_file(args.input)
    results = data["results"]
    items = build_judge_items(results, limit=args.limit, offset=args.offset)

    existing = {} if args.force else load_existing_jsonl(jsonl_path)
    pending = [item for item in items if result_key(item) not in existing]
    print(
        f"input={args.input}\noutput={output_path}\njsonl_output={jsonl_path}\n"
        f"items={len(items)} existing={len(existing)} pending={len(pending)} "
        f"judge_model={args.judge_model}",
        flush=True,
    )

    judge_args = argparse.Namespace(
        backbone_judge_api_url=args.judge_api_url,
        backbone_judge_api_key=args.judge_api_key,
        backbone_judge_model=args.judge_model,
        backbone_judge_max_tokens=args.judge_max_tokens,
        api_timeout=args.api_timeout,
        api_max_retries=args.api_max_retries,
        no_proxy=args.no_proxy,
    )

    def judge_one(item: dict[str, Any]) -> dict[str, Any]:
        judge = judge_backbone_final_answer(
            question=item["question"],
            ground_truth_targets=item["ground_truth_targets"],
            predicted_answer=item["final_answer"],
            args=judge_args,
        )
        return {
            **item,
            "llm_final_judge_score": judge.get("score"),
            "llm_final_judge_is_correct": judge.get("is_correct"),
            "llm_final_judge_response": judge.get("response"),
            "llm_final_judge_error": judge.get("error"),
            "llm_final_judge_usage": judge.get("usage", {}),
            "llm_final_judge_elapsed_seconds": judge.get("elapsed_seconds"),
            "llm_final_judge_model": args.judge_model,
            "llm_final_judge_created_at": datetime.now(timezone.utc).isoformat(),
        }

    mode = "w" if args.force else "a"
    with open(jsonl_path, mode, encoding="utf-8") as wf:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(judge_one, item) for item in pending]
            for idx, future in enumerate(as_completed(futures), 1):
                row = future.result()
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                existing[result_key(row)] = row
                if idx % 50 == 0:
                    print(f"judged {idx}/{len(pending)}", flush=True)

    judge_rows = list(load_existing_jsonl(jsonl_path).values())
    by_key = {result_key(row): row for row in judge_rows}
    for result in results:
        row = by_key.get(result_key(result))
        if row:
            apply_judge_result(result, row)

    llm_summary = summarize(judge_rows)
    data.setdefault("summary", {})
    data["summary"].update(llm_summary)
    data["llm_final_judge"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "output": output_path,
        "jsonl_output": jsonl_path,
        "judge_api_url": args.judge_api_url,
        "judge_model": args.judge_model,
        "judge_max_tokens": args.judge_max_tokens,
        **llm_summary,
    }

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)
    print(json.dumps(data["llm_final_judge"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
