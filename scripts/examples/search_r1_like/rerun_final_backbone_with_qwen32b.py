#!/usr/bin/env python3
"""Post-hoc rerun only the final outer backbone answer with Qwen3-32B."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.examples.search_r1_like.generate_sft_rollout import (  # noqa: E402
    add_api_response_usage,
    build_final_backbone_message,
    build_initial_backbone_messages,
    build_next_backbone_message,
    call_chat_completion,
    compute_final_scores,
    extract_backbone_final_answer,
    extract_token_usage,
    judge_backbone_final_answer,
    new_token_usage,
    parse_bool_judge_response,
    round_seconds,
    to_jsonable,
)


def load_env_file(path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_DIR / p
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_extra_body(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise ValueError(f"invalid extra body JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("extra body JSON must be an object")
    return parsed


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def done_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    out: set[int] = set()
    for row in iter_jsonl(path):
        try:
            out.add(int(row.get("source_index")))
        except Exception:
            pass
    return out


def last_policy_item(chain: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for item in reversed(chain):
        if item.get("stage") == "policy_outputs":
            return item
    return None


def build_qwen32b_final_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str, int]:
    question = str(row.get("question") or "")
    messages = build_initial_backbone_messages(question)
    chain = row.get("orchestrator_chain") or []
    lp = last_policy_item(chain)
    last_round = lp.get("round") if lp else None
    last_policy_output = ""

    for item in chain:
        stage = item.get("stage")
        if stage == "backbone_output":
            if last_round is not None and item.get("round") is not None and int(item.get("round")) > int(last_round):
                break
            response = str(item.get("response") or "")
            if response:
                messages.append({"role": "assistant", "content": response})
        elif stage == "policy_outputs":
            output = str(item.get("backbone_input") or item.get("final_output") or "")
            if not output:
                continue
            if item is lp:
                last_policy_output = output
                break
            messages.append(build_next_backbone_message(output))

    if last_policy_output:
        messages.append(build_final_backbone_message(last_policy_output))
    else:
        messages.append({
            "role": "user",
            "content": "Final round. Do not output another <search>. Answer the original question and output only <final answer>...</final answer>.",
        })
    return messages, last_policy_output, 0 if lp is None else int(lp.get("round", 0))


def judge_qwen32b_final_answer(
    *,
    question: str,
    ground_truth_targets: list[str],
    predicted_answer: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not question or not ground_truth_targets or not str(predicted_answer or "").strip():
        return {"score": None, "is_correct": None, "response": "", "error": "missing question, golden answer, or predicted answer", "usage": {}, "elapsed_seconds": None}
    golden_answer_text = ground_truth_targets[0] if len(ground_truth_targets) == 1 else json.dumps(ground_truth_targets, ensure_ascii=False)
    prompt = (
        "Given a Question and its Golden Answer, verify whether the Predicted Answer is correct.\n"
        "The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer.\n"
        "Respond with True if the prediction is correct and False otherwise.\n\n"
        f"Question:\n{question}\n"
        f"Golden Answer:\n{golden_answer_text}\n"
        f"Predicted Answer:\n{predicted_answer}"
    )
    started_at = time.perf_counter()
    try:
        response_text, raw_response = call_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            api_url=args.backbone_judge_api_url,
            api_key=args.backbone_judge_api_key,
            model=args.backbone_judge_model,
            timeout=args.api_timeout,
            max_retries=args.api_max_retries,
            temperature=0.0,
            max_tokens=int(getattr(args, "backbone_judge_max_tokens", 32) or 32),
            no_proxy=args.no_proxy,
            extra_body=parse_extra_body(args.judge_extra_body_json),
            retry_on_empty_content=True,
        )
        is_correct = parse_bool_judge_response(response_text)
        return {
            "score": None if is_correct is None else float(is_correct),
            "is_correct": is_correct,
            "response": response_text,
            "error": None if is_correct is not None else "judge response did not contain True or False",
            "usage": extract_token_usage(raw_response),
            "elapsed_seconds": round_seconds(time.perf_counter() - started_at),
        }
    except Exception as exc:
        return {"score": None, "is_correct": None, "response": "", "error": f"{type(exc).__name__}: {exc}", "usage": {}, "elapsed_seconds": round_seconds(time.perf_counter() - started_at)}


def process_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_index = int(row.get("source_index"))
    messages, last_policy_output, final_from_round = build_qwen32b_final_messages(row)
    usage = new_token_usage()
    api_stats: list[dict[str, Any]] = []
    error = None
    response_text = ""
    raw_response: dict[str, Any] = {}
    try:
        api_started = time.perf_counter()
        response_text, raw_response = call_chat_completion(
            messages=messages,
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            timeout=args.api_timeout,
            max_retries=args.api_max_retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            no_proxy=args.no_proxy,
            extra_body=parse_extra_body(args.extra_body_json),
            retry_on_empty_content=True,
        )
        call_usage = add_api_response_usage(usage, raw_response, args.model)
        api_stats.append({
            "stage": "qwen32b_final_backbone",
            "model": args.model,
            "elapsed_seconds": round_seconds(time.perf_counter() - api_started),
            "usage": call_usage,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    final_answer = extract_backbone_final_answer(response_text) or ""
    if not final_answer and response_text.strip() and not re.search(r"<search>.*?</search>", response_text, flags=re.S | re.I):
        final_answer = response_text.strip()
    final_em, final_f1, targets = compute_final_scores(final_answer, row.get("ground_truth"))

    judge_result = {
        "score": None,
        "is_correct": None,
        "response": "",
        "error": None,
        "usage": {},
        "elapsed_seconds": None,
    }
    if args.run_judge and final_answer:
        judge_result = judge_qwen32b_final_answer(
            question=str(row.get("question") or ""),
            ground_truth_targets=targets,
            predicted_answer=final_answer,
            args=args,
        )
        api_stats.append({
            "stage": "qwen32b_final_answer_judge",
            "model": args.backbone_judge_model,
            "elapsed_seconds": judge_result.get("elapsed_seconds"),
            "usage": judge_result.get("usage", {}),
        })

    return {
        "record_type": "qwen32b_final_round_rerun",
        "source_index": source_index,
        "uid": row.get("uid"),
        "data_source": row.get("data_source"),
        "question": row.get("question"),
        "ground_truth": row.get("ground_truth"),
        "ground_truth_targets": targets,
        "original_model_pair": row.get("model_pair"),
        "original_final_answer": row.get("final_answer"),
        "original_final_answer_source": row.get("final_answer_source"),
        "original_final_em": row.get("final_em"),
        "original_final_f1": row.get("final_f1"),
        "qwen32b_final_output": response_text,
        "qwen32b_final_answer": final_answer,
        "qwen32b_final_em": final_em,
        "qwen32b_final_f1": final_f1,
        "qwen32b_final_answer_llm_judge_score": judge_result.get("score"),
        "qwen32b_final_answer_llm_judge_is_correct": judge_result.get("is_correct"),
        "qwen32b_final_answer_llm_judge_response": judge_result.get("response", ""),
        "qwen32b_final_answer_llm_judge_error": judge_result.get("error"),
        "qwen32b_final_from_policy_round": final_from_round,
        "qwen32b_final_last_policy_output": last_policy_output if args.save_prompt_context else None,
        "qwen32b_final_messages": messages if args.save_prompt_context else None,
        "elapsed_seconds": round_seconds(time.perf_counter() - started),
        "token_usage": usage,
        "api_call_stats": api_stats,
        "error": error,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def summarize(output_path: Path, summary_path: Path, original_summary_path: Optional[Path]) -> dict[str, Any]:
    rows = list(iter_jsonl(output_path))
    ems = [float(r["qwen32b_final_em"]) for r in rows if r.get("qwen32b_final_em") is not None]
    f1s = [float(r["qwen32b_final_f1"]) for r in rows if r.get("qwen32b_final_f1") is not None]
    judges = [float(r["qwen32b_final_answer_llm_judge_score"]) for r in rows if r.get("qwen32b_final_answer_llm_judge_score") is not None]
    orig_ems = [float(r["original_final_em"]) for r in rows if r.get("original_final_em") is not None]
    orig_f1s = [float(r["original_final_f1"]) for r in rows if r.get("original_final_f1") is not None]
    errors = [r.get("error") for r in rows if r.get("error")]
    sources = Counter(str(r.get("data_source") or "<none>") for r in rows)
    token_usage_by_stage: dict[str, dict[str, Any]] = defaultdict(lambda: {"call_count": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    for r in rows:
        for call in r.get("api_call_stats") or []:
            stage = str(call.get("stage") or "<unknown>")
            usage = call.get("usage") or {}
            token_usage_by_stage[stage]["call_count"] += 1
            token_usage_by_stage[stage]["input_tokens"] += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
            token_usage_by_stage[stage]["output_tokens"] += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
            token_usage_by_stage[stage]["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    summary = {
        "output_path": str(output_path),
        "sample_count": len(rows),
        "scored_count": len(ems),
        "qwen32b_final_em_mean": statistics.fmean(ems) if ems else None,
        "qwen32b_final_f1_mean": statistics.fmean(f1s) if f1s else None,
        "qwen32b_final_answer_llm_judge_score_mean": statistics.fmean(judges) if judges else None,
        "qwen32b_final_answer_llm_judge_scored_count": len(judges),
        "original_final_em_mean_on_same_rows": statistics.fmean(orig_ems) if orig_ems else None,
        "original_final_f1_mean_on_same_rows": statistics.fmean(orig_f1s) if orig_f1s else None,
        "error_count": len(errors),
        "data_source_counts": dict(sources),
        "token_usage_by_stage": dict(token_usage_by_stage),
    }
    if original_summary_path and original_summary_path.exists():
        with original_summary_path.open("r", encoding="utf-8") as f:
            original = json.load(f)
        summary["original_summary"] = {
            "final_em_mean": original.get("final_em_mean"),
            "final_f1_mean": original.get("final_f1_mean"),
            "backbone_final_answer_llm_judge_score_mean": original.get("backbone_final_answer_llm_judge_score_mean"),
            "sample_count": original.get("sample_count"),
        }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--original-summary", type=Path, default=None)
    parser.add_argument("--env-file", default=".secrets/deepseek.env")
    parser.add_argument("--api-url", default="http://127.0.0.1:8022/v1")
    parser.add_argument("--model", default="Qwen3-32B-final")
    parser.add_argument("--api-key-env-var", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--extra-body-json", default='{"chat_template_kwargs":{"enable_thinking":false}}')
    parser.add_argument("--judge-extra-body-json", default='{"chat_template_kwargs":{"enable_thinking":false}}')
    parser.add_argument("--api-timeout", type=float, default=300.0)
    parser.add_argument("--api-max-retries", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backbone-judge-api-url", default="http://127.0.0.1:8022/v1")
    parser.add_argument("--backbone-judge-model", default="Qwen3-32B-final")
    parser.add_argument("--backbone-judge-api-key", default="")
    parser.add_argument("--backbone-judge-max-tokens", type=int, default=32)
    parser.add_argument("--no-proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-prompt-context", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    if not args.api_key:
        args.api_key = os.environ.get(args.api_key_env_var, "")
    if not args.backbone_judge_api_key:
        args.backbone_judge_api_key = os.environ.get("BACKBONE_JUDGE_API_KEY") or os.environ.get(args.api_key_env_var, "")
    args.backbone_judge_api_url = args.backbone_judge_api_url
    args.backbone_judge_model = args.backbone_judge_model
    args.backbone_judge_max_tokens = args.backbone_judge_max_tokens
    args.api_timeout = args.api_timeout
    args.api_max_retries = args.api_max_retries
    args.no_proxy = args.no_proxy

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = done_indices(args.output) if args.resume else set()
    rows = []
    for row in iter_jsonl(args.input):
        if args.resume and int(row.get("source_index")) in done:
            continue
        rows.append(row)
        if args.limit is not None and len(rows) >= args.limit:
            break
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"already_done={len(done)} processing={len(rows)} workers={args.num_workers} model={args.model}")

    lock = threading.Lock()
    if args.num_workers <= 1:
        for i, row in enumerate(rows, 1):
            rec = process_one(row, args)
            write_jsonl(args.output, rec, lock)
            if i % 10 == 0:
                print(f"processed={i}/{len(rows)}")
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futs = {ex.submit(process_one, row, args): row for row in rows}
            done_count = 0
            for fut in as_completed(futs):
                rec = fut.result()
                write_jsonl(args.output, rec, lock)
                done_count += 1
                if done_count % 10 == 0:
                    print(f"processed={done_count}/{len(rows)}")
    summary = summarize(args.output, args.summary, args.original_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
