#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests


DEFAULT_INPUT = (
    "data/deepseek_policy_sft_rollouts/"
    "train_mixed_2000.deepseek_v4_pro.search_xml.valid_both.train2000.sft.jsonl"
)
DEFAULT_API_URL = "https://api.deepseek.com/v1"
DEFAULT_API_MODEL = "deepseek-reasoner"
DEFAULT_ENV_FILE = "/ai/cqn/s3/.secrets/deepseek.env"

JUDGE_SYSTEM = (
    "You are a strict binary judge for policy retrieval quality and evidence summarization quality. "
    "Return only JSON."
)

JUDGE_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "You are given the full policy chain from receiving the backbone search request to producing the final "
    "answer and evidence. Judge two things:\n"
    "1. retrieval_effective: whether the policy's search queries were reasonable and the retrieved content was "
    "useful for answering the question.\n"
    "2. summary_reasonable: whether the policy's final answer and evidence are faithful to and reasonably "
    "supported by the retrieved content.\n\n"
    "Set score=1 only if both retrieval_effective=1 and summary_reasonable=1. Otherwise set score=0.\n\n"
    "Return JSON in this exact schema:\n"
    '{{"score": 0 or 1, "retrieval_effective": 0 or 1, "summary_reasonable": 0 or 1, '
    '"reason": "short explanation"}}\n\n'
    "Policy chain JSON:\n"
    "{policy_chain_text}"
)


def load_env_file(path: str) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def default_output_path(input_path: str) -> str:
    path = Path(input_path)
    suffixes = "".join(path.suffixes)
    if suffixes:
        return str(path)[: -len(suffixes)] + ".backbone_judge_scores" + suffixes
    return str(path) + ".backbone_judge_scores.jsonl"


def truncate_text(text: Any, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "...(truncated)"


def truncate_tool_doc_text(text: Any, max_chars: int) -> str:
    return truncate_text(text, max_chars)


def truncate_structured_tool_response(
    tool_name: str,
    tool_response_text: str,
    *,
    max_tool_response_docs: int,
    max_tool_response_doc_chars: int,
    max_tool_response_length: int,
) -> Optional[str]:
    try:
        payload = json.loads(tool_response_text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    has_docs = isinstance(payload.get("docs"), list)
    round_results = payload.get("round_results")
    if not has_docs and isinstance(round_results, list):
        has_docs = any(isinstance(item, dict) and isinstance(item.get("docs"), list) for item in round_results)
    if tool_name != "search_subagent" and not has_docs:
        return None

    def truncate_doc(doc: dict[str, Any], max_chars: int) -> dict[str, Any]:
        return {
            "doc_id": str(doc.get("doc_id", "")),
            "title": truncate_tool_doc_text(doc.get("title", ""), max_chars),
            "snippet": truncate_tool_doc_text(doc.get("snippet", ""), max_chars),
            "url": truncate_tool_doc_text(doc.get("url", ""), max_chars),
            "score": doc.get("score"),
        }

    def truncate_doc_list(docs: Any, max_docs: int, max_chars: int) -> list[dict[str, Any]]:
        if not isinstance(docs, list):
            return []
        limit = len(docs) if max_docs <= 0 else max_docs
        return [truncate_doc(doc, max_chars) for doc in docs[:limit] if isinstance(doc, dict)]

    def build_payload(max_docs: int, max_chars: int) -> dict[str, Any]:
        truncated = dict(payload)
        if isinstance(truncated.get("docs"), list):
            truncated["docs"] = truncate_doc_list(truncated["docs"], max_docs, max_chars)
            if truncated["docs"]:
                truncated.pop("raw_result_text", None)
        if isinstance(truncated.get("round_results"), list):
            new_round_results: list[dict[str, Any]] = []
            for item in truncated["round_results"]:
                if not isinstance(item, dict):
                    continue
                new_item: dict[str, Any] = {}
                for key in ["round", "query", "status", "doc_count"]:
                    if key in item:
                        new_item[key] = item[key]
                if isinstance(item.get("docs"), list):
                    new_item["docs"] = truncate_doc_list(item["docs"], max_docs, max_chars)
                raw_result_text = item.get("raw_result_text")
                if raw_result_text and not new_item.get("docs"):
                    new_item["raw_result_text"] = truncate_tool_doc_text(raw_result_text, max_chars * 2)
                new_round_results.append(new_item)
            truncated["round_results"] = new_round_results
        raw_result_text = truncated.get("raw_result_text")
        if raw_result_text and not truncated.get("docs"):
            truncated["raw_result_text"] = truncate_tool_doc_text(raw_result_text, max_chars * 2)
        return truncated

    current_max_docs = max(1, int(max_tool_response_docs))
    current_max_chars = max(120, int(max_tool_response_doc_chars))
    text = json.dumps(build_payload(current_max_docs, current_max_chars), ensure_ascii=False)
    while len(text) > max_tool_response_length and (current_max_chars > 120 or current_max_docs > 1):
        if current_max_chars > 120:
            current_max_chars = max(120, current_max_chars // 2)
        elif current_max_docs > 1:
            current_max_docs -= 1
        text = json.dumps(build_payload(current_max_docs, current_max_chars), ensure_ascii=False)
    return text


def prepare_tool_response_for_backbone_judge(
    tool_name: str,
    tool_response_text: Any,
    *,
    backbone_judge_max_chars: int,
    max_tool_response_docs: int,
    max_tool_response_doc_chars: int,
    max_tool_response_length: int,
) -> str:
    text = str(tool_response_text or "").strip()
    if not text:
        return ""
    structured = truncate_structured_tool_response(
        tool_name,
        text,
        max_tool_response_docs=max_tool_response_docs,
        max_tool_response_doc_chars=max_tool_response_doc_chars,
        max_tool_response_length=max_tool_response_length,
    )
    if structured is not None:
        return structured
    return truncate_text(text, backbone_judge_max_chars)


def parse_search_from_assistant(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for query in re.findall(r"<search>\s*(.*?)\s*</search>", text or "", flags=re.DOTALL | re.IGNORECASE):
        query = query.strip()
        if query:
            calls.append({"name": "search_subagent", "arguments": {"query": query}})
    return calls


def build_policy_chain(record: dict[str, Any], args: argparse.Namespace) -> str:
    messages = record.get("messages") or record.get("sft_messages") or []
    if not isinstance(messages, list):
        messages = []

    initial_messages: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    assistant_turn = 0
    seen_first_user = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            if not seen_first_user and not trace:
                initial_messages.append({"role": "system", "content": content})
            continue
        if role == "user" and not seen_first_user and not trace:
            initial_messages.append({"role": "user", "content": content})
            seen_first_user = True
            continue
        if role == "assistant":
            assistant_turn += 1
            response_text = str(content or "")
            trace.append(
                {
                    "stage": "assistant_output",
                    "assistant_turn": assistant_turn,
                    "response_text": truncate_text(response_text, args.backbone_judge_max_chars),
                    "tool_calls": parse_search_from_assistant(response_text),
                }
            )
            continue
        if role == "tool":
            tool_name = str(msg.get("name") or "search_subagent")
            raw_tool_response_text = prepare_tool_response_for_backbone_judge(
                tool_name,
                content,
                backbone_judge_max_chars=args.backbone_judge_max_chars,
                max_tool_response_docs=args.max_tool_response_docs,
                max_tool_response_doc_chars=args.max_tool_response_doc_chars,
                max_tool_response_length=args.max_tool_response_length,
            )
            tool_args: Any = {}
            tool_trace = record.get("tool_trace")
            if isinstance(tool_trace, list):
                matching = [item for item in tool_trace if isinstance(item, dict) and item.get("turn") == assistant_turn]
                if matching:
                    tool_args = matching[0].get("arguments", {})
            trace.append(
                {
                    "stage": "tool_result",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "raw_tool_response_text": raw_tool_response_text,
                    "policy_visible_tool_response_text": raw_tool_response_text,
                }
            )
            continue
        if role == "user":
            trace.append(
                {
                    "stage": "final_policy_turn_instruction",
                    "assistant_turn": assistant_turn + 1,
                    "message": str(content or ""),
                    "token_count": None,
                }
            )

    if not initial_messages:
        policy_input = str(record.get("policy_input") or "")
        initial_messages = [{"role": "user", "content": policy_input}]

    payload = {
        "policy_prompt": initial_messages,
        "policy_trace": trace,
        "final_policy_output": truncate_text(record.get("final_output", ""), args.backbone_judge_max_chars),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def coerce_binary_judge_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("score", "binary_score", "final_backbone_binary_score", "backbone_binary_score"):
            score = coerce_binary_judge_score(value.get(key))
            if score is not None:
                return score
        raw = value.get("raw_judge_response")
        if raw is not None:
            return coerce_binary_judge_score(raw)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return coerce_binary_judge_score(json.loads(text))
        except Exception:
            match = re.search(r"\b([01])\b", text)
            if match:
                return float(match.group(1))
            try:
                return float(text)
            except ValueError:
                return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return float(1.0 if score > 0.5 else 0.0)


def has_strict_answer_evidence(final_output: Any) -> bool:
    text = str(final_output or "").strip()
    if not text:
        return False
    return bool(
        re.search(
            r"^\s*<answer>.*?</answer>\s*<evidence>.*?</evidence>\s*$",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )


def call_backbone_binary_judge(
    *,
    question: str,
    policy_chain_text: str,
    args: argparse.Namespace,
) -> tuple[Optional[float], dict[str, Any]]:
    judge_user = JUDGE_USER_TEMPLATE.format(
        question=question,
        policy_chain_text=policy_chain_text,
    )
    endpoint = f"{args.api_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    payload = {
        "model": args.api_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_user},
        ],
        "temperature": 0.0,
    }

    disable_backbone_proxy = str(os.environ.get("BACKBONE_API_NO_PROXY", "")).strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )
    with requests.Session() as session:
        if disable_backbone_proxy or args.no_proxy:
            session.trust_env = False
        resp = session.post(endpoint, json=payload, headers=headers, timeout=args.timeout)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", []) if isinstance(data, dict) else []
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content", "") if isinstance(message, dict) else ""

    details: dict[str, Any] = {"raw_judge_response": content}
    score = coerce_binary_judge_score(content)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                details.update(
                    {
                        "retrieval_effective": int(parsed.get("retrieval_effective", score or 0)),
                        "summary_reasonable": int(parsed.get("summary_reasonable", score or 0)),
                        "reason": str(parsed.get("reason", "")).strip(),
                    }
                )
        except Exception:
            pass
    return score, details


def score_record(line_no: int, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    question = str(record.get("policy_input") or record.get("question") or "")
    policy_chain_text = build_policy_chain(record, args)
    result = {
        "record_type": "policy_round_sft_backbone_judge_score",
        "source_file": str(args.input),
        "source_line": line_no,
        "policy_round_id": record.get("policy_round_id"),
        "source_index": record.get("source_index"),
        "round": record.get("round"),
        "uid": record.get("uid"),
        "question": record.get("question"),
        "policy_input": record.get("policy_input"),
        "final_output": record.get("final_output", ""),
        "has_answer_evidence": record.get("has_answer_evidence"),
        "tool_call_count": record.get("tool_call_count"),
        "api_model": record.get("api_model"),
        "backbone_model": record.get("backbone_model"),
        "judge_api_model": args.api_model,
        "judge_api_url": args.api_url,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        score, details = call_backbone_binary_judge(question=question, policy_chain_text=policy_chain_text, args=args)
        details = details or {}
        details["policy_chain"] = policy_chain_text
        details["policy_full_trace_output"] = policy_chain_text
        format_penalty = 0.0 if has_strict_answer_evidence(record.get("final_output", "")) else float(args.policy_format_penalty)
        final_score = 0.0 if score is None else float(1.0 if score > 0.5 else 0.0)
        result.update(
            {
                "final_backbone_binary_score": final_score,
                "backbone_judge_binary_score": final_score,
                "policy_format_penalty": format_penalty,
                "final_policy_reward": final_score + format_penalty,
                "final_backbone_binary_details": details,
                "backbone_judge_error": None,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
        )
    except Exception as exc:
        details = {
            "error": f"{type(exc).__name__}: {exc}",
            "policy_chain": policy_chain_text,
            "policy_full_trace_output": policy_chain_text,
        }
        result.update(
            {
                "final_backbone_binary_score": 0.0,
                "backbone_judge_binary_score": 0.0,
                "policy_format_penalty": float(args.policy_format_penalty),
                "final_policy_reward": float(args.policy_format_penalty),
                "final_backbone_binary_details": details,
                "backbone_judge_error": details["error"],
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
        )
    return result


def load_done_ids(path: str) -> set[str]:
    done: set[str] = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = str(obj.get("policy_round_id") or obj.get("source_line") or "")
            if key:
                done.add(key)
    return done


def iter_records(path: str, *, limit: Optional[int], done: set[str], resume: bool) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = str(obj.get("policy_round_id") or line_no)
            if resume and key in done:
                continue
            records.append((line_no, obj))
    return records


def write_jsonl(path: str, record: dict[str, Any], lock: threading.Lock) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score policy SFT rollout JSONL with the training backbone binary judge.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--env_file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--api_url", default=os.environ.get("BACKBONE_API_URL", DEFAULT_API_URL))
    parser.add_argument("--api_model", default=os.environ.get("BACKBONE_API_MODEL", DEFAULT_API_MODEL))
    parser.add_argument("--api_key", default=os.environ.get("BACKBONE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("BACKBONE_JUDGE_TIMEOUT", "120")))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backbone_judge_max_chars", type=int, default=4000)
    parser.add_argument("--max_tool_response_length", type=int, default=4096)
    parser.add_argument("--max_tool_response_docs", type=int, default=3)
    parser.add_argument("--max_tool_response_doc_chars", type=int, default=1024)
    parser.add_argument("--policy_format_penalty", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if not args.api_key:
        args.api_key = os.environ.get("BACKBONE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not args.api_key:
        raise ValueError("Missing API key: set BACKBONE_API_KEY/DEEPSEEK_API_KEY or pass --api_key.")
    if args.output is None:
        args.output = default_output_path(args.input)

    done = load_done_ids(args.output) if args.resume else set()
    records = iter_records(args.input, limit=args.limit, done=done, resume=args.resume)
    print(f"loaded={len(records)} skipped_existing={len(done)} output={args.output}", flush=True)
    if not records:
        return

    write_lock = threading.Lock()
    completed = 0
    score_sum = 0.0
    error_count = 0
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = [executor.submit(score_record, line_no, record, args) for line_no, record in records]
        for future in as_completed(futures):
            result = future.result()
            write_jsonl(args.output, result, write_lock)
            completed += 1
            score_sum += float(result.get("final_backbone_binary_score", 0.0) or 0.0)
            if result.get("backbone_judge_error"):
                error_count += 1
            if completed == 1 or completed % 10 == 0 or completed == len(records):
                mean_score = score_sum / completed
                print(
                    f"scored={completed}/{len(records)} mean_score={mean_score:.4f} errors={error_count}",
                    flush=True,
                )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
