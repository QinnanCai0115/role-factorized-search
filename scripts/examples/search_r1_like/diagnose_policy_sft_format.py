#!/usr/bin/env python3
"""Diagnose rollout-time XML format stability for a policy SFT checkpoint.

The script runs the policy model alone, with the search_subagent tool loop, and
reports format metrics such as first-turn search validity and final
answer/evidence validity. It is intentionally separate from PPO/GRPO so the
format behavior can be measured without RL rewards in the loop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from generate_sft_rollout import (
    POLICY_SYSTEM_PROMPT,
    build_correction_message,
    build_final_policy_turn_message,
    load_env_file,
    load_records,
    parse_search_tool_calls,
    run_search_tool,
    strip_tool_role_for_api,
)


DEFAULT_MODEL = "/ai/cqn/s3/ckpt/search_subagent_policy_sft/qwen3_1p7b_policy_sft_20260429_200022/merged_hf_global_step_250"
DEFAULT_INPUT = "/ai/cqn/datacon/data/hotpotqa_2wiki_musique_train/val_mixed_900.parquet"
DEFAULT_OUTPUT = "/ai/cqn/s3/ckpt/search_subagent_policy_sft_val/policy_sft_format_diagnostic.jsonl"
DEFAULT_RETRIEVAL_URL = "http://162.30.4.229:8765/search"


def iter_input_rows(path: str, *, limit: Optional[int], offset: int, recursive: bool) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.is_dir():
        pattern = "**/*.jsonl" if recursive else "*.jsonl"
        files = sorted(input_path.glob(pattern), key=lambda p: str(p))
        rows: list[dict[str, Any]] = []
        skipped = offset
        remaining = limit
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    if skipped > 0:
                        skipped -= 1
                        continue
                    row = json.loads(line)
                    row.setdefault("__input_file", str(file_path))
                    row.setdefault("__input_line", line_no)
                    rows.append(row)
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            return rows
        return rows
    return load_records(path, limit=limit, offset=offset)


def extract_last_user_from_messages(value: Any) -> str:
    if isinstance(value, str):
        try:
            return extract_last_user_from_messages(json.loads(value))
        except Exception:
            return value.strip()
    if isinstance(value, list):
        for msg in reversed(value):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def extract_policy_prompt(row: dict[str, Any], prompt_field: str) -> tuple[str, str]:
    """Return (prompt, source_field)."""
    if prompt_field != "auto":
        value = row.get(prompt_field)
        if prompt_field in {"messages", "sft_messages", "raw_prompt", "prompt"}:
            prompt = extract_last_user_from_messages(value)
        else:
            prompt = "" if value is None else str(value)
        return prompt.strip(), prompt_field

    candidates = [
        "policy_prompt",
        "policy_input",
        "backbone_search_query",
        "search_query",
        "query",
        "question",
    ]
    for key in candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), key

    for key in ("messages", "sft_messages", "raw_prompt", "prompt"):
        prompt = extract_last_user_from_messages(row.get(key))
        if prompt:
            return prompt.strip(), key

    return "", "<missing>"


def has_search(text: str) -> bool:
    return bool(re.search(r"<search>\s*.*?\s*</search>", text or "", flags=re.DOTALL | re.IGNORECASE))


def search_blocks(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"<search>\s*(.*?)\s*</search>", text or "", flags=re.DOTALL | re.IGNORECASE)
        if match.group(1).strip()
    ]


def search_only_status(text: str) -> tuple[bool, int, str]:
    blocks = search_blocks(text)
    residual = re.sub(r"<search>\s*.*?\s*</search>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    return len(blocks) == 1 and not residual, len(blocks), residual


def extract_answer_evidence_pair(text: str) -> str:
    pair_pattern = re.compile(
        r"(<answer>.*?</answer>\s*<evidence>.*?</evidence>)",
        flags=re.DOTALL | re.IGNORECASE,
    )
    pairs = pair_pattern.findall(str(text or "").strip())
    return pairs[-1].strip() if pairs else ""


def has_complete_answer_evidence(text: str) -> bool:
    return bool(extract_answer_evidence_pair(text))


def has_strict_answer_evidence(text: str) -> bool:
    merged = str(text or "").strip()
    if not merged:
        return False
    if re.search(r"</?(?:search|tool_call)\b", merged, flags=re.IGNORECASE):
        return False
    pair_pattern = re.compile(
        r"\A<answer>(.*?)</answer>\s*<evidence>(.*?)</evidence>\Z",
        flags=re.DOTALL | re.IGNORECASE,
    )
    match = pair_pattern.fullmatch(merged)
    if not match:
        return False
    answer, evidence = match.groups()
    return bool(answer.strip()) and bool(evidence.strip())


def response_flags(text: str) -> dict[str, Any]:
    strict_search_only, search_count, residual = search_only_status(text)
    complete_answer_evidence = has_complete_answer_evidence(text)
    strict_answer_evidence = has_strict_answer_evidence(text)
    return {
        "has_search": search_count > 0,
        "search_count": search_count,
        "strict_one_search_only": strict_search_only,
        "multiple_search_blocks": search_count > 1,
        "search_with_extra_text": search_count > 0 and bool(residual),
        "has_complete_answer_evidence": complete_answer_evidence,
        "has_strict_answer_evidence": strict_answer_evidence,
        "has_answer_evidence_tags": bool(re.search(r"</?answer>|</?evidence>", text or "", flags=re.IGNORECASE)),
        "residual_after_search": residual,
    }


class ChatGenerator:
    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class APIChatGenerator(ChatGenerator):
    def __init__(self, args: argparse.Namespace) -> None:
        from generate_sft_rollout import call_chat_completion

        self.args = args
        self._call_chat_completion = call_chat_completion

    def generate(self, messages: list[dict[str, str]]) -> str:
        text, _ = self._call_chat_completion(
            messages=messages,
            api_url=self.args.api_url,
            api_key=self.args.api_key,
            model=self.args.api_model,
            timeout=self.args.api_timeout,
            max_retries=self.args.api_max_retries,
            temperature=self.args.temperature,
            max_tokens=self.args.max_new_tokens,
            no_proxy=self.args.no_proxy,
        )
        return text


class TransformersChatGenerator(ChatGenerator):
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype: Any
        dtype_name = str(args.torch_dtype).lower()
        if dtype_name in {"auto", ""}:
            torch_dtype = "auto"
        elif dtype_name in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif dtype_name in {"fp16", "float16", "half"}:
            torch_dtype = torch.float16
        elif dtype_name in {"fp32", "float32"}:
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported --torch-dtype: {args.torch_dtype}")

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": args.trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        if args.device_map and args.device_map.lower() != "none":
            model_kwargs["device_map"] = args.device_map
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if not args.device_map or args.device_map.lower() == "none":
            self.model.to(args.device)
        self.model.eval()

    def _input_device(self):
        return next(self.model.parameters()).device

    def _apply_chat_template(self, messages: list[dict[str, str]]):
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        except Exception:
            converted = strip_tool_role_for_api(messages)
            return self.tokenizer.apply_chat_template(
                converted,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )

    def generate(self, messages: list[dict[str, str]]) -> str:
        torch = self.torch
        input_ids = self._apply_chat_template(messages).to(self._input_device())
        attention_mask = torch.ones_like(input_ids)
        do_sample = self.args.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.args.temperature
            gen_kwargs["top_p"] = self.args.top_p
        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)
        new_tokens = output_ids[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_generator(args: argparse.Namespace) -> ChatGenerator:
    if args.backend == "api":
        load_env_file(args.env_file)
        if not args.api_key:
            args.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not args.api_key:
            raise ValueError("Missing API key: set DEEPSEEK_API_KEY or pass --api-key.")
        return APIChatGenerator(args)
    if args.backend == "transformers":
        return TransformersChatGenerator(args)
    raise ValueError(f"Unsupported backend: {args.backend}")


def append_tool_message(messages: list[dict[str, str]], tool_text: str, role: str) -> None:
    if role == "tool":
        messages.append({"role": "tool", "name": "search_subagent", "content": tool_text})
    elif role == "user":
        messages.append({"role": "user", "content": f"<tool_response>\n{tool_text}\n</tool_response>"})
    else:
        raise ValueError(f"Unsupported --tool-response-role: {role}")


def run_policy_diagnostic(
    *,
    prompt: str,
    source_index: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    generator: ChatGenerator,
    retrieval_semaphore: Optional[Any],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": prompt},
    ]
    assistant_turns: list[dict[str, Any]] = []
    tool_trace: list[dict[str, Any]] = []
    invalid_turn_count = 0
    final_turn_reached = False
    final_turn_search_violation = False
    final_output = ""
    error = None

    try:
        for turn_idx in range(args.max_assistant_turns):
            is_final_turn = turn_idx == args.max_assistant_turns - 1
            if is_final_turn:
                final_turn_reached = True
                messages.append(build_final_policy_turn_message())

            response_started = time.perf_counter()
            assistant_text = generator.generate(messages)
            response_elapsed = round(time.perf_counter() - response_started, 6)
            flags = response_flags(assistant_text)
            assistant_turns.append(
                {
                    "turn": turn_idx + 1,
                    "is_final_turn": is_final_turn,
                    "response": assistant_text,
                    "flags": flags,
                    "elapsed_seconds": response_elapsed,
                }
            )
            messages.append({"role": "assistant", "content": assistant_text})

            if is_final_turn and flags["has_search"]:
                final_turn_search_violation = True

            if flags["has_search"]:
                tool_calls = parse_search_tool_calls(assistant_text)
                for call in tool_calls[: args.max_parallel_calls]:
                    query = str((call.get("arguments") or {}).get("query") or "").strip()
                    if not query:
                        messages.append(build_correction_message("search_subagent query is empty"))
                        invalid_turn_count += 1
                        continue
                    tool_started = time.perf_counter()
                    tool_text, tool_payload = run_search_tool(
                        query=query,
                        retrieval_url=args.retrieval_url,
                        topk=args.topk,
                        timeout=args.retrieval_timeout,
                        semaphore=retrieval_semaphore,
                        no_proxy=args.no_proxy,
                        save_raw_retrieval_response=args.save_raw_retrieval_response,
                    )
                    append_tool_message(messages, tool_text, args.tool_response_role)
                    tool_trace.append(
                        {
                            "turn": turn_idx + 1,
                            "query": query,
                            "status": tool_payload.get("status"),
                            "doc_count": len(tool_payload.get("docs") or []),
                            "doc_ids": [str(doc.get("doc_id")) for doc in (tool_payload.get("docs") or [])],
                            "elapsed_seconds": round(time.perf_counter() - tool_started, 6),
                        }
                    )
                continue

            if flags["has_complete_answer_evidence"]:
                final_output = assistant_text
                break

            invalid_turn_count += 1
            messages.append(build_correction_message("missing tool call or final <answer>/<evidence> blocks"))

        if not final_output and assistant_turns:
            final_output = assistant_turns[-1]["response"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    first_flags = assistant_turns[0]["flags"] if assistant_turns else {}
    last_flags = assistant_turns[-1]["flags"] if assistant_turns else {}
    final_strict = has_strict_answer_evidence(final_output)
    final_complete = has_complete_answer_evidence(final_output)
    strict_format_valid = (
        bool(first_flags.get("strict_one_search_only"))
        and final_strict
        and not final_turn_search_violation
        and invalid_turn_count == 0
        and not error
    )

    return {
        "record_type": "policy_sft_format_diagnostic",
        "source_index": source_index,
        "uid": row.get("uid") or row.get("source_uid") or row.get("id"),
        "data_source": row.get("data_source") or row.get("dataset"),
        "prompt": prompt,
        "prompt_source_field": row.get("__prompt_source_field"),
        "assistant_turn_count": len(assistant_turns),
        "tool_call_count": len(tool_trace),
        "first_turn_has_search": bool(first_flags.get("has_search")),
        "first_turn_strict_one_search_only": bool(first_flags.get("strict_one_search_only")),
        "first_turn_multiple_search_blocks": bool(first_flags.get("multiple_search_blocks")),
        "first_turn_search_with_extra_text": bool(first_flags.get("search_with_extra_text")),
        "first_turn_has_answer_evidence_tags": bool(first_flags.get("has_answer_evidence_tags")),
        "final_complete_answer_evidence": final_complete,
        "final_strict_answer_evidence": final_strict,
        "final_turn_reached": final_turn_reached,
        "final_turn_search_violation": final_turn_search_violation,
        "invalid_turn_count": invalid_turn_count,
        "strict_format_valid": strict_format_valid,
        "last_turn_flags": last_flags,
        "final_output": final_output,
        "tool_trace": tool_trace,
        "assistant_turns": assistant_turns if args.save_turns else [],
        "error": error,
        "elapsed_seconds": round(time.perf_counter() - started_at, 6),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def update_summary(summary: dict[str, Any], record: dict[str, Any]) -> None:
    summary["count"] += 1
    for key in [
        "first_turn_has_search",
        "first_turn_strict_one_search_only",
        "first_turn_multiple_search_blocks",
        "first_turn_search_with_extra_text",
        "first_turn_has_answer_evidence_tags",
        "final_complete_answer_evidence",
        "final_strict_answer_evidence",
        "final_turn_reached",
        "final_turn_search_violation",
        "strict_format_valid",
    ]:
        summary[key] += int(bool(record.get(key)))
    summary["error_count"] += int(bool(record.get("error")))
    summary["assistant_turn_sum"] += int(record.get("assistant_turn_count") or 0)
    summary["tool_call_sum"] += int(record.get("tool_call_count") or 0)
    summary["invalid_turn_sum"] += int(record.get("invalid_turn_count") or 0)


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    count = int(summary.get("count", 0))
    final_turn_reached = int(summary.get("final_turn_reached", 0))
    result = dict(summary)
    if count > 0:
        for key in [
            "first_turn_has_search",
            "first_turn_strict_one_search_only",
            "first_turn_multiple_search_blocks",
            "first_turn_search_with_extra_text",
            "first_turn_has_answer_evidence_tags",
            "final_complete_answer_evidence",
            "final_strict_answer_evidence",
            "final_turn_reached",
            "final_turn_search_violation",
            "strict_format_valid",
            "error_count",
        ]:
            result[f"{key}_rate"] = result.get(key, 0) / count
        result["avg_assistant_turns"] = result.get("assistant_turn_sum", 0) / count
        result["avg_tool_calls"] = result.get("tool_call_sum", 0) / count
        result["avg_invalid_turns"] = result.get("invalid_turn_sum", 0) / count
    result["final_turn_search_violation_rate_among_final_turn_reached"] = (
        result.get("final_turn_search_violation", 0) / final_turn_reached if final_turn_reached else None
    )
    return result


def default_summary_path(output: str) -> str:
    path = Path(output)
    suffix = "".join(path.suffixes)
    if suffix:
        return str(path)[: -len(suffix)] + ".summary.json"
    return output + ".summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input parquet/json/jsonl file or directory of jsonl files.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Per-sample diagnostic JSONL output.")
    parser.add_argument("--summary-output", default=None, help="Summary JSON path. Defaults beside --output.")
    parser.add_argument("--prompt-field", default="auto", help="Prompt field to use, or auto.")
    parser.add_argument("--recursive", action="store_true", help="When --input is a directory, read jsonl files recursively.")
    parser.add_argument("--limit", type=int, default=100, help="Number of prompts to run. Use -1 for all.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dedupe-prompts", action="store_true")
    parser.add_argument("--backend", choices=["transformers", "api"], default="transformers")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model/checkpoint path for --backend transformers.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--api-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-model", default="deepseek-reasoner")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--api-max-retries", type=int, default=3)
    parser.add_argument("--env-file", default="/ai/cqn/s3/.secrets/deepseek.env")
    parser.add_argument("--no-proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retrieval-url", default=DEFAULT_RETRIEVAL_URL)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retrieval-timeout", type=int, default=180)
    parser.add_argument("--retrieval-max-concurrent", type=int, default=64)
    parser.add_argument("--save-raw-retrieval-response", action="store_true")
    parser.add_argument("--max-assistant-turns", type=int, default=3)
    parser.add_argument("--max-parallel-calls", type=int, default=1)
    parser.add_argument("--tool-response-role", choices=["tool", "user"], default="tool")
    parser.add_argument("--system-prompt", default=POLICY_SYSTEM_PROMPT)
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--save-turns", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        args.limit = None
    if args.system_prompt_file:
        args.system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    if args.summary_output is None:
        args.summary_output = default_summary_path(args.output)

    rows = iter_input_rows(args.input, limit=args.limit, offset=args.offset, recursive=args.recursive)
    extracted: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for local_idx, row in enumerate(rows):
        prompt, source_field = extract_policy_prompt(row, args.prompt_field)
        if not prompt:
            continue
        if args.dedupe_prompts and prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        row = dict(row)
        row["__source_index"] = args.offset + local_idx
        row["__prompt_source_field"] = source_field
        extracted.append(row)

    print(f"Loaded {len(rows)} rows; running {len(extracted)} prompts from {args.input}")
    print(f"Model/backend: {args.backend} {args.model if args.backend == 'transformers' else args.api_model}")

    generator = build_generator(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    import threading

    retrieval_semaphore = (
        threading.BoundedSemaphore(args.retrieval_max_concurrent)
        if args.retrieval_max_concurrent and args.retrieval_max_concurrent > 0
        else None
    )

    overall = defaultdict(int)
    by_source: dict[str, Any] = defaultdict(lambda: defaultdict(int))
    records_written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(extracted):
            prompt, _source = extract_policy_prompt(row, args.prompt_field)
            record = run_policy_diagnostic(
                prompt=prompt,
                source_index=int(row.get("__source_index", i)),
                row=row,
                args=args,
                generator=generator,
                retrieval_semaphore=retrieval_semaphore,
            )
            out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            update_summary(overall, record)
            source_key = str(record.get("data_source") or "<missing>")
            update_summary(by_source[source_key], record)
            records_written += 1
            if records_written % 10 == 0 or records_written == len(extracted):
                print(f"processed={records_written}/{len(extracted)} output={output_path}")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "output": str(output_path),
        "backend": args.backend,
        "model": args.model if args.backend == "transformers" else args.api_model,
        "prompt_field": args.prompt_field,
        "max_assistant_turns": args.max_assistant_turns,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "summary": finalize_summary(overall),
        "by_data_source": {key: finalize_summary(value) for key, value in sorted(by_source.items())},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {records_written} records to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
