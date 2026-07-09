#!/usr/bin/env python3
"""Generate API-only two-stage policy SFT rollouts.

This script does not start verl PPO, does not load the Qwen policy checkpoint,
and does not initialize a local rollout engine. It keeps the same high-level
two-stage shape as the verl orchestrator:

1. a backbone model emits <search>...</search> requests or final answers;
2. each search request, or each query in a JSON-style search list, is handled by a policy model that calls search_subagent;
3. each policy round is dumped as one SFT record.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import string
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from uuid import uuid4

import requests

from verl.utils.two_stage_prompts import (
    BACKBONE_SYSTEM_PROMPT as SHARED_BACKBONE_SYSTEM_PROMPT,
    POLICY_SYSTEM_PROMPT as SHARED_POLICY_SYSTEM_PROMPT,
    build_final_backbone_message as shared_build_final_backbone_message,
    build_final_policy_turn_message as shared_build_final_policy_turn_message,
    build_initial_backbone_messages as shared_build_initial_backbone_messages,
    build_next_backbone_message as shared_build_next_backbone_message,
    build_policy_correction_message as shared_build_policy_correction_message,
    build_policy_failure_backbone_message as shared_build_policy_failure_backbone_message,
)


DEFAULT_API_URL = "http://127.0.0.1:8000/v1"
DEFAULT_POLICY_API_URL = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "Qwen3-1.7B"
DEFAULT_BACKBONE_MODEL = "Qwen3-32B"
DEFAULT_BACKBONE_JUDGE_API_URL = "https://api.deepseek.com/v1"
DEFAULT_BACKBONE_JUDGE_MODEL = "deepseek-reasoner"
DEFAULT_INPUT = "/ai/cqn/datacon/data/hotpotqa_2wiki_musique_train/train_mixed_2000_sft.jsonl"
DEFAULT_OUTPUT = "/ai/cqn/datacon/data/qwen3_policy_sft_rollouts/train_mixed_2000.qwen3_32b_backbone.qwen3_1p7b_policy.sft.jsonl"
DEFAULT_RETRIEVAL_URL = "http://162.30.4.229:8765/search"
DEFAULT_BACKBONE_MODEL_PATH = "/ai/zjm/Models/Qwen3-32B/"
DEFAULT_POLICY_MODEL_PATH = "/ai/cqn/model/Qwen3-1.7B/"

VLLM_STARTUP_HELP = (
    "Local vLLM defaults:\n"
    "  Backbone server:\n"
    f"    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model {DEFAULT_BACKBONE_MODEL_PATH} "
    f"--served-model-name {DEFAULT_BACKBONE_MODEL} --host 0.0.0.0 --port 8000 --tensor-parallel-size 2 "
    "--gpu-memory-utilization 0.7\n"
    "\n"
    "  Policy server:\n"
    f"    CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server --model {DEFAULT_POLICY_MODEL_PATH} "
    f"--served-model-name {DEFAULT_MODEL} --host 0.0.0.0 --port 8001 --gpu-memory-utilization 0.7\n"
    "\n"
    "If you launch vLLM with a different --served-model-name, pass the same value via\n"
    "--backbone_model or --policy_model.\n"
)
BACKBONE_SYSTEM_PROMPT = """Please answer the question.

You are the backbone model in a two-stage question-answering system.

Your job is to:
1. Understand the original question.
2. Identify the missing factual evidence.
3. Decompose the question into atomic evidence requests.
4. Use <search> only to ask for those missing facts.
5. Produce the final answer yourself after enough evidence is available.

Do not delegate the original question to the search subagent.
The search subagent will return evidence for your specific search requests, but the final reasoning and final answer are your responsibility.

Output format:

If you already have enough evidence, output only:
<final answer>...</final answer>

Otherwise, output exactly one <search>...</search> block.

A <search> block should contain either:
- one concise, specific, answerable natural-language question; or
- a JSON-style list of such questions, if multiple independent facts are needed.

Search decomposition rules:

- Never copy or lightly rewrite the whole original question into <search>.
- Ask only for missing atomic facts needed to answer the original question.
- Each search question should usually focus on one entity and one attribute, relation, date, location, event, or fact.
- For comparison questions involving multiple entities, ask one focused question per entity.
- For multi-hop questions, ask for the next missing bridge fact first; after evidence is returned, ask the next focused follow-up if needed.
- Use the exact entity names and requested relations from the original question or retrieved evidence.
- Do not add guesses, candidate answers, unsupported locations, dates, aliases, categories, near-synonyms, or alternate entities.
- Do not rewrite an entity into a different entity.
- Do not ask the subagent to answer the final comparison, judgment, counting, temporal ordering, or multi-hop question directly.
- If multiple independent facts are needed, put them inside one single <search> block as a JSON-style list.
- Do not output multiple <search> blocks in the same assistant message.

Final answer style:

- Use short-answer QA style.
- The final answer should usually be one short phrase or one short sentence.
- Do not include explanations, evidence, citations, reasoning steps, or background.
- Do not restate the retrieved evidence.
- Answer only the original question.
- For entity/date/place/country questions, output only the answer value when possible.
- For yes/no questions, start with "Yes" or "No" and include only the minimal fact needed.

You may receive search results in this format:

<search_results>
<result index="0">
<request>...</request>
<answer>...</answer>
<evidence>...</evidence>
</result>
</search_results>

Use the returned answer and evidence to decide whether to output a final answer or issue another focused <search>.

Complete two-stage example 1:

Original question:
Are both rivers, Turkey Ridge Creek and Diamond Brook, located in the same country?

Correct first backbone output:
<search>
["Which country is Turkey Ridge Creek located in?", "Which country is Diamond Brook located in?"]
</search>

Search results:
<search_results>
<result index="0">
<request>Which country is Turkey Ridge Creek located in?</request>
<answer>Turkey Ridge Creek is located in the United States.</answer>
<evidence>Turkey Ridge Creek is a stream in the U.S. state of South Dakota.</evidence>
</result>
<result index="1">
<request>Which country is Diamond Brook located in?</request>
<answer>Diamond Brook is located in the United States.</answer>
<evidence>Diamond Brook is a tributary of the Passaic River in Bergen County, New Jersey, United States.</evidence>
</result>
</search_results>

Correct next backbone output:
<final answer>Yes</final answer>

Incorrect first backbone output:
<search>Are Turkey Ridge Creek and Diamond Brook located in the same country?</search>

Why incorrect:
This asks for the final comparison instead of the missing atomic facts.

Incorrect first backbone output:
<search>
Find the countries where Turkey Ridge Creek and Diamond Brook are located. Turkey Ridge Creek is likely in Arkansas. Diamond Brook is possibly in Vermont.
</search>

Why incorrect:
This adds unsupported candidate locations and mixes reasoning with retrieval.

Complete two-stage example 2:

Original question:
What agreement did the country Niulakita is located in commit to?

Correct first backbone output:
<search>Which country is Niulakita located in?</search>

Search results:
<search_results>
<result index="0">
<request>Which country is Niulakita located in?</request>
<answer>Niulakita is located in Tuvalu.</answer>
<evidence>Niulakita is an island of Tuvalu.</evidence>
</result>
</search_results>

Correct next backbone output:
<search>What agreement did Tuvalu commit to?</search>

Search results:
<search_results>
<result index="0">
<request>What agreement did Tuvalu commit to?</request>
<answer>Tuvalu committed to the Majuro Declaration.</answer>
<evidence>Tuvalu is listed as a country that committed to the Majuro Declaration.</evidence>
</result>
</search_results>

Correct final backbone output:
<final answer>Majuro Declaration</final answer>

Incorrect second backbone output:
<search>What international agreements has Tuvalu committed to?</search>

Why incorrect:
The original question asks for a specific agreement, not a broad list of international agreements.

Incorrect second backbone output:
<search>What treaties has Tuvalu ratified?</search>

Why incorrect:
It changes the requested relation from "committed to an agreement" to the different relation "ratified treaties".
"""

POLICY_SYSTEM_PROMPT = (
    "Policy agent rules: You are a tool-calling policy model. "
    "For factual or open-domain questions, you MUST call the search tool on the first assistant turn before giving any final answer. "

    "Search format rules: When calling the search tool, output EXACTLY ONE XML search block and nothing else: "
    "<search>query</search>. "
    "The query must be a single retrieval request. "
    "It may be a natural-language question or a compact search query. "
    "Keep all key entities, relations, dates, locations, and disambiguating constraints from the current request. "
    "Do NOT drop qualifiers that distinguish the target entity from similarly named entities. "
    "Do NOT broaden the request beyond the current evidence objective. "
    "Prefer preserving the current request when it is already concise and searchable. "
    "Do NOT output multiple <search> blocks in the same assistant turn. "
    "Do NOT output a list of queries, numbered queries, JSON, explanations, thoughts, or plain text in the same turn as a <search> block. "
    "Each assistant turn may contain at most one search query. "

    "After receiving raw tool results, carefully decide whether more retrieval is needed before answering. "
    "If the retrieved evidence is insufficient to answer the current request, you MUST output another single <search>...</search> query. "
    "If the retrieved evidence is conflicting, ambiguous, incomplete, or does not mention the key entities in the request, "
    "you MUST output another single <search>...</search> query to clarify. "

    "You may stop searching only when the retrieved documents directly support the answer to the current request. "
    "Do NOT answer from memory, prior knowledge, assumptions, or unstated background information. "
    "The final answer and evidence MUST be strictly grounded in the retrieved documents. "
    "Do NOT add details that are not explicitly present in the retrieved documents. "
    "Do NOT infer dates, locations, names, relationships, or explanations unless they are directly supported by the retrieved documents. "
    "Simple extraction from explicit evidence is allowed, but unsupported inference is not. "

    "Final answer format: Only when the retrieved documents directly support the answer, output exactly two XML blocks in this order: "
    "<answer>...</answer><evidence>...</evidence>. "
    "<answer> must be concise and directly answer the current request using only information supported by the retrieved documents. "
    "Do NOT output an <answer> that merely says the documents do not contain enough information. "
    "<evidence> must contain only 1-3 short evidence points copied or tightly paraphrased from the retrieved documents. "
    "Each evidence point must support the answer directly. "
    "If the retrieved documents do not contain enough evidence, do not guess; issue another single <search>...</search> query instead. "
    "Never dump raw JSON, full retrieved passages, irrelevant snippets, or unsupported details."
)

# Keep the standalone rollout generator and verl validation on the exact same prompt text.
BACKBONE_SYSTEM_PROMPT = SHARED_BACKBONE_SYSTEM_PROMPT
POLICY_SYSTEM_PROMPT = SHARED_POLICY_SYSTEM_PROMPT


def load_env_file(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = os.path.expandvars(value.strip().strip("'\""))
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_api_key(explicit_key: str, env_var: str, fallback_env_vars: tuple[str, ...] = ()) -> str:
    key = str(explicit_key or "").strip()
    if key:
        return key
    candidates = [str(env_var or "").strip(), *fallback_env_vars]
    for candidate in candidates:
        if candidate and os.environ.get(candidate):
            return os.environ[candidate]
    return ""


def is_local_api_url(api_url: str) -> bool:
    host = (urlparse(str(api_url or "")).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    return str(value)


def load_records(path: str, limit: Optional[int] = None, offset: int = 0) -> list[dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        try:
            rows = pd.read_parquet(path).to_dict(orient="records")
        except ImportError as exc:
            raise ImportError(
                "Reading parquet requires pyarrow or fastparquet. Run this script in the verl/data environment "
                "or install one of those parquet engines."
            ) from exc
    elif suffix in {".jsonl", ".json"}:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            if suffix == ".json":
                data = json.load(f)
                rows = data if isinstance(data, list) else data.get("data", data.get("records", []))
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported input file extension: {suffix}")

    rows = [to_jsonable(row) for row in rows]
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return normalize_messages(parsed)
        except Exception:
            return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        role = str(value.get("role", "user"))
        content = value.get("content", "")
        return [{"role": role, "content": json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content}]
    if isinstance(value, list):
        messages: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = item.get("content", "")
                messages.append(
                    {
                        "role": role,
                        "content": json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content,
                    }
                )
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages
    return []


def extract_question(row: dict[str, Any]) -> str:
    for key in ("question", "query", "input", "prompt", "problem"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("raw_prompt", "messages"):
        messages = normalize_messages(row.get(key))
        for msg in reversed(messages):
            if msg.get("role") == "user" and str(msg.get("content", "")).strip():
                return str(msg["content"]).strip()
    return ""


def extract_ground_truth(row: dict[str, Any]) -> Any:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        for key in ("ground_truth", "target", "answer", "answers"):
            value = reward_model.get(key)
            if value not in (None, "", []):
                return value
    for key in ("ground_truth", "target", "answer", "answers", "golden_answers", "gt", "gts"):
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return None


def round_seconds(value: float) -> float:
    return round(float(value), 6)


def flatten_numeric_usage(value: Any, prefix: str = "") -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}

    usage: dict[str, int | float] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            usage[name] = int(item)
        elif isinstance(item, float):
            usage[name] = float(item)
        elif isinstance(item, dict):
            usage.update(flatten_numeric_usage(item, name))
    return usage


def extract_token_usage(raw_response: Any) -> dict[str, int | float]:
    if not isinstance(raw_response, dict):
        return {}
    return flatten_numeric_usage(raw_response.get("usage"))


def new_token_usage() -> dict[str, Any]:
    return {"call_count": 0, "by_model": {}}


def add_token_usage(
    aggregate: dict[str, Any],
    usage: dict[str, int | float],
    model: Optional[str] = None,
) -> None:
    aggregate["call_count"] = int(aggregate.get("call_count", 0)) + 1
    for key, value in usage.items():
        aggregate[key] = aggregate.get(key, 0) + value

    if model:
        by_model = aggregate.setdefault("by_model", {})
        model_usage = by_model.setdefault(str(model), {"call_count": 0})
        model_usage["call_count"] = int(model_usage.get("call_count", 0)) + 1
        for key, value in usage.items():
            model_usage[key] = model_usage.get(key, 0) + value


def merge_token_usage(aggregate: dict[str, Any], other: Any) -> None:
    if not isinstance(other, dict):
        return

    aggregate["call_count"] = int(aggregate.get("call_count", 0)) + int(other.get("call_count", 0))
    for key, value in other.items():
        if key in {"call_count", "by_model"} or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        aggregate[key] = aggregate.get(key, 0) + value

    other_by_model = other.get("by_model", {})
    if not isinstance(other_by_model, dict):
        return
    by_model = aggregate.setdefault("by_model", {})
    for model, model_other in other_by_model.items():
        if not isinstance(model_other, dict):
            continue
        model_usage = by_model.setdefault(str(model), {"call_count": 0})
        model_usage["call_count"] = int(model_usage.get("call_count", 0)) + int(model_other.get("call_count", 0))
        for key, value in model_other.items():
            if key == "call_count" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            model_usage[key] = model_usage.get(key, 0) + value


def add_api_response_usage(
    aggregate: dict[str, Any],
    raw_response: Any,
    model: str,
) -> dict[str, int | float]:
    usage = extract_token_usage(raw_response)
    add_token_usage(aggregate, usage, model)
    return usage


def collect_ground_truth_targets(value: Any) -> list[str]:
    targets: list[str] = []

    def add(candidate: Any) -> None:
        if candidate is None:
            return
        if isinstance(candidate, str) and not candidate.strip():
            return
        if isinstance(candidate, (list, tuple, set)) and not candidate:
            return
        if isinstance(candidate, dict):
            for key in ("target", "answer", "answers", "golden_answers", "ground_truth", "gt", "gts"):
                if key in candidate:
                    add(candidate[key])
            return
        if isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                add(item)
            return
        text = str(candidate).strip()
        if text:
            targets.append(text)

    add(value)
    return targets


def normalize_answer_for_score(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match_score(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer_for_score(prediction) == normalize_answer_for_score(gold) else 0.0


def token_f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer_for_score(prediction).split()
    gold_tokens = normalize_answer_for_score(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_final_scores(final_answer: str, ground_truth: Any) -> tuple[Optional[float], Optional[float], list[str]]:
    targets = collect_ground_truth_targets(ground_truth)
    if not targets:
        return None, None, targets
    if not str(final_answer or "").strip():
        return 0.0, 0.0, targets
    final_em = max(exact_match_score(final_answer, target) for target in targets)
    final_f1 = max(token_f1_score(final_answer, target) for target in targets)
    return float(final_em), float(final_f1), targets


BACKBONE_FINAL_ANSWER_JUDGE_PROMPT = """Given a Question and its Golden Answer, verify whether the Predicted Answer is correct.
The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer.
Respond with True if the prediction is correct and False otherwise.

Question:
{question}
Golden Answer:
{golden_answer}
Predicted Answer:
{predicted_answer}"""


def parse_bool_judge_response(text: Any) -> Optional[bool]:
    normalized = str(text or "").strip().lower()
    if re.fullmatch(r"true[\s\.\!]*", normalized):
        return True
    if re.fullmatch(r"false[\s\.\!]*", normalized):
        return False

    match = re.search(r"\b(true|false)\b", normalized)
    if not match:
        return None
    return match.group(1) == "true"


def judge_backbone_final_answer(
    *,
    question: str,
    ground_truth_targets: list[str],
    predicted_answer: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not question or not ground_truth_targets or not str(predicted_answer or "").strip():
        return {
            "score": None,
            "is_correct": None,
            "response": "",
            "error": "missing question, golden answer, or predicted answer",
            "usage": {},
            "elapsed_seconds": None,
        }

    golden_answer_text = (
        ground_truth_targets[0]
        if len(ground_truth_targets) == 1
        else json.dumps(ground_truth_targets, ensure_ascii=False)
    )
    prompt = BACKBONE_FINAL_ANSWER_JUDGE_PROMPT.format(
        question=question,
        golden_answer=golden_answer_text,
        predicted_answer=str(predicted_answer).strip(),
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
            max_tokens=int(getattr(args, "backbone_judge_max_tokens", 16) or 16),
            no_proxy=args.no_proxy,
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
        return {
            "score": None,
            "is_correct": None,
            "response": "",
            "error": f"{type(exc).__name__}: {exc}",
            "usage": {},
            "elapsed_seconds": round_seconds(time.perf_counter() - started_at),
        }


def strip_tool_role_for_api(
    messages: list[dict[str, Any]],
    *,
    preserve_reasoning_content: bool = False,
) -> list[dict[str, Any]]:
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role == "tool":
            api_messages.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})
        elif role in {"system", "user", "assistant"}:
            api_message: dict[str, Any] = {"role": role, "content": content}
            reasoning_content = str(msg.get("reasoning_content") or "").strip()
            if preserve_reasoning_content and role == "assistant" and reasoning_content:
                api_message["reasoning_content"] = reasoning_content
            api_messages.append(api_message)
        else:
            api_messages.append({"role": "user", "content": content})
    return api_messages


def parse_json_object(text: str, *, field_name: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return parsed


def extract_chat_message_fields(raw_response: Any) -> dict[str, Any]:
    if not isinstance(raw_response, dict):
        return {"content": "", "reasoning_content": ""}
    choices = raw_response.get("choices", [])
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return {"content": "", "reasoning_content": ""}

    content = message.get("content", "")
    if content is None:
        content = ""
    reasoning_content = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("reasoning_text")
        or ""
    )
    if reasoning_content is None:
        reasoning_content = ""

    reasoning_details = message.get("reasoning_details")
    return {
        "content": str(content),
        "reasoning_content": str(reasoning_content),
        "reasoning_details": reasoning_details if reasoning_details is not None else None,
        "message": to_jsonable(message),
    }


def call_chat_completion(
    *,
    messages: list[dict[str, str]],
    api_url: str,
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
    temperature: float,
    max_tokens: int,
    no_proxy: bool,
    extra_body: Optional[dict[str, Any]] = None,
    preserve_reasoning_content: bool = False,
    retry_on_empty_content: bool = False,
) -> tuple[str, dict[str, Any]]:
    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": strip_tool_role_for_api(messages, preserve_reasoning_content=preserve_reasoning_content),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            with requests.Session() as session:
                if no_proxy:
                    session.trust_env = False
                response = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            message_fields = extract_chat_message_fields(data)
            content = str(message_fields.get("content", ""))
            if retry_on_empty_content and not content.strip():
                raise RuntimeError("empty chat completion content")
            return content, data
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 8))
    raise last_error if last_error is not None else RuntimeError("Unknown API error")


def parse_search_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"<search>\s*(.*?)\s*</search>", text or "", flags=re.DOTALL | re.IGNORECASE):
        query = match.group(1).strip()
        if query:
            calls.append({"name": "search_subagent", "arguments": {"query": query}})
    return calls


def parse_backbone_search(text: str) -> Optional[str]:
    matches = list(re.finditer(r"<search>\s*(.*?)\s*</search>", text or "", flags=re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    query = matches[-1].group(1).strip()
    return query or None


def normalize_search_query_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        queries: list[str] = []
        for item in value:
            queries.extend(normalize_search_query_list(item))
        return queries
    return []


def strip_list_item_marker(line: str) -> str:
    text = line.strip().rstrip(",").strip()
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def parse_backbone_search_queries(search_block: Optional[str]) -> list[str]:
    text = (search_block or "").strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        queries = normalize_search_query_list(parsed)
        if queries:
            return queries

    lines = [strip_list_item_marker(line) for line in text.splitlines()]
    line_queries = [line for line in lines if line]
    if len(line_queries) > 1:
        return line_queries
    return [text]


def extract_answer_evidence_blocks(text: str) -> str:
    matches = list(
        re.finditer(
            r"(?:(<think>\s*.*?\s*</think>)\s*)?<answer>\s*(.*?)\s*</answer>\s*<evidence>\s*(.*?)\s*</evidence>",
            text or "",
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    for match in reversed(matches):
        answer = match.group(2).strip()
        evidence = match.group(3).strip()
        if answer and evidence:
            return f"<answer>{answer}</answer>\n<evidence>{evidence}</evidence>"
    return ""


def has_final_answer(text: str) -> bool:
    return bool(extract_answer_evidence_blocks(text))


def extract_final_answer(text: str) -> str:
    matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text or "", flags=re.DOTALL | re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()
    return str(text or "").strip()


def extract_backbone_final_answer(text: str) -> Optional[str]:
    matches = list(
        re.finditer(r"<final[_ ]answer>\s*(.*?)\s*</final[_ ]answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    )
    if not matches:
        return None
    answer = matches[-1].group(1).strip()
    return answer or None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None).strip()
    return str(value).strip()


def normalize_score(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_structured_doc(doc_item: Any, fallback_doc_id: str) -> dict[str, Any]:
    doc_dict = doc_item if isinstance(doc_item, dict) else {}
    document = doc_dict.get("document", {})
    if not isinstance(document, dict):
        document = {}

    contents = normalize_text(document.get("contents") or doc_dict.get("contents"))
    title = normalize_text(document.get("title") or doc_dict.get("title"))
    snippet = normalize_text(document.get("snippet") or doc_dict.get("snippet"))

    if contents:
        lines = [line.strip() for line in contents.splitlines()]
        non_empty = [line for line in lines if line]
        if non_empty:
            if not title:
                title = non_empty[0]
                snippet = "\n".join(non_empty[1:]).strip()
            elif not snippet:
                snippet = "\n".join(non_empty[1:] if non_empty[0] == title else non_empty).strip()

    doc_id = (
        document.get("doc_id")
        or document.get("id")
        or doc_dict.get("doc_id")
        or doc_dict.get("id")
        or fallback_doc_id
    )
    url = document.get("url") or document.get("source_url") or doc_dict.get("url") or doc_dict.get("source_url") or ""
    score = normalize_score(
        doc_dict.get("score")
        or doc_dict.get("retrieval_score")
        or document.get("score")
        or document.get("retrieval_score")
    )
    return {
        "doc_id": str(doc_id),
        "title": title,
        "snippet": snippet,
        "url": normalize_text(url),
        "score": score,
    }


def extract_docs_from_retrieval_response(response_json: Any) -> list[dict[str, Any]]:
    payload = response_json
    if isinstance(payload, dict):
        for key in ("docs", "documents", "result", "results", "data", "passages"):
            if key in payload:
                payload = payload[key]
                break

    # Batched retrievers often return one list per query. This script sends one query.
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        payload = payload[0]
    if isinstance(payload, dict):
        for key in ("docs", "documents", "result", "results", "data", "passages"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return []

    return [extract_structured_doc(item, fallback_doc_id=str(i)) for i, item in enumerate(payload)]


def call_retrieval_service(
    *,
    retrieval_url: str,
    query: str,
    topk: int,
    timeout: int,
    no_proxy: bool,
) -> tuple[list[dict[str, Any]], Optional[str], Any]:
    payload = {"query_list": [query], "k": topk}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        with requests.Session() as session:
            if no_proxy:
                session.trust_env = False
            response = session.post(retrieval_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        response_json = response.json()
        docs = extract_docs_from_retrieval_response(response_json)
        return docs, None, response_json
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}", None


def run_search_tool(
    *,
    query: str,
    retrieval_url: str,
    topk: int,
    timeout: int,
    semaphore: Optional[threading.BoundedSemaphore],
    no_proxy: bool,
    save_raw_retrieval_response: bool,
) -> tuple[str, dict[str, Any]]:
    if semaphore is None:
        docs, error, raw_response = call_retrieval_service(
            retrieval_url=retrieval_url,
            query=query,
            topk=topk,
            timeout=timeout,
            no_proxy=no_proxy,
        )
    else:
        with semaphore:
            docs, error, raw_response = call_retrieval_service(
                retrieval_url=retrieval_url,
                query=query,
                topk=topk,
                timeout=timeout,
                no_proxy=no_proxy,
            )
    status = "error" if error else "success"
    payload = {"query": query, "status": status, "docs": docs}
    if error:
        payload["error"] = error
    if save_raw_retrieval_response:
        payload["raw_response"] = raw_response
    return json.dumps(payload, ensure_ascii=False), payload


def build_correction_message(reason: str) -> dict[str, str]:
    return shared_build_policy_correction_message(reason)


def build_final_policy_turn_message() -> dict[str, str]:
    return shared_build_final_policy_turn_message()


def build_initial_backbone_messages(question: str) -> list[dict[str, str]]:
    return shared_build_initial_backbone_messages(question)


def escape_xml_text(text: Any) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_policy_result_for_backbone(search_query: str, policy_result: dict[str, Any]) -> str:
    answer_evidence = extract_answer_evidence_blocks(str(policy_result.get("final_output", "")))
    if answer_evidence:
        return answer_evidence

    error = str(policy_result.get("error") or "").strip()
    detail = "The policy model did not produce a valid answer/evidence result."
    if error:
        detail = f"{detail} Error: {error}"
    return (
        "<evidence_unavailable>"
        f"{escape_xml_text(detail)} Request: {escape_xml_text(search_query)}"
        "</evidence_unavailable>"
    )


def build_parallel_policy_results_for_backbone(policy_runs: list[dict[str, Any]]) -> str:
    if not policy_runs:
        return "<evidence_unavailable>No policy search results were produced.</evidence_unavailable>"

    chunks = ["<search_results>"]
    for run in policy_runs:
        query_index = int(run.get("query_index", 0))
        search_query = str(run.get("search_query", ""))
        policy_result = run.get("policy_result", {})
        if not isinstance(policy_result, dict):
            policy_result = {}
        chunks.append(
            f'<result index="{query_index}">\n'
            f"<request>{escape_xml_text(search_query)}</request>\n"
            f"{build_policy_result_for_backbone(search_query, policy_result)}\n"
            "</result>"
        )
    chunks.append("</search_results>")
    return "\n".join(chunks)


def build_next_backbone_message(policy_output: str) -> dict[str, str]:
    return shared_build_next_backbone_message(policy_output)


def build_final_backbone_message(policy_output: str) -> dict[str, str]:
    return shared_build_final_backbone_message(policy_output)


def build_policy_failure_backbone_message() -> dict[str, str]:
    return shared_build_policy_failure_backbone_message()


def build_sft_messages(messages: list[dict[str, str]], final_output: str) -> list[dict[str, str]]:
    sft_messages = [dict(message) for message in messages]
    if final_output:
        for message in reversed(sft_messages):
            if message.get("role") == "assistant":
                message["content"] = final_output
                break
    return sft_messages


def build_assistant_message(content: str, message_fields: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    reasoning_content = str(message_fields.get("reasoning_content") or "").strip()
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    reasoning_details = message_fields.get("reasoning_details")
    if reasoning_details is not None:
        message["reasoning_details"] = to_jsonable(reasoning_details)
    return message


def build_policy_extra_body(args: argparse.Namespace) -> dict[str, Any]:
    extra_body = parse_json_object(getattr(args, "policy_extra_body_json", ""), field_name="--policy_extra_body_json")
    if bool(getattr(args, "policy_enable_thinking", False)):
        thinking_field = str(getattr(args, "policy_thinking_field", "thinking") or "thinking")
        if thinking_field == "enable_thinking":
            extra_body.setdefault("enable_thinking", True)
        else:
            thinking = extra_body.setdefault("thinking", {})
            if isinstance(thinking, dict):
                thinking.setdefault("type", str(getattr(args, "policy_thinking_type", "enabled") or "enabled"))
                if bool(getattr(args, "policy_preserve_reasoning_content", False)):
                    thinking.setdefault("clear_thinking", False)
    return extra_body


def run_policy_tool_loop(
    *,
    search_query: str,
    args: argparse.Namespace,
    semaphore: Optional[threading.BoundedSemaphore],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": search_query},
    ]

    tool_trace: list[dict[str, Any]] = []
    assistant_turns = 0
    final_output = ""
    error = None
    raw_api_responses: list[Any] = []
    token_usage = new_token_usage()
    api_call_stats: list[dict[str, Any]] = []
    assistant_message_trace: list[dict[str, Any]] = []
    policy_extra_body = build_policy_extra_body(args)
    final_reasoning_content = ""

    try:
        if not search_query:
            raise ValueError("empty policy search query")

        while assistant_turns < args.max_assistant_turns:
            if assistant_turns == args.max_assistant_turns - 1:
                messages.append(build_final_policy_turn_message())

            api_started_at = time.perf_counter()
            assistant_text, raw_response = call_chat_completion(
                messages=messages,
                api_url=args.policy_api_url,
                api_key=args.policy_api_key,
                model=args.policy_model,
                timeout=args.api_timeout,
                max_retries=args.api_max_retries,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                no_proxy=args.no_proxy,
                extra_body=policy_extra_body,
                preserve_reasoning_content=bool(getattr(args, "policy_preserve_reasoning_content", False)),
            )
            message_fields = extract_chat_message_fields(raw_response)
            api_elapsed_seconds = round_seconds(time.perf_counter() - api_started_at)
            assistant_turns += 1
            usage = add_api_response_usage(token_usage, raw_response, args.policy_model)
            reasoning_content = str(message_fields.get("reasoning_content") or "")
            reasoning_details = message_fields.get("reasoning_details")
            api_call_stats.append(
                {
                    "stage": "policy",
                    "assistant_turn": assistant_turns,
                    "model": args.policy_model,
                    "elapsed_seconds": api_elapsed_seconds,
                    "usage": usage,
                    "content": assistant_text,
                    "reasoning_content": reasoning_content,
                    "reasoning_details": to_jsonable(reasoning_details) if reasoning_details is not None else None,
                }
            )
            messages.append(build_assistant_message(assistant_text, message_fields))
            assistant_message_trace.append(
                {
                    "turn": assistant_turns,
                    "content": assistant_text,
                    "reasoning_content": reasoning_content,
                    "reasoning_details": to_jsonable(reasoning_details) if reasoning_details is not None else None,
                }
            )
            if args.save_raw_api_response:
                raw_api_responses.append(raw_response)

            tool_calls = parse_search_tool_calls(assistant_text)
            if tool_calls:
                for call in tool_calls[: args.max_parallel_calls]:
                    name = str(call.get("name", ""))
                    arguments = call.get("arguments", {})
                    if not isinstance(arguments, dict):
                        arguments = {}
                    if name != "search_subagent":
                        messages.append(build_correction_message(f"unknown tool {name!r}"))
                        continue
                    query = str(arguments.get("query", "")).strip()
                    if not query:
                        messages.append(build_correction_message("search_subagent query is empty"))
                        continue
                    tool_started_at = time.perf_counter()
                    tool_text, tool_payload = run_search_tool(
                        query=query,
                        retrieval_url=args.retrieval_url,
                        topk=args.topk,
                        timeout=args.retrieval_timeout,
                        semaphore=semaphore,
                        no_proxy=args.no_proxy,
                        save_raw_retrieval_response=args.save_raw_retrieval_response,
                    )
                    tool_elapsed_seconds = round_seconds(time.perf_counter() - tool_started_at)
                    messages.append({"role": "tool", "name": "search_subagent", "content": tool_text})
                    tool_trace.append(
                        {
                            "turn": assistant_turns,
                            "name": "search_subagent",
                            "arguments": {"query": query},
                            "response": tool_payload,
                            "elapsed_seconds": tool_elapsed_seconds,
                        }
                    )
                continue

            answer_evidence_output = extract_answer_evidence_blocks(assistant_text)
            if answer_evidence_output:
                final_output = answer_evidence_output
                final_reasoning_content = reasoning_content
                break

            messages.append(build_correction_message("missing tool call or final <answer>/<evidence> blocks"))

        if not final_output:
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    final_output = msg.get("content", "")
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    for item in reversed(assistant_message_trace):
        item_content = str(item.get("content") or "")
        if item_content == str(final_output or "") or extract_answer_evidence_blocks(item_content) == str(final_output or ""):
            final_reasoning_content = str(item.get("reasoning_content") or "")
            break

    result = {
        "messages": messages,
        "sft_messages": build_sft_messages(messages, final_output),
        "tool_trace": tool_trace,
        "final_output": final_output,
        "final_content": final_output,
        "final_reasoning_content": final_reasoning_content,
        "has_answer_evidence": has_final_answer(final_output),
        "assistant_turns": assistant_turns,
        "tool_call_count": len(tool_trace),
        "api_model": args.policy_model,
        "policy_model": args.policy_model,
        "assistant_message_trace": assistant_message_trace,
        "elapsed_seconds": round_seconds(time.perf_counter() - started_at),
        "token_usage": token_usage,
        "api_call_stats": api_call_stats,
        "error": error,
    }
    if args.save_raw_api_response:
        result["raw_api_responses"] = raw_api_responses
    return result


def run_policy_queries_parallel(
    *,
    search_queries: list[str],
    args: argparse.Namespace,
    semaphore: Optional[threading.BoundedSemaphore],
) -> list[dict[str, Any]]:
    indexed_queries = [(idx, query.strip()) for idx, query in enumerate(search_queries) if query.strip()]
    if not indexed_queries:
        return []

    max_parallel_policy_queries = int(getattr(args, "max_parallel_policy_queries", 1) or 1)
    max_workers = max(1, min(len(indexed_queries), max_parallel_policy_queries))
    results: list[Optional[dict[str, Any]]] = [None] * len(indexed_queries)

    def run_one(query_index: int, query: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        result = run_policy_tool_loop(search_query=query, args=args, semaphore=semaphore)
        return {
            "query_index": query_index,
            "search_query": query,
            "policy_result": result,
            "elapsed_seconds": round_seconds(time.perf_counter() - started_at),
        }

    if max_workers == 1:
        for position, (query_index, query) in enumerate(indexed_queries):
            results[position] = run_one(query_index, query)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_position = {
                executor.submit(run_one, query_index, query): position
                for position, (query_index, query) in enumerate(indexed_queries)
            }
            for future in as_completed(future_to_position):
                position = future_to_position[future]
                try:
                    results[position] = future.result()
                except Exception as exc:
                    query_index, query = indexed_queries[position]
                    results[position] = {
                        "query_index": query_index,
                        "search_query": query,
                        "policy_result": {
                            "messages": [],
                            "sft_messages": [],
                            "tool_trace": [],
                            "final_output": "",
                            "final_content": "",
                            "final_reasoning_content": "",
                            "has_answer_evidence": False,
                            "assistant_turns": 0,
                            "tool_call_count": 0,
                            "api_model": args.policy_model,
                            "policy_model": args.policy_model,
                            "elapsed_seconds": 0.0,
                            "token_usage": new_token_usage(),
                            "api_call_stats": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        "elapsed_seconds": 0.0,
                    }

    return [result for result in results if result is not None]


ALWAYS_KEEP_FIELDS = {
    "final_em",
    "final_f1",
    "final_answer_em",
    "final_answer_f1",
    "backbone_final_answer_llm_judge_score",
}


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: to_jsonable(v) for k, v in record.items() if v is not None or k in ALWAYS_KEEP_FIELDS}


def process_one(index: int, row: dict[str, Any], args: argparse.Namespace, semaphore) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_index = int(row.get("__source_index", index))
    question = extract_question(row)
    ground_truth = extract_ground_truth(row)
    uid = str(row.get("uid") or row.get("id") or uuid4().hex)
    records: list[dict[str, Any]] = []
    sample_started_at = time.perf_counter()
    sample_token_usage = new_token_usage()
    api_call_stats: list[dict[str, Any]] = []
    orchestrator_chain: list[dict[str, Any]] = []
    last_policy_output = ""
    last_policy_backbone_output = ""
    last_policy_query_count = 0
    final_output = ""
    final_answer = ""
    final_answer_source = None
    sample_error = None
    final_round_search_rejected = False

    if not question:
        elapsed_seconds = round_seconds(time.perf_counter() - sample_started_at)
        final_em, final_f1, ground_truth_targets = compute_final_scores("", ground_truth)
        sample_summary = {
            "source_index": source_index,
            "uid": uid,
            "data_source": row.get("data_source"),
            "question": question,
            "ground_truth": ground_truth,
            "ground_truth_targets": ground_truth_targets,
            "policy_round_count": 0,
            "orchestrator_chain": orchestrator_chain,
            "final_output": "",
            "final_answer": "",
            "final_answer_source": final_answer_source,
            "final_answer_em": final_em,
            "final_answer_f1": final_f1,
            "final_em": final_em,
            "final_f1": final_f1,
            "elapsed_seconds": elapsed_seconds,
            "token_usage": sample_token_usage,
            "api_call_stats": api_call_stats,
            "error": "empty question",
        }
        record = {
            "record_type": "policy_round_sft",
            "source_index": source_index,
            "uid": uid,
            "data_source": row.get("data_source"),
            "ground_truth": ground_truth,
            "ground_truth_targets": ground_truth_targets,
            "messages": [],
            "sft_messages": [],
            "tool_trace": [],
            "final_output": "",
            "has_answer_evidence": False,
            "final_answer": "",
            "final_answer_source": final_answer_source,
            "final_answer_em": final_em,
            "final_answer_f1": final_f1,
            "final_em": final_em,
            "final_f1": final_f1,
            "sample_elapsed_seconds": elapsed_seconds,
            "sample_token_usage": sample_token_usage,
            "error": "empty question",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return [compact_record(record)], compact_record(sample_summary)

    backbone_messages = build_initial_backbone_messages(question)

    for round_idx in range(args.max_orchestrator_rounds):
        forced_final_backbone_turn = False
        if round_idx == args.max_orchestrator_rounds - 1 and last_policy_output:
            forced_final_backbone_turn = True
            policy_output_for_backbone = last_policy_backbone_output or (
                "<evidence_unavailable>The policy model did not produce a valid answer/evidence result."
                "</evidence_unavailable>"
            )
            backbone_messages.append(build_final_backbone_message(policy_output_for_backbone))

        backbone_error = None
        backbone_response = ""
        backbone_elapsed_seconds = None
        backbone_usage: dict[str, int | float] = {}
        try:
            backbone_started_at = time.perf_counter()
            backbone_response, _raw_backbone = call_chat_completion(
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
            backbone_elapsed_seconds = round_seconds(time.perf_counter() - backbone_started_at)
            backbone_usage = add_api_response_usage(sample_token_usage, _raw_backbone, args.backbone_model)
            api_call_stats.append(
                {
                    "round": round_idx,
                    "stage": "backbone",
                    "model": args.backbone_model,
                    "elapsed_seconds": backbone_elapsed_seconds,
                    "usage": backbone_usage,
                }
            )
        except Exception as exc:
            backbone_elapsed_seconds = round_seconds(time.perf_counter() - backbone_started_at)
            backbone_error = f"{type(exc).__name__}: {exc}"

        backbone_messages.append({"role": "assistant", "content": backbone_response})
        search_query = parse_backbone_search(backbone_response)
        search_queries = parse_backbone_search_queries(search_query)
        search_query_truncated_count = 0
        max_backbone_search_queries = int(getattr(args, "max_backbone_search_queries", 1) or 0)
        if (
            search_queries
            and max_backbone_search_queries > 0
            and len(search_queries) > max_backbone_search_queries
        ):
            search_query_truncated_count = len(search_queries) - max_backbone_search_queries
            search_queries = search_queries[:max_backbone_search_queries]
        backbone_final_answer = extract_backbone_final_answer(backbone_response)
        if backbone_final_answer:
            final_output = backbone_response
            final_answer = backbone_final_answer
            final_answer_source = "backbone"
        elif not search_query and backbone_response.strip() and not backbone_error:
            final_output = backbone_response
            final_answer = backbone_response.strip()
            final_answer_source = "backbone"

        orchestrator_chain.append(
            {
                "round": round_idx,
                "stage": "backbone_output",
                "response": backbone_response,
                "search_query": search_query,
                "search_queries": search_queries,
                "search_query_count": len(search_queries),
                "search_query_truncated_count": search_query_truncated_count,
                "final_answer": backbone_final_answer,
                "elapsed_seconds": backbone_elapsed_seconds,
                "token_usage": backbone_usage,
                "error": backbone_error,
            }
        )

        if forced_final_backbone_turn and search_query:
            sample_error = "final_round_backbone_emitted_search"
            final_round_search_rejected = True
            final_output = backbone_response
            final_answer = ""
            final_answer_source = None
            orchestrator_chain[-1]["error"] = sample_error
            break

        if backbone_error or not search_query:
            break

        policy_started_at = time.perf_counter()
        policy_runs = run_policy_queries_parallel(search_queries=search_queries, args=args, semaphore=semaphore)
        policy_elapsed_seconds = round_seconds(time.perf_counter() - policy_started_at)
        last_policy_query_count = len(policy_runs)
        last_policy_backbone_output = build_parallel_policy_results_for_backbone(policy_runs)
        last_policy_output = last_policy_backbone_output

        policy_group_token_usage = new_token_usage()
        policy_group_errors: list[str] = []
        policy_group_results: list[dict[str, Any]] = []
        query_count = len(search_queries)

        for run in policy_runs:
            query_index = int(run.get("query_index", 0))
            query = str(run.get("search_query", ""))
            policy_result = run.get("policy_result", {})
            if not isinstance(policy_result, dict):
                policy_result = {}

            merge_token_usage(sample_token_usage, policy_result.get("token_usage", {}))
            merge_token_usage(policy_group_token_usage, policy_result.get("token_usage", {}))
            for call_stat in policy_result.get("api_call_stats", []):
                if isinstance(call_stat, dict):
                    api_call_stats.append({"round": round_idx, "query_index": query_index, "search_query": query, **call_stat})

            policy_output = str(policy_result.get("final_output", ""))
            policy_backbone_input = build_policy_result_for_backbone(query, policy_result)
            if policy_result.get("error"):
                policy_group_errors.append(f"q{query_index}: {policy_result.get('error')}")

            policy_event = {
                "round": round_idx,
                "stage": "policy_output",
                "query_index": query_index,
                "query_count": query_count,
                "search_query": query,
                "backbone_search_block": search_query,
                "final_output": policy_output,
                "backbone_input": policy_backbone_input,
                "has_answer_evidence": bool(policy_result.get("has_answer_evidence", False)),
                "elapsed_seconds": policy_result.get("elapsed_seconds", run.get("elapsed_seconds", policy_elapsed_seconds)),
                "token_usage": policy_result.get("token_usage", {}),
                "error": policy_result.get("error"),
            }
            policy_group_results.append(
                {
                    "query_index": query_index,
                    "search_query": query,
                    "final_output": policy_output,
                    "final_content": policy_result.get("final_content", policy_output),
                    "final_reasoning_content": policy_result.get("final_reasoning_content", ""),
                    "backbone_input": policy_backbone_input,
                    "has_answer_evidence": bool(policy_result.get("has_answer_evidence", False)),
                    "elapsed_seconds": policy_result.get("elapsed_seconds", run.get("elapsed_seconds")),
                    "error": policy_result.get("error"),
                }
            )
            policy_chain = [*orchestrator_chain, policy_event]
            policy_round_id = f"{source_index}:r{round_idx}" if query_count == 1 else f"{source_index}:r{round_idx}:q{query_index}"

            record = {
                "record_type": "policy_round_sft",
                "policy_round_id": policy_round_id,
                "parallel_group_id": f"{source_index}:r{round_idx}",
                "source_index": source_index,
                "round": round_idx,
                "query_index": query_index,
                "query_count": query_count,
                "uid": uid,
                "data_source": row.get("data_source"),
                "question": question,
                "ground_truth": ground_truth,
                "policy_input": query,
                "backbone_search_block": search_query,
                "messages": policy_result.get("messages", []),
                "sft_messages": policy_result.get("sft_messages", []),
                "tool_trace": policy_result.get("tool_trace", []),
                "final_output": policy_output,
                "final_content": policy_result.get("final_content", policy_output),
                "final_reasoning_content": policy_result.get("final_reasoning_content", ""),
                "has_answer_evidence": policy_result.get("has_answer_evidence", False),
                "assistant_turns": policy_result.get("assistant_turns", 0),
                "tool_call_count": policy_result.get("tool_call_count", 0),
                "api_model": args.policy_model,
                "policy_model": args.policy_model,
                "backbone_model": args.backbone_model,
                "assistant_message_trace": policy_result.get("assistant_message_trace", []),
                "orchestrator_chain": policy_chain,
                "policy_elapsed_seconds": policy_result.get("elapsed_seconds", run.get("elapsed_seconds", policy_elapsed_seconds)),
                "policy_token_usage": policy_result.get("token_usage", {}),
                "policy_api_call_stats": policy_result.get("api_call_stats", []),
                "error": policy_result.get("error"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if args.save_raw_api_response and "raw_api_responses" in policy_result:
                record["raw_api_responses"] = policy_result["raw_api_responses"]
            records.append(record)

        policy_group_event = {
            "round": round_idx,
            "stage": "policy_outputs",
            "search_query": search_query,
            "search_queries": search_queries,
            "parallel_query_count": len(policy_runs),
            "results": policy_group_results,
            "final_output": last_policy_output,
            "backbone_input": last_policy_backbone_output,
            "has_answer_evidence": bool(policy_runs) and all(
                bool((run.get("policy_result") or {}).get("has_answer_evidence", False))
                for run in policy_runs
                if isinstance(run.get("policy_result"), dict)
            ),
            "elapsed_seconds": policy_elapsed_seconds,
            "token_usage": policy_group_token_usage,
            "error": "; ".join(policy_group_errors) if policy_group_errors else None,
        }
        orchestrator_chain.append(policy_group_event)

        if last_policy_backbone_output:
            backbone_messages.append(build_next_backbone_message(last_policy_backbone_output))
        else:
            backbone_messages.append(build_policy_failure_backbone_message())

    if not final_answer and last_policy_backbone_output and not final_round_search_rejected:
        final_output = last_policy_backbone_output
        if last_policy_query_count == 1:
            final_answer = extract_final_answer(last_policy_backbone_output)
            final_answer_source = "policy"
    elif not final_answer and last_policy_output:
        final_output = last_policy_output

    final_em, final_f1, ground_truth_targets = compute_final_scores(final_answer, ground_truth)
    if final_answer_source == "backbone":
        backbone_final_answer_llm_judge = judge_backbone_final_answer(
            question=question,
            ground_truth_targets=ground_truth_targets,
            predicted_answer=final_answer,
            args=args,
        )
        judge_usage = backbone_final_answer_llm_judge.get("usage", {})
        add_token_usage(sample_token_usage, judge_usage, args.backbone_judge_model)
        api_call_stats.append(
            {
                "stage": "backbone_judge",
                "model": args.backbone_judge_model,
                "temperature": 0.0,
                "elapsed_seconds": backbone_final_answer_llm_judge.get("elapsed_seconds"),
                "usage": judge_usage,
                "response": backbone_final_answer_llm_judge.get("response", ""),
                "score": backbone_final_answer_llm_judge.get("score"),
                "is_correct": backbone_final_answer_llm_judge.get("is_correct"),
                "error": backbone_final_answer_llm_judge.get("error"),
            }
        )
        orchestrator_chain.append(
            {
                "stage": "backbone_final_answer_llm_judge",
                "model": args.backbone_judge_model,
                "temperature": 0.0,
                "predicted_answer": final_answer,
                "ground_truth_targets": ground_truth_targets,
                "score": backbone_final_answer_llm_judge.get("score"),
                "is_correct": backbone_final_answer_llm_judge.get("is_correct"),
                "response": backbone_final_answer_llm_judge.get("response", ""),
                "elapsed_seconds": backbone_final_answer_llm_judge.get("elapsed_seconds"),
                "token_usage": judge_usage,
                "error": backbone_final_answer_llm_judge.get("error"),
            }
        )
    else:
        backbone_final_answer_llm_judge = {
            "score": None,
            "is_correct": None,
            "response": "",
            "error": "final answer was not produced by backbone",
            "usage": {},
            "elapsed_seconds": None,
        }
    elapsed_seconds = round_seconds(time.perf_counter() - sample_started_at)
    sample_summary = {
        "source_index": source_index,
        "uid": uid,
        "data_source": row.get("data_source"),
        "question": question,
        "ground_truth": ground_truth,
        "ground_truth_targets": ground_truth_targets,
        "policy_round_count": len(records),
        "orchestrator_chain": orchestrator_chain,
        "final_output": final_output,
        "final_answer": final_answer,
        "final_answer_source": final_answer_source,
        "final_answer_em": final_em,
        "final_answer_f1": final_f1,
        "final_em": final_em,
        "final_f1": final_f1,
        "backbone_final_answer_llm_judge_score": backbone_final_answer_llm_judge.get("score"),
        "backbone_final_answer_llm_judge_is_correct": backbone_final_answer_llm_judge.get("is_correct"),
        "backbone_final_answer_llm_judge_response": backbone_final_answer_llm_judge.get("response", ""),
        "backbone_final_answer_llm_judge_error": backbone_final_answer_llm_judge.get("error"),
        "backbone_final_answer_llm_judge_elapsed_seconds": backbone_final_answer_llm_judge.get("elapsed_seconds"),
        "elapsed_seconds": elapsed_seconds,
        "token_usage": sample_token_usage,
        "api_call_stats": api_call_stats,
        "model_pair": {"backbone_model": args.backbone_model, "policy_model": args.policy_model},
        "error": sample_error,
    }

    for record in records:
        record.update(
            {
                "ground_truth_targets": ground_truth_targets,
                "final_answer": final_answer,
                "final_answer_source": final_answer_source,
                "final_answer_em": final_em,
                "final_answer_f1": final_f1,
                "final_em": final_em,
                "final_f1": final_f1,
                "backbone_final_answer_llm_judge_score": backbone_final_answer_llm_judge.get("score"),
                "backbone_final_answer_llm_judge_is_correct": backbone_final_answer_llm_judge.get("is_correct"),
                "backbone_final_answer_llm_judge_response": backbone_final_answer_llm_judge.get("response", ""),
                "backbone_final_answer_llm_judge_error": backbone_final_answer_llm_judge.get("error"),
                "backbone_final_answer_llm_judge_elapsed_seconds": backbone_final_answer_llm_judge.get("elapsed_seconds"),
                "sample_elapsed_seconds": elapsed_seconds,
                "sample_token_usage": sample_token_usage,
                "sample_api_call_stats": api_call_stats,
                "model_pair": {"backbone_model": args.backbone_model, "policy_model": args.policy_model},
                "sample_error": sample_error,
            }
        )

    return [compact_record(record) for record in records], compact_record(sample_summary)


def process_one_with_trace(
    index: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    semaphore,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, sample_summary = process_one(index, row, args, semaphore)
    source_index = int(row.get("__source_index", index))
    policy_rounds = [
        {
            "policy_round_id": record.get("policy_round_id"),
            "round": record.get("round"),
            "query_index": record.get("query_index"),
            "query_count": record.get("query_count"),
            "policy_input": record.get("policy_input"),
            "backbone_search_block": record.get("backbone_search_block"),
            "final_output": record.get("final_output"),
            "final_content": record.get("final_content"),
            "final_reasoning_content": record.get("final_reasoning_content"),
            "has_answer_evidence": record.get("has_answer_evidence"),
            "assistant_message_trace": record.get("assistant_message_trace", []),
            "tool_trace": record.get("tool_trace", []),
            "policy_elapsed_seconds": record.get("policy_elapsed_seconds"),
            "policy_token_usage": record.get("policy_token_usage", {}),
            "policy_api_call_stats": record.get("policy_api_call_stats", []),
            "error": record.get("error"),
        }
        for record in records
    ]

    trace_record = {
        "record_type": "orchestrator_trace",
        "source_index": source_index,
        "uid": sample_summary.get("uid", str(row.get("uid") or row.get("id") or "")),
        "data_source": sample_summary.get("data_source", row.get("data_source")),
        "question": sample_summary.get("question", extract_question(row)),
        "ground_truth": sample_summary.get("ground_truth", extract_ground_truth(row)),
        "ground_truth_targets": sample_summary.get("ground_truth_targets", []),
        "policy_round_count": len(records),
        "policy_rounds": policy_rounds,
        "orchestrator_chain": sample_summary.get("orchestrator_chain", []),
        "final_output": sample_summary.get("final_output", ""),
        "final_answer": sample_summary.get("final_answer", ""),
        "final_answer_source": sample_summary.get("final_answer_source"),
        "final_answer_em": sample_summary.get("final_answer_em"),
        "final_answer_f1": sample_summary.get("final_answer_f1"),
        "final_em": sample_summary.get("final_em"),
        "final_f1": sample_summary.get("final_f1"),
        "backbone_final_answer_llm_judge_score": sample_summary.get("backbone_final_answer_llm_judge_score"),
        "backbone_final_answer_llm_judge_is_correct": sample_summary.get("backbone_final_answer_llm_judge_is_correct"),
        "backbone_final_answer_llm_judge_response": sample_summary.get("backbone_final_answer_llm_judge_response", ""),
        "backbone_final_answer_llm_judge_error": sample_summary.get("backbone_final_answer_llm_judge_error"),
        "backbone_final_answer_llm_judge_elapsed_seconds": sample_summary.get("backbone_final_answer_llm_judge_elapsed_seconds"),
        "elapsed_seconds": sample_summary.get("elapsed_seconds"),
        "token_usage": sample_summary.get("token_usage", new_token_usage()),
        "api_call_stats": sample_summary.get("api_call_stats", []),
        "model_pair": sample_summary.get("model_pair", {"backbone_model": args.backbone_model, "policy_model": args.policy_model}),
        "error": sample_summary.get("error"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return records, compact_record(trace_record)


def load_done_indices(output_path: str) -> set[str]:
    done: set[str] = set()
    if not output_path or not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "policy_round_id" in obj:
                done.add(str(obj["policy_round_id"]))
    return done


def load_done_source_indices(trace_output_path: str) -> set[int]:
    done: set[int] = set()
    if not trace_output_path or not os.path.exists(trace_output_path):
        return done
    with open(trace_output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("record_type") != "orchestrator_trace":
                continue
            try:
                done.add(int(obj["source_index"]))
            except (KeyError, TypeError, ValueError):
                continue
    return done


def write_jsonl_record(path: str, record: dict[str, Any], lock: threading.Lock) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def default_trace_output_path(output_path: str) -> str:
    path = Path(output_path)
    suffix = "".join(path.suffixes)
    if suffix:
        base = str(path)[: -len(suffix)]
        return f"{base}.orchestrator_traces{suffix}"
    return f"{output_path}.orchestrator_traces.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate policy SFT rollouts with search_subagent tools.",
        epilog=VLLM_STARTUP_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input parquet/jsonl/json file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--orchestrator_output",
        default=None,
        help="Optional JSONL path for per-source-sample orchestrator chains. Defaults beside --output.",
    )
    parser.add_argument("--env_file", default="", help="Optional backbone env file.")
    parser.add_argument("--api_url", default=DEFAULT_API_URL, help="Backbone API base URL.")
    parser.add_argument("--api_key_env_var", default="BACKBONE_API_KEY", help="Env var for the backbone API key.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Policy model alias kept for backward compatibility.")
    parser.add_argument("--policy_api_url", default=DEFAULT_POLICY_API_URL, help="Policy API base URL.")
    parser.add_argument("--policy_model", default=None, help="Policy model name. Defaults to --model.")
    parser.add_argument("--policy_env_file", default="", help="Optional policy env file.")
    parser.add_argument("--policy_api_key", default=os.environ.get("POLICY_API_KEY", ""))
    parser.add_argument("--policy_api_key_env_var", default="POLICY_API_KEY", help="Env var for the policy API key.")
    parser.add_argument(
        "--policy_extra_body_json",
        default="",
        help="Extra JSON object merged into policy chat completion payload.",
    )
    parser.add_argument(
        "--policy_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable policy model thinking mode via the configured thinking field.",
    )
    parser.add_argument(
        "--policy_thinking_field",
        choices=("thinking", "enable_thinking"),
        default="thinking",
        help="Request field used to enable policy thinking mode.",
    )
    parser.add_argument("--policy_thinking_type", default="enabled", help="Value for policy thinking.type.")
    parser.add_argument(
        "--policy_preserve_reasoning_content",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send previous assistant reasoning_content back to the policy API on later turns.",
    )
    parser.add_argument("--backbone_model", default=DEFAULT_BACKBONE_MODEL)
    parser.add_argument("--backbone_judge_api_url", default=DEFAULT_BACKBONE_JUDGE_API_URL, help="Backbone final-answer judge API base URL.")
    parser.add_argument("--backbone_judge_model", default=DEFAULT_BACKBONE_JUDGE_MODEL, help="Backbone final-answer judge model name.")
    parser.add_argument("--backbone_judge_env_file", default=".secrets/deepseek.env", help="Optional env file for the backbone final-answer judge API key.")
    parser.add_argument("--backbone_judge_api_key", default=os.environ.get("BACKBONE_JUDGE_API_KEY", ""))
    parser.add_argument("--backbone_judge_api_key_env_var", default="BACKBONE_JUDGE_API_KEY", help="Env var for the backbone final-answer judge API key.")
    parser.add_argument("--api_key", default=os.environ.get("BACKBONE_API_KEY", ""))
    parser.add_argument("--api_timeout", type=float, default=120.0)
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--backbone_temperature", type=float, default=0.0)
    parser.add_argument("--backbone_max_tokens", type=int, default=8192)
    parser.add_argument(
        "--backbone_judge_max_tokens",
        type=int,
        default=16,
        help="Max completion tokens for the backbone final-answer LLM judge.",
    )
    parser.add_argument("--no_proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retrieval_url", default=DEFAULT_RETRIEVAL_URL)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retrieval_timeout", type=int, default=180)
    parser.add_argument("--retrieval_max_concurrent", type=int, default=64)
    parser.add_argument("--save_raw_retrieval_response", action="store_true")
    parser.add_argument("--max_orchestrator_rounds", type=int, default=4)
    parser.add_argument("--max_assistant_turns", type=int, default=3)
    parser.add_argument(
        "--max_backbone_search_queries",
        type=int,
        default=3,
        help="Maximum focused questions to execute from one backbone <search> block. Use <=0 for no cap.",
    )
    parser.add_argument(
        "--max_parallel_policy_queries",
        type=int,
        default=3,
        help="Maximum policy subagent loops to run concurrently for one backbone <search> block.",
    )
    parser.add_argument("--max_parallel_calls", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_raw_api_response", action="store_true")
    parser.add_argument("--system_prompt", default=POLICY_SYSTEM_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    load_env_file(args.policy_env_file)
    load_env_file(args.backbone_judge_env_file)
    if not args.policy_model:
        args.policy_model = args.model
    args.model = args.policy_model
    args.api_key = resolve_api_key(args.api_key, args.api_key_env_var, ("DEEPSEEK_API_KEY",))
    args.backbone_judge_api_key = resolve_api_key(
        args.backbone_judge_api_key,
        args.backbone_judge_api_key_env_var,
        ("DEEPSEEK_API_KEY", "BACKBONE_API_KEY"),
    )
    args.policy_api_key = resolve_api_key(
        args.policy_api_key,
        args.policy_api_key_env_var,
        ("ZAI_API_KEY", "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"),
    )
    if not args.policy_api_key and args.policy_api_url.rstrip("/") == args.api_url.rstrip("/"):
        args.policy_api_key = args.api_key
    if not args.api_key and not is_local_api_url(args.api_url):
        raise ValueError(f"Missing backbone API key: set {args.api_key_env_var} or pass --api_key.")
    if not args.backbone_judge_api_key and not is_local_api_url(args.backbone_judge_api_url):
        raise ValueError(
            f"Missing backbone judge API key: set {args.backbone_judge_api_key_env_var}/DEEPSEEK_API_KEY "
            "or pass --backbone_judge_api_key."
        )
    if not args.policy_api_key and not is_local_api_url(args.policy_api_url):
        raise ValueError(
            f"Missing policy API key: set {args.policy_api_key_env_var}, pass --policy_api_key, "
            "or use --policy_env_file."
        )
    if args.orchestrator_output is None:
        args.orchestrator_output = default_trace_output_path(args.output)

    rows = load_records(args.input, limit=args.limit, offset=args.offset)
    for local_idx, row in enumerate(rows):
        if "__source_index" in row:
            row["__source_index"] = int(row["__source_index"])
        elif "source_index" in row:
            row["__source_index"] = int(row["source_index"])
        else:
            row["__source_index"] = args.offset + local_idx

    done_round_ids = load_done_indices(args.output) if args.resume else set()
    done_source_indices = load_done_source_indices(args.orchestrator_output) if args.resume else set()
    rows_to_process = [
        row for row in rows if int(row.get("__source_index", -1)) not in done_source_indices
    ]
    print(
        f"Loaded {len(rows)} rows from {args.input}; "
        f"found {len(done_round_ids)} existing policy-round records and "
        f"{len(done_source_indices)} existing orchestrator traces; "
        f"processing {len(rows_to_process)} rows."
    )

    write_lock = threading.Lock()
    semaphore = (
        threading.BoundedSemaphore(args.retrieval_max_concurrent)
        if args.retrieval_max_concurrent and args.retrieval_max_concurrent > 0
        else None
    )

    if args.num_workers <= 1:
        for i, row in enumerate(rows_to_process):
            records, trace_record = process_one_with_trace(i, row, args, semaphore)
            for record in records:
                if str(record.get("policy_round_id", "")) in done_round_ids:
                    continue
                write_jsonl_record(args.output, record, write_lock)
            write_jsonl_record(args.orchestrator_output, trace_record, write_lock)
            if (i + 1) % 10 == 0:
                print(f"processed={i + 1}/{len(rows_to_process)}")
        return

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_one_with_trace, i, row, args, semaphore): row
            for i, row in enumerate(rows_to_process)
        }
        completed = 0
        for future in as_completed(futures):
            records, trace_record = future.result()
            for record in records:
                if str(record.get("policy_round_id", "")) in done_round_ids:
                    continue
                write_jsonl_record(args.output, record, write_lock)
            write_jsonl_record(args.orchestrator_output, trace_record, write_lock)
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                print(
                    f"processed={completed}/{len(futures)} "
                    f"output={args.output} orchestrator_output={args.orchestrator_output}"
                )


if __name__ == "__main__":
    main()
