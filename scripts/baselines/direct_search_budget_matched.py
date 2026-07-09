#!/usr/bin/env python3
"""Budget-matched direct-search baseline.

DirectSearch-BudgetMatched lets the main agent call the same retrieval backend
used by SearchSubagentTool, but enforces the two-stage backbone-equivalent
search budget: 4 outer rounds, at most 3 backbone search turns total, each
search turn may contain at most 3 parallel queries, and one no-tool fallback
finalization after the 4 rounds if needed.
"""

from __future__ import annotations

import argparse
import ast
import copy
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
from uuid import uuid4

import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


BASELINE_NAME = "DirectSearch-BudgetMatched"
DEFAULT_API_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_INPUT = "/ai/cqn/datacon/data/hotpotqa_2wiki_musique_train/val_mixed_900.parquet"
DEFAULT_OUTPUT = "/ai/cqn/datacon/data/direct_search_budget_matched/predictions.json"
DEFAULT_TOOL_CONFIG = "/ai/cqn/datacon/scripts/examples/config/tool_config/search_subagent_tool_config.yaml"
DEFAULT_RETRIEVAL_URL = "http://162.30.4.229:8765/search"

DEFAULT_OUTER_ROUNDS = 4
DEFAULT_SEARCHES_PER_OUTER_ROUND = 1
DEFAULT_GLOBAL_SEARCH_CAP = 3
DEFAULT_MAX_PARALLEL_SEARCH_QUERIES = 3
DEFAULT_TOPK = 3
DEFAULT_RETRIEVAL_TIMEOUT = 180
DEFAULT_RETRIEVAL_MAX_CONCURRENT = 64
SEARCH_MAX_RETRIES = 10
SEARCH_INITIAL_RETRY_DELAY = 1

DIRECT_SYSTEM_PROMPT = f"""Please answer the question.

You are the backbone model in a direct-search question-answering system for the {BASELINE_NAME} baseline.

Your job is to:
1. Understand the original question.
2. Identify the missing factual evidence.
3. Decompose the question into atomic evidence requests.
4. Use <search> only to ask for those missing facts.
5. Produce the final answer yourself after enough evidence is available.

Output format:

If you already have enough evidence, output only:
<final answer>...</final answer>

Otherwise, output exactly one <search>...</search> block.

A <search> block should contain either:
- one concise, specific, answerable natural-language question; or
- a JSON-style list of up to 3 such questions, if multiple independent facts are needed.

Search decomposition rules:

- Never copy or lightly rewrite the whole original question into <search>.
- Ask only for missing atomic facts needed to answer the original question.
- Each search question should usually focus on one entity and one attribute, relation, date, location, event, or fact.
- For comparison questions involving multiple entities, ask one focused question per entity.
- For multi-hop questions, ask for the next missing bridge fact first; after evidence is returned, ask the next focused follow-up if needed.
- Use the exact entity names and requested relations from the original question or retrieved evidence.
- Do not add guesses, candidate answers, unsupported locations, dates, aliases, categories, near-synonyms, or alternate entities.
- Do not rewrite an entity into a different entity.
- Do not ask search to answer the final comparison, judgment, counting, temporal ordering, or multi-hop question directly.
- If multiple independent facts are needed, put them inside one single <search> block as a JSON-style list.
- Do not output multiple <search> blocks in the same assistant message.

Final answer style:

- Use short-answer QA style.
- The final answer should usually be one short phrase or one short sentence.
- Base the final answer only on facts explicitly supported by returned search results.
- Do not answer from memory, prior knowledge, assumptions, or unstated background information.
- Do not infer dates, locations, names, relationships, or explanations unless they are directly supported by the returned search results.
- If the returned search results do not support an answer and search budget remains, issue another focused <search> request instead of guessing.
- If the search budget is exhausted and the returned evidence still does not support an answer, output <final answer>Insufficient evidence</final answer>.
- Do not include explanations, evidence, citations, reasoning steps, or background.
- Do not restate the retrieved evidence.
- Answer only the original question.
- For entity/date/place/country questions, output only the answer value when possible.
- For yes/no questions, start with "Yes" or "No" and include only the minimal fact needed.

Budget rules:

- There are 4 outer rounds.
- Across the whole sample you may use at most 3 backbone search turns.
- Each backbone search turn may execute at most 3 parallel retrieval queries.
- Output at most one <search> block in one assistant turn.
- If you output a <search> block, output no other text in that assistant turn.
- Search results are the only external evidence you may use.
"""


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
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


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

        rows = pd.read_parquet(path).to_dict(orient="records")
    elif suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("data") or data.get("records") or data.get("results") or []
        else:
            rows = []
    else:
        raise ValueError(f"Unsupported input extension: {suffix}")

    rows = [to_jsonable(row) for row in rows]
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            return normalize_messages(json.loads(value))
        except Exception:
            return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        content = value.get("content", "")
        return [
            {
                "role": str(value.get("role", "user")),
                "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
            }
        ]
    if isinstance(value, list):
        messages: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                messages.append(
                    {
                        "role": str(item.get("role", "user")),
                        "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                    }
                )
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages
    return []


def extract_question(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict):
            question = ground_truth.get("question")
            if isinstance(question, str) and question.strip():
                return question.strip()

    for key in ("question", "query", "input", "prompt", "problem"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("raw_prompt", "messages"):
        for msg in reversed(normalize_messages(row.get(key))):
            if msg.get("role") == "user" and str(msg.get("content", "")).strip():
                return str(msg["content"]).strip()
    return ""


def extract_ground_truth(row: dict[str, Any]) -> Any:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict):
            for key in ("target", "answer", "answers", "golden_answers", "gt", "gts"):
                value = ground_truth.get(key)
                if value not in (None, "", []):
                    return value
        elif ground_truth not in (None, "", []):
            return ground_truth

    for key in ("ground_truth", "target", "answer", "answers", "golden_answers", "gt", "gts"):
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return None


def collect_ground_truth_targets(value: Any) -> list[str]:
    targets: list[str] = []

    def add(candidate: Any) -> None:
        if candidate is None:
            return
        if isinstance(candidate, str):
            text = candidate.strip()
            if text:
                targets.append(text)
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


def add_token_usage(aggregate: dict[str, Any], usage: dict[str, int | float], model: str) -> None:
    aggregate["call_count"] = int(aggregate.get("call_count", 0)) + 1
    for key, value in usage.items():
        aggregate[key] = aggregate.get(key, 0) + value

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

    by_model = aggregate.setdefault("by_model", {})
    for model, model_other in other.get("by_model", {}).items():
        if not isinstance(model_other, dict):
            continue
        model_usage = by_model.setdefault(str(model), {"call_count": 0})
        model_usage["call_count"] = int(model_usage.get("call_count", 0)) + int(model_other.get("call_count", 0))
        for key, value in model_other.items():
            if key == "call_count" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            model_usage[key] = model_usage.get(key, 0) + value


def strip_tool_role_for_api(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    api_messages: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role == "tool":
            api_messages.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})
        elif role in {"system", "user", "assistant"}:
            api_messages.append({"role": role, "content": content})
        else:
            api_messages.append({"role": "user", "content": content})
    return api_messages


def call_chat_completion(
    *,
    messages: list[dict[str, str]],
    api_url: str,
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
    temperature: float,
    max_tokens: Optional[int],
    no_proxy: bool,
    extra_body: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": strip_tool_role_for_api(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra_body:
        payload.update(copy.deepcopy(extra_body))

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            with requests.Session() as session:
                if no_proxy:
                    session.trust_env = False
                response = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            data = response.json()
            choices = data.get("choices", []) if isinstance(data, dict) else []
            message = choices[0].get("message", {}) if choices else {}
            return str(message.get("content", "") if isinstance(message, dict) else ""), data
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 8))
    raise last_error if last_error is not None else RuntimeError("Unknown API error")


def parse_search_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for match in re.finditer(r"<search>\s*(.*?)\s*</search>", text or "", flags=re.DOTALL | re.IGNORECASE):
        block = match.group(1).strip()
        if block:
            blocks.append(block)
    return blocks


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
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", "\""}:
        text = text[1:-1].strip()
    return text


def parse_search_block_queries(search_block: Optional[str]) -> list[str]:
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


def parse_search_queries(text: str) -> list[str]:
    queries: list[str] = []
    for block in parse_search_blocks(text):
        queries.extend(parse_search_block_queries(block))
    return queries


def extract_tag_block(text: str, tag_pattern: str) -> Optional[str]:
    matches = list(re.finditer(tag_pattern, text or "", flags=re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    value = matches[-1].group(1).strip()
    return value or None


def extract_final_answer(text: str) -> str:
    patterns = [
        r"<final[_ ]answer>\s*(.*?)\s*</final[_ ]answer>",
        r"<answer>\s*(.*?)\s*</answer>",
    ]
    for pattern in patterns:
        value = extract_tag_block(text, pattern)
        if value:
            return value

    answer = str(text or "").strip().strip("`\"' \n\t")
    answer = re.sub(r"(?is)^final answer\s*[:：]\s*", "", answer).strip()
    answer = re.sub(r"(?is)^answer\s*[:：]\s*", "", answer).strip()
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(lines) > 1:
        answer = lines[-1]
    return answer.strip("`\"' \n\t")


def extract_evidence_text(text: str) -> str:
    return extract_tag_block(text, r"<evidence>\s*(.*?)\s*</evidence>") or ""


def strip_thinking_blocks(text: str) -> str:
    stripped = re.sub(r"<think>\s*.*?\s*</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    return stripped


def assistant_content_for_history(text: str, args: argparse.Namespace) -> str:
    if getattr(args, "strip_thinking_from_history", False):
        return strip_thinking_blocks(text)
    return text


def has_final_answer(text: str) -> bool:
    return bool(
        re.search(r"<final[_ ]answer>\s*.*?\s*</final[_ ]answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
        or re.search(r"<answer>\s*.*?\s*</answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    )


def extract_evidence_ids(text: str) -> list[str]:
    ids = re.findall(r"\bE\d+\b", text or "", flags=re.IGNORECASE)
    seen = set()
    ordered: list[str] = []
    for item in ids:
        evidence_id = item.upper()
        if evidence_id not in seen:
            seen.add(evidence_id)
            ordered.append(evidence_id)
    return ordered


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
    url = (
        document.get("url")
        or document.get("source_url")
        or document.get("source")
        or doc_dict.get("url")
        or doc_dict.get("source_url")
        or doc_dict.get("source")
        or ""
    )
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


def extract_structured_docs_from_api_response(api_response: dict[str, Any]) -> list[list[dict[str, Any]]]:
    raw_results = api_response.get("result", []) if isinstance(api_response, dict) else []
    if not isinstance(raw_results, list):
        raw_results = [raw_results]

    docs_by_query: list[list[dict[str, Any]]] = []
    for query_idx, retrieval in enumerate(raw_results):
        retrieval_items = retrieval if isinstance(retrieval, list) else [retrieval]
        docs: list[dict[str, Any]] = []
        for doc_idx, doc_item in enumerate(retrieval_items):
            docs.append(extract_structured_doc(doc_item, fallback_doc_id=f"q{query_idx}_doc{doc_idx}"))
        docs_by_query.append(docs)
    return docs_by_query


def docs_to_string(docs: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(docs):
        evidence_id = normalize_text(doc.get("evidence_id"))
        title = normalize_text(doc.get("title")) or f"Doc {idx + 1}"
        snippet = normalize_text(doc.get("snippet"))
        heading = f"{evidence_id} (Title: {title})" if evidence_id else f"Doc {idx + 1} (Title: {title})"
        block = heading
        if snippet:
            block = f"{block}\n{snippet}"
        blocks.append(block.strip())
    return "\n\n".join(blocks).strip()


def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    timeout: int = DEFAULT_RETRIEVAL_TIMEOUT,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    payload = {"query_list": query_list, "k": topk}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    last_error: Optional[str] = None

    for attempt in range(SEARCH_MAX_RETRIES):
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(retrieval_service_url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in {500, 502, 503, 504}:
                last_error = f"Server Error ({response.status_code}) on attempt {attempt + 1}/{SEARCH_MAX_RETRIES}"
                if attempt < SEARCH_MAX_RETRIES - 1:
                    time.sleep(SEARCH_INITIAL_RETRY_DELAY * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json(), None
        except requests.exceptions.ConnectionError as exc:
            last_error = f"Connection Error: {exc}"
            if attempt < SEARCH_MAX_RETRIES - 1:
                time.sleep(SEARCH_INITIAL_RETRY_DELAY * (attempt + 1))
            continue
        except requests.exceptions.Timeout as exc:
            last_error = f"Timeout Error: {exc}"
            if attempt < SEARCH_MAX_RETRIES - 1:
                time.sleep(SEARCH_INITIAL_RETRY_DELAY * (attempt + 1))
            continue
        except requests.exceptions.RequestException as exc:
            return None, f"API Request Error: {exc}"
        except json.JSONDecodeError as exc:
            raw = response.text if "response" in locals() else "N/A"
            return None, f"API Response JSON Decode Error: {exc}, Response: {raw[:200]}"
        except Exception as exc:
            return None, f"Unexpected Error: {exc}"

    return None, f"API Call Failed: {last_error}" if last_error else "API Call Failed after retries"


def perform_single_search(
    *,
    retrieval_service_url: str,
    query: str,
    topk: int,
    timeout: int,
    semaphore: Optional[threading.BoundedSemaphore],
) -> tuple[str, dict[str, Any]]:
    api_response = None
    error_msg = None
    try:
        if semaphore is None:
            api_response, error_msg = call_search_api(retrieval_service_url, [query], topk=topk, timeout=timeout)
        else:
            with semaphore:
                api_response, error_msg = call_search_api(retrieval_service_url, [query], topk=topk, timeout=timeout)
    except Exception as exc:
        error_msg = f"API Request Exception during batch search: {exc}"

    metadata: dict[str, Any] = {
        "query_count": 1,
        "queries": [query],
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "docs": [],
        "docs_by_query": [],
        "formatted_result": None,
    }
    result_text = json.dumps({"result": "Search request failed or timed out after retries."}, ensure_ascii=False)

    if error_msg:
        metadata["status"] = "api_error"
        result_text = json.dumps({"result": f"Search error: {error_msg}"}, ensure_ascii=False)
    elif api_response:
        metadata["api_response"] = api_response
        try:
            docs_by_query = extract_structured_docs_from_api_response(api_response)
            metadata["docs_by_query"] = docs_by_query
            if len(docs_by_query) == 1:
                metadata["docs"] = docs_by_query[0]
            total_results = sum(len(docs) for docs in docs_by_query)
            if total_results > 0:
                pretty_results = [docs_to_string(docs) for docs in docs_by_query]
                final_result = "\n---\n".join(result for result in pretty_results if result.strip())
                result_text = json.dumps({"result": final_result}, ensure_ascii=False)
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = final_result
            else:
                result_text = json.dumps({"result": "No search results found."}, ensure_ascii=False)
                metadata["status"] = "no_results"
                metadata["total_results"] = 0
        except Exception as exc:
            error_msg = f"Error processing search results: {exc}"
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
            metadata["status"] = "processing_error"
            metadata["api_request_error"] = error_msg
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps({"result": "Unknown API state (no response and no error message)."}, ensure_ascii=False)

    return result_text, metadata


def run_parallel_search_queries(
    *,
    retrieval_service_url: str,
    queries: list[str],
    topk: int,
    timeout: int,
    semaphore: Optional[threading.BoundedSemaphore],
    max_parallel_queries: int,
) -> list[dict[str, Any]]:
    indexed_queries = [(idx, query.strip()) for idx, query in enumerate(queries) if query.strip()]
    if not indexed_queries:
        return []

    max_workers = len(indexed_queries) if max_parallel_queries <= 0 else max(1, min(len(indexed_queries), max_parallel_queries))
    results: list[Optional[dict[str, Any]]] = [None] * len(indexed_queries)

    def run_one(query_index: int, query: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        result_text, metadata = perform_single_search(
            retrieval_service_url=retrieval_service_url,
            query=query,
            topk=topk,
            timeout=timeout,
            semaphore=semaphore,
        )
        return {
            "query_index": query_index,
            "query": query,
            "result_text": result_text,
            "metadata": metadata,
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
                        "query": query,
                        "result_text": json.dumps({"result": f"Search error: {type(exc).__name__}: {exc}"}, ensure_ascii=False),
                        "metadata": {
                            "query_count": 1,
                            "queries": [query],
                            "api_request_error": f"{type(exc).__name__}: {exc}",
                            "api_response": None,
                            "status": "api_error",
                            "total_results": 0,
                            "docs": [],
                            "docs_by_query": [],
                            "formatted_result": None,
                        },
                        "elapsed_seconds": 0.0,
                    }

    return [result for result in results if result is not None]


def assign_evidence_ids(docs: list[dict[str, Any]], evidence_bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for doc in docs:
        item = dict(doc)
        item["evidence_id"] = f"E{len(evidence_bank) + 1}"
        evidence_bank.append(item)
        assigned.append(item)
    return assigned


def build_initial_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": f"<question>\n{question}\n</question>"},
    ]


def build_round_message(round_idx: int, max_rounds: int, round_remaining: int, global_remaining: int) -> dict[str, str]:
    if round_remaining > 0 and global_remaining > 0:
        content = (
            f"Outer round {round_idx + 1}/{max_rounds}. "
            f"Backbone search turns remaining: {round_remaining} in this round, {global_remaining} globally. "
            "Either output exactly one <search>...</search> block, or output only "
            "<final answer>...</final answer> if the returned search results support the answer."
        )
    else:
        content = (
            f"Outer round {round_idx + 1}/{max_rounds}. Backbone search budget is exhausted for "
            "this round or globally. Do not output <search>. Use only returned search results and output only "
            "<final answer>...</final answer>; if they do not support an answer, output "
            "<final answer>Insufficient evidence</final answer>."
        )
    return {"role": "user", "content": content}


def build_invalid_message(reason: str, can_search: bool) -> dict[str, str]:
    if can_search:
        action = "one <search>...</search> block only, or a grounded final answer with a <final answer> block"
    else:
        action = "a grounded final answer with a <final answer> block; do not call search"
    return {"role": "user", "content": f"Your previous response was invalid: {reason}. Output {action}."}


def build_budget_block_message(reason: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"Search was not executed because {reason}. Do not output another <search> in this context. "
            "Use only returned search results and produce <final answer>...</final answer>; "
            "if they do not support an answer, output <final answer>Insufficient evidence</final answer>."
        ),
    }


def build_fallback_message(max_rounds: int, evidence_bank: list[dict[str, Any]]) -> dict[str, str]:
    evidence_ids = ", ".join(str(doc.get("evidence_id")) for doc in evidence_bank if doc.get("evidence_id"))
    if not evidence_ids:
        evidence_ids = "none"
    return {
        "role": "user",
        "content": (
            f"Fallback finalization after {max_rounds} outer rounds. Do not call any tool or output <search>. "
            "Base the answer only on returned search results. "
            f"Available evidence ids: {evidence_ids}. "
            "Output only <final answer>...</final answer>; if the returned search results do not support an answer, "
            "output <final answer>Insufficient evidence</final answer>."
        ),
    }


def build_tool_response_payload(query: str, metadata: dict[str, Any], result_text: str) -> dict[str, Any]:
    docs = metadata.get("docs", [])
    if not isinstance(docs, list):
        docs = []
    payload: dict[str, Any] = {
        "query": query,
        "status": str(metadata.get("status", "unknown")),
        "docs": docs,
    }
    if result_text and not docs:
        payload["raw_result_text"] = result_text
    return payload


def load_tool_config(path: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if not path or not os.path.exists(path):
        return config
    patterns = {
        "retrieval_service_url": str,
        "topk": int,
        "timeout": int,
        "max_concurrent": int,
    }
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            key = key.strip()
            if key not in patterns:
                continue
            value = raw_value.strip().strip("'\"")
            if not value:
                continue
            try:
                config[key] = patterns[key](value)
            except Exception:
                config[key] = value
    return config


def resolve_search_config(args: argparse.Namespace) -> None:
    tool_config = load_tool_config(args.tool_config)
    args.retrieval_url = args.retrieval_url or tool_config.get("retrieval_service_url") or DEFAULT_RETRIEVAL_URL
    args.topk = args.topk if args.topk is not None else int(tool_config.get("topk", DEFAULT_TOPK))
    args.retrieval_timeout = (
        args.retrieval_timeout
        if args.retrieval_timeout is not None
        else int(tool_config.get("timeout", DEFAULT_RETRIEVAL_TIMEOUT))
    )
    args.retrieval_max_concurrent = (
        args.retrieval_max_concurrent
        if args.retrieval_max_concurrent is not None
        else int(tool_config.get("max_concurrent", DEFAULT_RETRIEVAL_MAX_CONCURRENT))
    )


def validate_evidence_refs(final_output: str, evidence_bank: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_text = extract_evidence_text(final_output)
    refs = extract_evidence_ids(evidence_text or final_output)
    available = {str(doc.get("evidence_id", "")).upper() for doc in evidence_bank if doc.get("evidence_id")}
    invalid = [ref for ref in refs if ref not in available]
    return {
        "final_evidence_ids": refs,
        "available_evidence_ids": sorted(available, key=lambda x: int(x[1:]) if x[1:].isdigit() else x),
        "invalid_evidence_ids": invalid,
        "final_evidence_refs_valid": len(invalid) == 0,
        "final_answer_cites_existing_evidence": bool(refs) and len(invalid) == 0,
    }


def call_model_for_sample(
    *,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    token_usage: dict[str, Any],
    api_call_stats: list[dict[str, Any]],
    stage: str,
    outer_round: Optional[int],
    turn_in_round: Optional[int],
) -> tuple[str, Optional[str], dict[str, Any]]:
    api_started_at = time.perf_counter()
    try:
        assistant_text, raw_response = call_chat_completion(
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
        elapsed = round_seconds(time.perf_counter() - api_started_at)
        usage = extract_token_usage(raw_response)
        add_token_usage(token_usage, usage, args.model)
        api_call_stats.append(
            {
                "stage": stage,
                "outer_round": outer_round,
                "turn_in_round": turn_in_round,
                "model": args.model,
                "elapsed_seconds": elapsed,
                "usage": usage,
            }
        )
        return assistant_text, None, raw_response
    except Exception as exc:
        elapsed = round_seconds(time.perf_counter() - api_started_at)
        error = f"{type(exc).__name__}: {exc}"
        api_call_stats.append(
            {
                "stage": stage,
                "outer_round": outer_round,
                "turn_in_round": turn_in_round,
                "model": args.model,
                "elapsed_seconds": elapsed,
                "usage": {},
                "error": error,
            }
        )
        return "", error, {}


def process_one(
    index: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    search_semaphore: Optional[threading.BoundedSemaphore],
) -> dict[str, Any]:
    source_index = int(row.get("__source_index", index))
    uid = str(row.get("uid") or row.get("id") or uuid4().hex)
    question = extract_question(row)
    ground_truth = extract_ground_truth(row)
    started_at = time.perf_counter()

    token_usage = new_token_usage()
    api_call_stats: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    evidence_bank: list[dict[str, Any]] = []
    messages = build_initial_messages(question)
    total_search_calls = 0
    total_retrieval_queries = 0
    search_calls_per_outer_round = [0 for _ in range(args.max_outer_rounds)]
    natural_final_answer = False
    fallback_triggered = False
    final_output = ""
    final_answer = ""
    final_answer_source = None
    error = None

    if not question:
        final_em, final_f1, targets = compute_final_scores("", ground_truth)
        return {
            "baseline_name": BASELINE_NAME,
            "source_index": source_index,
            "uid": uid,
            "data_source": row.get("data_source"),
            "question": "",
            "ground_truth": ground_truth,
            "ground_truth_targets": targets,
            "final_output": "",
            "final_answer": "",
            "final_answer_source": None,
            "final_em": final_em,
            "final_f1": final_f1,
            "total_search_calls": 0,
            "total_retrieval_queries": 0,
            "search_calls_per_outer_round": search_calls_per_outer_round,
            "fallback_triggered": False,
            "natural_final_answer": False,
            "token_usage": token_usage,
            "api_call_stats": api_call_stats,
            "trajectory": trajectory,
            "evidence_bank": evidence_bank,
            "error": "empty question",
            "elapsed_seconds": round_seconds(time.perf_counter() - started_at),
        }

    for outer_round in range(args.max_outer_rounds):
        if final_answer:
            break

        round_event: dict[str, Any] = {
            "outer_round": outer_round,
            "assistant_turns": [],
            "search_calls": [],
            "blocked_searches": [],
        }
        trajectory.append(round_event)
        round_remaining = max(args.searches_per_outer_round - search_calls_per_outer_round[outer_round], 0)
        if outer_round >= args.max_outer_rounds - 1:
            round_remaining = 0
        global_remaining = max(args.global_search_cap - total_search_calls, 0)
        messages.append(build_round_message(outer_round, args.max_outer_rounds, round_remaining, global_remaining))

        for turn_in_round in range(args.max_assistant_turns_per_outer_round):
            if final_answer:
                break

            assistant_text, api_error, raw_response = call_model_for_sample(
                messages=messages,
                args=args,
                token_usage=token_usage,
                api_call_stats=api_call_stats,
                stage="direct_outer_round",
                outer_round=outer_round,
                turn_in_round=turn_in_round,
            )
            history_assistant_text = assistant_content_for_history(assistant_text, args)
            messages.append({"role": "assistant", "content": history_assistant_text})
            turn_event: dict[str, Any] = {
                "turn_in_round": turn_in_round,
                "response": assistant_text,
                "history_response": history_assistant_text,
                "has_final_answer": has_final_answer(assistant_text),
                "search_queries": parse_search_queries(assistant_text),
                "error": api_error,
            }
            if args.save_raw_api_response:
                turn_event["raw_api_response"] = raw_response
            round_event["assistant_turns"].append(turn_event)

            if api_error:
                error = api_error
                break

            if has_final_answer(assistant_text):
                final_output = assistant_text
                final_answer = extract_final_answer(assistant_text)
                final_answer_source = "natural"
                natural_final_answer = True
                break

            search_queries = parse_search_queries(assistant_text)
            if search_queries:
                final_round_search_disallowed = outer_round >= args.max_outer_rounds - 1
                round_cap_reached = search_calls_per_outer_round[outer_round] >= args.searches_per_outer_round
                global_cap_reached = total_search_calls >= args.global_search_cap
                if final_round_search_disallowed or round_cap_reached or global_cap_reached:
                    if final_round_search_disallowed:
                        reason = "the final outer round is reserved for no-tool finalization"
                    else:
                        reason = "the per-round search cap was reached" if round_cap_reached else "the global search cap was reached"
                    blocked = {
                        "turn_in_round": turn_in_round,
                        "queries": search_queries,
                        "reason": reason,
                        "total_search_calls": total_search_calls,
                        "round_search_calls": search_calls_per_outer_round[outer_round],
                    }
                    round_event["blocked_searches"].append(blocked)
                    turn_event["blocked_search"] = blocked
                    messages.append(build_budget_block_message(reason))
                    continue

                executed_queries = search_queries
                search_query_truncated_count = 0
                if args.max_parallel_search_queries > 0 and len(executed_queries) > args.max_parallel_search_queries:
                    search_query_truncated_count = len(executed_queries) - args.max_parallel_search_queries
                    executed_queries = executed_queries[: args.max_parallel_search_queries]
                if search_query_truncated_count:
                    turn_event["search_query_truncated_count"] = search_query_truncated_count
                    turn_event["truncated_search_queries"] = search_queries[args.max_parallel_search_queries :]

                search_started_at = time.perf_counter()
                total_search_calls += 1
                search_calls_per_outer_round[outer_round] += 1
                query_runs = run_parallel_search_queries(
                    retrieval_service_url=args.retrieval_url,
                    queries=executed_queries,
                    topk=args.topk,
                    timeout=args.retrieval_timeout,
                    semaphore=search_semaphore,
                    max_parallel_queries=args.max_parallel_search_queries,
                )
                total_retrieval_queries += len(query_runs)

                query_payloads: list[dict[str, Any]] = []
                assigned_docs_by_query: list[list[dict[str, Any]]] = []
                all_assigned_docs: list[dict[str, Any]] = []
                for run in query_runs:
                    query = str(run.get("query", ""))
                    metadata = run.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    result_text = str(run.get("result_text", ""))
                    docs = metadata.get("docs", [])
                    if not isinstance(docs, list):
                        docs = []
                    assigned_docs = assign_evidence_ids(docs, evidence_bank)
                    all_assigned_docs.extend(assigned_docs)
                    assigned_docs_by_query.append(assigned_docs)
                    metadata["docs"] = assigned_docs
                    metadata["docs_by_query"] = [assigned_docs]
                    metadata["formatted_result"] = docs_to_string(assigned_docs) if assigned_docs else metadata.get("formatted_result")
                    query_payload = build_tool_response_payload(query, metadata, result_text)
                    query_payload["query_index"] = run.get("query_index")
                    query_payload["elapsed_seconds"] = run.get("elapsed_seconds")
                    query_payloads.append(query_payload)

                tool_payload = {
                    "status": "success" if any(payload.get("docs") for payload in query_payloads) else "no_results",
                    "query_count": len(executed_queries),
                    "queries": executed_queries,
                    "results": query_payloads,
                    "docs": all_assigned_docs,
                    "docs_by_query": assigned_docs_by_query,
                }
                tool_text = json.dumps(tool_payload, ensure_ascii=False)
                messages.append({"role": "tool", "name": "search_subagent", "content": tool_text})
                search_event = {
                    "turn_in_round": turn_in_round,
                    "name": "search_subagent",
                    "arguments": {"queries": executed_queries},
                    "response": tool_payload,
                    "parallel_query_count": len(query_runs),
                    "search_query_truncated_count": search_query_truncated_count,
                    "elapsed_seconds": round_seconds(time.perf_counter() - search_started_at),
                    "counts_after_call": {
                        "total_search_calls": total_search_calls,
                        "round_search_calls": search_calls_per_outer_round[outer_round],
                        "total_retrieval_queries": total_retrieval_queries,
                    },
                }
                round_event["search_calls"].append(search_event)
                continue

            can_search = (
                outer_round < args.max_outer_rounds - 1
                and search_calls_per_outer_round[outer_round] < args.searches_per_outer_round
                and total_search_calls < args.global_search_cap
            )
            messages.append(build_invalid_message("missing <search> or <final answer>/<evidence> blocks", can_search))

    if not final_answer:
        fallback_triggered = True
        messages.append(build_fallback_message(args.max_outer_rounds, evidence_bank))
        assistant_text, api_error, raw_response = call_model_for_sample(
            messages=messages,
            args=args,
            token_usage=token_usage,
            api_call_stats=api_call_stats,
            stage="fallback_finalization",
            outer_round=None,
            turn_in_round=None,
        )
        history_assistant_text = assistant_content_for_history(assistant_text, args)
        messages.append({"role": "assistant", "content": history_assistant_text})
        fallback_event: dict[str, Any] = {
            "stage": "fallback_finalization",
            "response": assistant_text,
            "history_response": history_assistant_text,
            "has_final_answer": has_final_answer(assistant_text),
            "unexpected_search_queries": parse_search_queries(assistant_text),
            "error": api_error,
        }
        if args.save_raw_api_response:
            fallback_event["raw_api_response"] = raw_response
        trajectory.append(fallback_event)
        if api_error:
            error = api_error
        final_output = assistant_text
        final_answer = extract_final_answer(assistant_text)
        final_answer_source = "fallback"

    final_em, final_f1, targets = compute_final_scores(final_answer, ground_truth)
    evidence_validation = validate_evidence_refs(final_output, evidence_bank)
    elapsed_seconds = round_seconds(time.perf_counter() - started_at)

    result = {
        "baseline_name": BASELINE_NAME,
        "source_index": source_index,
        "uid": uid,
        "data_source": row.get("data_source"),
        "question": question,
        "ground_truth": ground_truth,
        "ground_truth_targets": targets,
        "final_output": final_output,
        "final_answer": final_answer,
        "final_answer_source": final_answer_source,
        "final_em": final_em,
        "final_f1": final_f1,
        "final_answer_em": final_em,
        "final_answer_f1": final_f1,
        "total_search_calls": total_search_calls,
        "total_retrieval_queries": total_retrieval_queries,
        "search_calls_per_outer_round": search_calls_per_outer_round,
        "fallback_triggered": fallback_triggered,
        "natural_final_answer": natural_final_answer,
        "token_usage": token_usage,
        "api_call_stats": api_call_stats,
        "trajectory": trajectory,
        "evidence_bank": evidence_bank,
        "error": error,
        "elapsed_seconds": elapsed_seconds,
        **evidence_validation,
    }
    if args.save_messages:
        result["messages"] = messages
    return to_jsonable(result)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    if count == 0:
        return {
            "count": 0,
            "final_em": 0.0,
            "final_f1": 0.0,
            "error_count": 0,
            "fallback_rate": 0.0,
            "natural_final_answer_rate": 0.0,
            "avg_total_search_calls": 0.0,
            "max_total_search_calls": 0,
            "avg_total_retrieval_queries": 0.0,
            "max_total_retrieval_queries": 0,
            "max_search_calls_in_outer_round": 0,
            "evidence_ref_invalid_count": 0,
        }

    em_values = [r.get("final_em") for r in results if isinstance(r.get("final_em"), (int, float))]
    f1_values = [r.get("final_f1") for r in results if isinstance(r.get("final_f1"), (int, float))]
    total_search_calls = [int(r.get("total_search_calls", 0)) for r in results]
    total_retrieval_queries = [int(r.get("total_retrieval_queries", 0)) for r in results]
    max_round_calls = []
    for result in results:
        per_round = result.get("search_calls_per_outer_round", [])
        if isinstance(per_round, list) and per_round:
            max_round_calls.append(max(int(x) for x in per_round))
        else:
            max_round_calls.append(0)

    aggregate_usage = new_token_usage()
    for result in results:
        merge_token_usage(aggregate_usage, result.get("token_usage", {}))

    return {
        "count": count,
        "final_em": sum(em_values) / len(em_values) if em_values else None,
        "final_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "error_count": sum(1 for r in results if r.get("error")),
        "fallback_count": sum(1 for r in results if r.get("fallback_triggered")),
        "fallback_rate": sum(1 for r in results if r.get("fallback_triggered")) / count,
        "natural_final_answer_count": sum(1 for r in results if r.get("natural_final_answer")),
        "natural_final_answer_rate": sum(1 for r in results if r.get("natural_final_answer")) / count,
        "avg_total_search_calls": sum(total_search_calls) / count,
        "max_total_search_calls": max(total_search_calls),
        "avg_total_retrieval_queries": sum(total_retrieval_queries) / count,
        "max_total_retrieval_queries": max(total_retrieval_queries),
        "max_search_calls_in_outer_round": max(max_round_calls),
        "evidence_ref_invalid_count": sum(1 for r in results if not r.get("final_evidence_refs_valid", True)),
        "evidence_cited_count": sum(1 for r in results if r.get("final_answer_cites_existing_evidence")),
        "token_usage": aggregate_usage,
    }


def save_json(data: Any, path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{BASELINE_NAME} evaluation")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input parquet/jsonl/json file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON file.")
    parser.add_argument("--env_file", default=".secrets/deepseek.env", help="Optional env file with API key.")
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--api_timeout", type=float, default=120.0)
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--no_proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tool_config", default=DEFAULT_TOOL_CONFIG)
    parser.add_argument("--retrieval_url", default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--retrieval_timeout", type=int, default=None)
    parser.add_argument("--retrieval_max_concurrent", type=int, default=None)
    parser.add_argument("--max_outer_rounds", type=int, default=DEFAULT_OUTER_ROUNDS)
    parser.add_argument("--searches_per_outer_round", type=int, default=DEFAULT_SEARCHES_PER_OUTER_ROUND)
    parser.add_argument("--global_search_cap", type=int, default=DEFAULT_GLOBAL_SEARCH_CAP)
    parser.add_argument(
        "--max_parallel_search_queries",
        type=int,
        default=DEFAULT_MAX_PARALLEL_SEARCH_QUERIES,
        help="Maximum focused queries to execute from one backbone <search> block. Use <=0 for no cap.",
    )
    parser.add_argument(
        "--max_assistant_turns_per_outer_round",
        type=int,
        default=1,
        help="Assistant generations allowed inside each outer round. Default is one backbone generation per outer round.",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--save_messages", action="store_true", help="Include full chat messages per sample.")
    parser.add_argument("--save_raw_api_response", action="store_true", help="Include raw API responses in trajectory.")
    parser.add_argument(
        "--strip_thinking_from_history",
        action="store_true",
        help="Keep raw assistant output in trajectory, but remove <think>...</think> blocks from subsequent chat history.",
    )
    parser.add_argument(
        "--disable_qwen_thinking",
        action="store_true",
        help="Pass chat_template_kwargs.enable_thinking=false to OpenAI-compatible Qwen3 servers.",
    )
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
    if args.disable_qwen_thinking:
        chat_template_kwargs = args.extra_body.setdefault("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError("extra_body_json.chat_template_kwargs must be an object when --disable_qwen_thinking is used.")
        chat_template_kwargs.setdefault("enable_thinking", False)
    load_env_file(args.env_file)
    if not args.api_key:
        args.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not args.api_key:
        raise ValueError("Missing API key: set DEEPSEEK_API_KEY or pass --api_key.")
    resolve_search_config(args)

    expected_global_cap = max(args.max_outer_rounds - 1, 0) * args.searches_per_outer_round
    if args.global_search_cap > expected_global_cap:
        raise ValueError(
            f"global_search_cap={args.global_search_cap} exceeds (max_outer_rounds-1)*searches_per_outer_round="
            f"{expected_global_cap}; this would not be budget-matched."
        )

    rows = load_records(args.input, limit=args.limit, offset=args.offset)
    for local_idx, row in enumerate(rows):
        row["__source_index"] = args.offset + local_idx

    search_semaphore = (
        threading.BoundedSemaphore(args.retrieval_max_concurrent)
        if args.retrieval_max_concurrent and args.retrieval_max_concurrent > 0
        else None
    )

    results: list[Optional[dict[str, Any]]] = [None] * len(rows)
    if args.num_workers <= 1:
        for i, row in enumerate(tqdm(rows, desc=BASELINE_NAME)):
            results[i] = process_one(i, row, args, search_semaphore)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_one, i, row, args, search_semaphore): i for i, row in enumerate(rows)}
            for future in tqdm(as_completed(futures), total=len(futures), desc=BASELINE_NAME):
                idx = futures[future]
                results[idx] = future.result()

    ordered_results = [result for result in results if result is not None]
    summary = summarize_results(ordered_results)
    output = {
        "baseline_name": BASELINE_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "model": args.model,
        "api_url": args.api_url,
        "extra_body": args.extra_body,
        "search_backend": {
            "tool_config": args.tool_config,
            "retrieval_url": args.retrieval_url,
            "topk": args.topk,
            "retrieval_timeout": args.retrieval_timeout,
            "retrieval_max_concurrent": args.retrieval_max_concurrent,
            "dedup_logic": "none; raw docs are preserved like SearchSubagentTool",
            "snippet_length_cap": None,
            "evidence_token_cap": None,
            "max_parallel_search_queries": args.max_parallel_search_queries,
            "parallel_search_enabled": True,
            "strip_thinking_from_history": args.strip_thinking_from_history,
        },
        "budget": {
            "max_outer_rounds": args.max_outer_rounds,
            "searches_per_outer_round": args.searches_per_outer_round,
            "global_search_cap": args.global_search_cap,
            "max_parallel_search_queries_per_search_turn": args.max_parallel_search_queries,
            "fallback_finalization_calls": 1,
            "fallback_allows_tools": False,
        },
        "summary": summary,
        "results": ordered_results,
    }
    save_json(output, args.output)
    print(
        f"[{BASELINE_NAME}] count={summary['count']} "
        f"EM={summary['final_em'] if summary['final_em'] is not None else 'NA'} "
        f"F1={summary['final_f1'] if summary['final_f1'] is not None else 'NA'} "
        f"fallback_rate={summary['fallback_rate']:.4f} "
        f"max_search_calls={summary['max_total_search_calls']} "
        f"max_round_search_calls={summary['max_search_calls_in_outer_round']}"
    )
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
