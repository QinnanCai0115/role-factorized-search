# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import threading
import time
import traceback
import uuid
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter

DEFAULT_TIMEOUT = 30  # Default search request timeout
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10
HTTP_POOL_SIZE = 64

logger = logging.getLogger(__name__)

# Force retrieval HTTP calls to bypass process-level proxy environment variables.
_NO_PROXY_SESSION = requests.Session()
_NO_PROXY_SESSION.trust_env = False
_HTTP_ADAPTER = HTTPAdapter(pool_connections=HTTP_POOL_SIZE, pool_maxsize=HTTP_POOL_SIZE, pool_block=True)
_NO_PROXY_SESSION.mount("http://", _HTTP_ADAPTER)
_NO_PROXY_SESSION.mount("https://", _HTTP_ADAPTER)


def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Calls the remote search API to perform retrieval with retry logic for various errors,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        return_scores: Whether to return scores.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"query_list": query_list, "k": topk}

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
            )
            response = _NO_PROXY_SESSION.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None).strip()
    return str(value).strip()


def _normalize_score(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_structured_doc(doc_item: Any, fallback_doc_id: str) -> dict[str, Any]:
    doc_dict = doc_item if isinstance(doc_item, dict) else {}
    document = doc_dict.get("document", {})
    if not isinstance(document, dict):
        document = {}

    contents = _normalize_text(document.get("contents") or doc_dict.get("contents"))
    title = _normalize_text(document.get("title") or doc_dict.get("title"))
    snippet = _normalize_text(document.get("snippet") or doc_dict.get("snippet"))

    if contents:
        lines = [line.strip() for line in contents.splitlines()]
        non_empty_lines = [line for line in lines if line]
        if non_empty_lines:
            if not title:
                title = non_empty_lines[0]
                snippet = "\n".join(non_empty_lines[1:]).strip()
            elif not snippet:
                if non_empty_lines[0] == title:
                    snippet = "\n".join(non_empty_lines[1:]).strip()
                else:
                    snippet = "\n".join(non_empty_lines).strip()

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
    score = _normalize_score(
        doc_dict.get("score")
        or doc_dict.get("retrieval_score")
        or document.get("score")
        or document.get("retrieval_score")
    )

    return {
        "doc_id": str(doc_id),
        "title": title,
        "snippet": snippet,
        "url": _normalize_text(url),
        "score": score,
    }


def extract_structured_docs_from_api_response(api_response: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Normalize retrieval API results into per-query document lists."""
    raw_results = api_response.get("result", []) if isinstance(api_response, dict) else []
    if not isinstance(raw_results, list):
        raw_results = [raw_results]

    docs_by_query: list[list[dict[str, Any]]] = []
    for query_idx, retrieval in enumerate(raw_results):
        retrieval_items = retrieval if isinstance(retrieval, list) else [retrieval]
        docs: list[dict[str, Any]] = []
        for doc_idx, doc_item in enumerate(retrieval_items):
            docs.append(_extract_structured_doc(doc_item, fallback_doc_id=f"q{query_idx}_doc{doc_idx}"))
        docs_by_query.append(docs)
    return docs_by_query


def _docs2string(docs: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(docs):
        title = _normalize_text(doc.get("title")) or f"Doc {idx + 1}"
        snippet = _normalize_text(doc.get("snippet"))
        block = f"Doc {idx + 1} (Title: {title})"
        if snippet:
            block = f"{block}\n{snippet}"
        blocks.append(block.strip())
    return "\n\n".join(blocks).strip()


def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    """
    Performs a single batch search for multiple queries (original search tool behavior).

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        concurrent_semaphore: Optional semaphore for concurrency control.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (result_text, metadata).
        result_text: The search result JSON string.
        metadata: Metadata dictionary for the batch search.
    """
    logger.info(f"Starting batch search for {len(query_list)} queries.")

    api_response = None
    error_msg = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_search_api(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query_list,
                    topk=topk,
                    return_scores=True,
                    timeout=timeout,
                )
        else:
            api_response, error_msg = call_search_api(
                retrieval_service_url=retrieval_service_url,
                query_list=query_list,
                topk=topk,
                return_scores=True,
                timeout=timeout,
            )
    except Exception as e:
        error_msg = f"API Request Exception during batch search: {e}"
        logger.error(f"Batch search: {error_msg}")
        traceback.print_exc()

    metadata = {
        "query_count": len(query_list),
        "queries": query_list,
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
        logger.error(f"Batch search: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Batch search: API Response: {api_response}")
        metadata["api_response"] = api_response

        try:
            docs_by_query = extract_structured_docs_from_api_response(api_response)
            metadata["docs_by_query"] = docs_by_query
            if len(docs_by_query) == 1:
                metadata["docs"] = docs_by_query[0]

            total_results = sum(len(docs) for docs in docs_by_query)
            if total_results > 0:
                pretty_results = [_docs2string(docs) for docs in docs_by_query]
                final_result = "\n---\n".join(result for result in pretty_results if result.strip())
                result_text = json.dumps({"result": final_result}, ensure_ascii=False)
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = final_result
                logger.info(f"Batch search: Successful, got {total_results} total results")
            else:
                result_text = json.dumps({"result": "No search results found."}, ensure_ascii=False)
                metadata["status"] = "no_results"
                metadata["total_results"] = 0
                logger.info("Batch search: No results found")
        except Exception as e:
            error_msg = f"Error processing search results: {e}"
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
            metadata["status"] = "processing_error"
            logger.error(f"Batch search: {error_msg}")
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps(
            {"result": "Unknown API state (no response and no error message)."}, ensure_ascii=False
        )
        logger.error("Batch search: Unknown API state.")

    return result_text, metadata
