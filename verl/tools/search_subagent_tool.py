# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import asyncio
import json
import logging
import os
import threading
from typing import Any, Optional
from uuid import uuid4

from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchSubagentTool(BaseTool):
    """Simple retrieval tool that returns structured raw documents.

    Intended usage:
    - The policy issues a search query.
    - The tool returns raw retrieved docs for the policy to read and summarize.
    """

    _semaphore_lock = threading.Lock()
    _shared_semaphores: dict[tuple[str, int], threading.BoundedSemaphore] = {}

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.topk = int(config.get("topk", 3))
        self.timeout = int(config.get("timeout", 30))
        self.max_concurrent = int(config.get("max_concurrent", 8))
        self._concurrent_semaphore = self._get_shared_semaphore()

    def _get_shared_semaphore(self) -> Optional[threading.BoundedSemaphore]:
        if self.max_concurrent <= 0:
            return None

        key = (self.retrieval_service_url, self.max_concurrent)
        with self._semaphore_lock:
            semaphore = self._shared_semaphores.get(key)
            if semaphore is None:
                semaphore = threading.BoundedSemaphore(self.max_concurrent)
                self._shared_semaphores[key] = semaphore
        return semaphore

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "reward": [],
            "subagents": {"0": []},
        }
        return instance_id, ToolResponse()

    def _single_round(self, query: str, topk: int, timeout: int) -> tuple[str, dict[str, Any]]:
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=self.retrieval_service_url,
            query_list=[query],
            topk=topk,
            concurrent_semaphore=self._concurrent_semaphore,
            timeout=timeout,
        )
        return result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        if not query or not isinstance(query, str):
            return ToolResponse(text=json.dumps({"error": "query must be a non-empty string"})), 0.0, {}

        # Keep retrieval breadth fixed by tool config instead of letting the policy
        # override it per call. This makes rollout behavior more stable.
        topk = self.topk
        # Keep timeout fixed by tool config so the policy cannot accidentally
        # shorten retrieval calls and amplify backend timeout/retry cascades.
        timeout = self.timeout

        instance_data = self._instance_dict[instance_id]
        trajectory = instance_data["subagents"]["0"]
        result_text, metadata = await asyncio.to_thread(self._single_round, query, topk=topk, timeout=timeout)
        docs = metadata.get("docs", [])
        if not isinstance(docs, list):
            docs = []
        status = str(metadata.get("status", "unknown"))

        call_item = {
            "query": query,
            "docs": docs,
            "status": status,
        }
        trajectory.append(call_item)

        # expose trajectory to reward via tool_extra_fields
        agent_data = kwargs.get("agent_data")
        if agent_data is not None:
            extras = agent_data.extra_fields.setdefault("extras", {})
            extras["search_subagent"] = instance_data["subagents"]

        response_payload = {
            "query": query,
            "status": status,
            "docs": docs,
        }
        if result_text and not docs:
            response_payload["raw_result_text"] = result_text

        response_text = json.dumps(response_payload, ensure_ascii=False)
        instance_data["reward"].append(response_text)
        metrics = {
            "trajectory_len": len(trajectory),
            "doc_count": len(docs),
            "status": status,
        }
        return ToolResponse(text=response_text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> Any:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
