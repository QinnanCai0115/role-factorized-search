# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import os
import re
import time
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import requests
import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.two_stage_prompts import POLICY_SYSTEM_PROMPT, build_final_policy_turn_message
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        # Keep one tool instance per tool name for the whole trajectory so
        # stateful tools (e.g. subagent slots) can continue across turns.
        self.tool_instances: dict[str, str] = {}

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.max_tool_response_docs = self.rollout_config.multi_turn.max_tool_response_docs
        self.max_tool_response_doc_chars = self.rollout_config.multi_turn.max_tool_response_doc_chars
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        configured_total_response_length = self.rollout_config.multi_turn.max_total_response_length
        self.total_response_length = self._resolve_total_response_length(configured_total_response_length)

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )
        custom_cfg = self.rollout_config.get("custom", {}) or {}
        self.policy_system_prompt = custom_cfg.get(
            "policy_system_prompt",
            POLICY_SYSTEM_PROMPT,
        )
        self.step_reward_judge_url = custom_cfg.get("step_reward_judge_url", None)
        self.step_reward_judge_timeout = float(custom_cfg.get("step_reward_judge_timeout", 20.0))
        self.tool_reward_source = str(custom_cfg.get("tool_reward_source", "mixed")).lower()
        self.backbone_judge_timeout = float(custom_cfg.get("backbone_judge_timeout", 20.0))
        self.backbone_judge_max_chars = int(custom_cfg.get("backbone_judge_max_chars", 4000))
        self.model_score_keys = list(
            custom_cfg.get(
                "model_score_keys",
                [
                    "score",
                    "relevance_score",
                    "tool_reward",
                    "reward",
                    "judge_score",
                ],
            )
        )
        self.io_trace_log_path = str(custom_cfg.get("io_trace_log_path", os.getenv("VERL_IO_TRACE_LOG_PATH", "")) or "")
        self.io_trace_max_chars = int(custom_cfg.get("io_trace_max_chars", 4000))
        self.io_trace_max_items = int(custom_cfg.get("io_trace_max_items", 6))
        self.policy_debug_trace = self._cfg_bool(
            custom_cfg.get(
                "policy_debug_trace",
                os.getenv("VERL_POLICY_DEBUG_TRACE", os.getenv("VERL_TOOL_AGENT_DEBUG_TRACE", False)),
            )
        )
        self.inject_final_policy_turn_instruction = self._cfg_bool(
            custom_cfg.get("inject_final_policy_turn_instruction", True)
        )
        self.policy_generation_stop_enabled = self._cfg_bool(
            custom_cfg.get("policy_generation_stop_enabled", False)
        )
        self.policy_generation_include_stop_str = self._cfg_bool(
            custom_cfg.get("policy_generation_include_stop_str", True)
        )
        self.policy_generation_stop_sequences = self._cfg_str_list(
            custom_cfg.get("policy_generation_stop_sequences", "</search>|</evidence>")
        )
        self.policy_reward_mode = str(custom_cfg.get("policy_reward_mode", "backbone_dense_judge")).lower()
        self.policy_format_penalty = float(custom_cfg.get("policy_format_penalty", -0.5))
        self.retrieval_effective_reward_weight = float(custom_cfg.get("retrieval_effective_reward_weight", 0.4))
        self.summary_reasonable_reward_weight = float(custom_cfg.get("summary_reasonable_reward_weight", 0.6))
        self.format_invalid_reward = float(custom_cfg.get("format_invalid_reward", -0.5))
        self.both_good_reward = float(custom_cfg.get("both_good_reward", 1.0))
        self.summary_only_reward = float(custom_cfg.get("summary_only_reward", 0.2))
        self.retrieval_only_reward = float(custom_cfg.get("retrieval_only_reward", 0.0))
        self.both_bad_reward = float(custom_cfg.get("both_bad_reward", -0.1))
        self.policy_reward_max_output_chars = int(
            custom_cfg.get(
                "policy_reward_max_output_chars",
                custom_cfg.get("policy_continue_max_output_chars", 1500),
            )
        )
        self.policy_reward_max_answer_chars = int(
            custom_cfg.get(
                "policy_reward_max_answer_chars",
                custom_cfg.get("policy_continue_max_answer_chars", 256),
            )
        )
        self.policy_reward_max_evidence_chars = int(
            custom_cfg.get(
                "policy_reward_max_evidence_chars",
                custom_cfg.get("policy_continue_max_evidence_chars", 768),
            )
        )
        self.policy_use_api = self._cfg_bool(custom_cfg.get("policy_use_api", False))
        self.policy_api_mode = str(custom_cfg.get("policy_api_mode", "openai_compatible")).lower()
        self.policy_api_url = str(custom_cfg.get("policy_api_url", ""))
        self.policy_api_model = str(custom_cfg.get("policy_api_model", "deepseek-chat"))
        self.policy_api_timeout = float(custom_cfg.get("policy_api_timeout", 120.0))
        self.policy_api_max_retries = int(custom_cfg.get("policy_api_max_retries", 3))
        self.policy_api_temperature = float(custom_cfg.get("policy_api_temperature", 0.2))
        self.policy_api_max_tokens = custom_cfg.get("policy_api_max_tokens", None)
        if str(self.policy_api_max_tokens).strip().lower() in {"", "none", "null"}:
            self.policy_api_max_tokens = None
        if self.policy_api_max_tokens is not None:
            self.policy_api_max_tokens = int(self.policy_api_max_tokens)
        self.policy_api_no_proxy = self._cfg_bool(custom_cfg.get("policy_api_no_proxy", True))
        self.policy_api_key = str(
            custom_cfg.get("policy_api_key", "")
            or os.environ.get("POLICY_API_KEY", "")
            or os.environ.get("DEEPSEEK_API_KEY", "")
        )

    def _chat_template_tool_schemas(self) -> Optional[list[dict[str, Any]]]:
        if self.tool_parser_name == "search_xml":
            return None
        return self.tool_schemas

    def _resolve_total_response_length(self, configured_total_response_length: Optional[int]) -> int:
        if configured_total_response_length is not None:
            resolved = int(configured_total_response_length)
            if resolved < int(self.response_length):
                logger.warning(
                    "max_total_response_length=%s is smaller than assistant response_length=%s; "
                    "clamping to response_length.",
                    resolved,
                    self.response_length,
                )
                resolved = int(self.response_length)
            return resolved

        observation_turns = 1
        if self.max_assistant_turns is not None:
            observation_turns = max(1, int(self.max_assistant_turns) - 1)

        # Keep the default moderate: give each potential observation turn one prompt-length block,
        # while preserving response_length as the assistant-only generation budget.
        return int(self.response_length) + int(self.prompt_length) * observation_turns

    @staticmethod
    def _cfg_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}

    @staticmethod
    def _cfg_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            separator = "|" if "|" in stripped else ","
            return [item.strip() for item in stripped.split(separator) if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        try:
            return [str(item).strip() for item in list(value) if str(item).strip()]
        except Exception:
            text = str(value).strip()
            return [text] if text else []

    def _apply_policy_generation_stop_sequences(self, sampling_params: dict[str, Any]) -> None:
        if not self.policy_generation_stop_enabled or not self.policy_generation_stop_sequences:
            return

        merged = self._cfg_str_list(sampling_params.get("stop"))
        for stop_sequence in self.policy_generation_stop_sequences:
            if stop_sequence not in merged:
                merged.append(stop_sequence)
        if merged:
            sampling_params["stop"] = merged
        if self.policy_generation_include_stop_str:
            sampling_params["include_stop_str_in_output"] = True

    @staticmethod
    def _assistant_response_tokens(agent_data: AgentData) -> int:
        return int(sum(agent_data.response_mask))

    @staticmethod
    def _total_response_tokens(agent_data: AgentData) -> int:
        return len(agent_data.response_mask)

    def _remaining_assistant_budget(self, agent_data: AgentData) -> int:
        return max(int(self.response_length) - self._assistant_response_tokens(agent_data), 0)

    def _remaining_total_response_window(self, agent_data: AgentData) -> int:
        return max(int(self.total_response_length) - self._total_response_tokens(agent_data), 0)

    def _remaining_total_context_budget(self, agent_data: AgentData) -> int:
        return max(int(self.prompt_length) + int(self.total_response_length) - len(agent_data.prompt_ids), 0)

    def _remaining_append_budget(self, agent_data: AgentData) -> int:
        return min(
            self._remaining_total_response_window(agent_data),
            self._remaining_total_context_budget(agent_data),
        )

    def _build_turn_sampling_params(
        self,
        base_sampling_params: dict[str, Any],
        agent_data: AgentData,
    ) -> tuple[dict[str, Any], int, int, int]:
        assistant_budget = self._remaining_assistant_budget(agent_data)
        response_window_budget = self._remaining_total_response_window(agent_data)
        append_budget = self._remaining_append_budget(agent_data)
        turn_sampling_params = dict(base_sampling_params)

        explicit_limit = None
        for key in ("max_tokens", "max_new_tokens"):
            value = turn_sampling_params.pop(key, None)
            if value is None:
                continue
            explicit_limit = int(value) if explicit_limit is None else min(explicit_limit, int(value))

        max_tokens = min(assistant_budget, append_budget)
        if explicit_limit is not None:
            max_tokens = min(max_tokens, explicit_limit)
        turn_sampling_params["max_tokens"] = max_tokens
        self._apply_policy_generation_stop_sequences(turn_sampling_params)
        return turn_sampling_params, assistant_budget, response_window_budget, append_budget

    def _truncate_observation_tokens_to_budget(
        self,
        agent_data: AgentData,
        response_ids: list[int],
        *,
        metric_key: str,
    ) -> tuple[list[int], bool]:
        append_budget = self._remaining_append_budget(agent_data)
        if append_budget <= 0:
            return [], bool(response_ids)
        if len(response_ids) <= append_budget:
            return response_ids, False

        agent_data.metrics[metric_key] = agent_data.metrics.get(metric_key, 0) + 1
        return response_ids[:append_budget], True

    def _append_zero_mask_tokens(self, agent_data: AgentData, response_ids: list[int]) -> None:
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

    def _truncate_trace_value(self, value: Any, depth: int = 0) -> Any:
        if depth > 3:
            return "...(max_depth)"
        if isinstance(value, str):
            if self.io_trace_max_chars <= 0 or len(value) <= self.io_trace_max_chars:
                return value
            return value[: self.io_trace_max_chars] + "...(truncated)"
        if isinstance(value, dict):
            out = {}
            items = list(value.items())
            limit = len(items) if self.io_trace_max_items <= 0 else self.io_trace_max_items
            for k, v in items[:limit]:
                out[str(k)] = self._truncate_trace_value(v, depth + 1)
            if self.io_trace_max_items > 0 and len(items) > self.io_trace_max_items:
                out["_truncated_items"] = len(items) - self.io_trace_max_items
            return out
        if isinstance(value, (list, tuple)):
            seq = list(value)
            limit = len(seq) if self.io_trace_max_items <= 0 else self.io_trace_max_items
            out = [self._truncate_trace_value(v, depth + 1) for v in seq[:limit]]
            if self.io_trace_max_items > 0 and len(seq) > self.io_trace_max_items:
                out.append(f"...(truncated_items={len(seq) - self.io_trace_max_items})")
            return out
        return value

    def _append_io_trace(self, event: str, payload: dict[str, Any]) -> None:
        if not self.io_trace_log_path:
            return
        try:
            log_dir = os.path.dirname(self.io_trace_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "payload": self._truncate_trace_value(payload),
            }
            with open(self.io_trace_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append IO trace log: {e}")

    def _inject_policy_system_prompt(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject policy-specific retrieval rules into system prompt context."""
        if not self.policy_system_prompt:
            return messages

        prompt = str(self.policy_system_prompt).strip()
        if not prompt:
            return messages

        injected = {"role": "system", "content": prompt}
        if messages and messages[0].get("role") == "system":
            first_content = messages[0].get("content", "")
            if isinstance(first_content, str) and prompt in first_content:
                return messages
            merged = [dict(messages[0])]
            existing = merged[0].get("content", "")
            merged[0]["content"] = f"{existing}\n\n{prompt}" if existing else prompt
            merged.extend(messages[1:])
            return merged

        return [injected, *messages]

    def _build_policy_api_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Build OpenAI-compatible messages for API policy rollout.

        Tool observations are represented as user messages because DeepSeek's chat
        endpoint may reject bare `tool` role messages without OpenAI tool-call ids.
        """
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user") or "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            else:
                content = str(content)
            if role == "tool":
                api_messages.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})
            elif role in {"system", "user", "assistant"}:
                api_messages.append({"role": role, "content": content})
            else:
                api_messages.append({"role": "user", "content": content})
        return api_messages

    def _call_policy_api(self, messages: list[dict[str, Any]], max_tokens: int) -> tuple[str, dict[str, Any]]:
        if self.policy_api_mode != "openai_compatible":
            raise ValueError(f"Unsupported policy_api_mode={self.policy_api_mode!r}")
        if not self.policy_api_url:
            raise ValueError("policy_use_api=true requires actor_rollout_ref.rollout.custom.policy_api_url")
        if not self.policy_api_model:
            raise ValueError("policy_use_api=true requires actor_rollout_ref.rollout.custom.policy_api_model")

        endpoint = f"{self.policy_api_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.policy_api_key:
            headers["Authorization"] = f"Bearer {self.policy_api_key}"

        payload: dict[str, Any] = {
            "model": self.policy_api_model,
            "messages": self._build_policy_api_messages(messages),
            "temperature": self.policy_api_temperature,
            "max_tokens": max(1, max_tokens),
        }
        if self.policy_api_max_tokens is not None:
            payload["max_tokens"] = min(int(payload["max_tokens"]), int(self.policy_api_max_tokens))
        if self.policy_generation_stop_enabled and self.policy_generation_stop_sequences:
            payload["stop"] = list(self.policy_generation_stop_sequences)
            if self.policy_generation_include_stop_str:
                payload["include_stop_str_in_output"] = True

        last_error: Optional[Exception] = None
        for attempt in range(self.policy_api_max_retries + 1):
            try:
                with requests.Session() as session:
                    if self.policy_api_no_proxy:
                        session.trust_env = False
                    response = session.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=self.policy_api_timeout,
                    )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", []) if isinstance(data, dict) else []
                message = choices[0].get("message", {}) if choices else {}
                content = message.get("content", "") if isinstance(message, dict) else ""
                return str(content or ""), data
            except Exception as exc:
                last_error = exc
                if attempt < self.policy_api_max_retries:
                    time.sleep(min(2**attempt, 8))

        raise last_error if last_error is not None else RuntimeError("Unknown policy API error")

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_forced_search_query(self, messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            m = re.search(r"<question>\s*(.*?)\s*</question>", content, flags=re.DOTALL)
            if m:
                q = m.group(1).strip()
                if q:
                    return q
            stripped = content.strip()
            if stripped:
                return stripped
        return ""

    @staticmethod
    def _extract_orchestrator_trace(tools_kwargs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tools_kwargs, dict):
            return {}
        trace = tools_kwargs.get("_orchestrator_trace", {})
        if isinstance(trace, dict):
            return trace
        return {}

    def _extract_model_score(
        self,
        tool_args: dict[str, Any],
        tool_response_text: Optional[str],
        tool_meta: Optional[dict[str, Any]],
    ) -> Optional[float]:
        """Extract model-provided relevance/reward score from tool args or tool output payloads."""
        for key in self.model_score_keys:
            if key in tool_args:
                score = self._safe_float(tool_args.get(key))
                if score is not None:
                    return score

        if isinstance(tool_meta, dict):
            for key in self.model_score_keys:
                if key in tool_meta:
                    score = self._safe_float(tool_meta.get(key))
                    if score is not None:
                        return score

        if tool_response_text:
            try:
                payload = json.loads(tool_response_text)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                for key in self.model_score_keys:
                    if key in payload:
                        score = self._safe_float(payload.get(key))
                        if score is not None:
                            return score
        return None

    def _prepare_tool_response_for_policy(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_response_text: Optional[str],
    ) -> str:
        """Pass raw tool output back to the policy so the policy model does its own evidence organization."""
        raw_text = tool_response_text or ""
        if raw_text:
            return raw_text

        query = str(tool_args.get("query", "")).strip()
        if query:
            return json.dumps({"query": query, "docs": []}, ensure_ascii=False)
        if tool_name:
            return json.dumps({"tool_name": tool_name, "docs": []}, ensure_ascii=False)
        return json.dumps({"docs": []}, ensure_ascii=False)

    @staticmethod
    def _parse_tool_arguments_for_trace(arguments: Any) -> Any:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except Exception:
                return arguments
        return deepcopy(arguments)

    def _append_policy_judge_event(self, agent_data: AgentData, event: dict[str, Any]) -> None:
        trace = agent_data.extra_fields.setdefault("policy_judge_trace", [])
        if isinstance(trace, list):
            trace.append(event)

    def _append_policy_debug_trace(self, event: str, payload: dict[str, Any]) -> None:
        if not self.policy_debug_trace:
            return
        self._append_io_trace(event, payload)

    @staticmethod
    def _build_final_policy_turn_message() -> dict[str, str]:
        return build_final_policy_turn_message()

    def _is_final_policy_turn(self, agent_data: AgentData) -> bool:
        if self.max_assistant_turns is None:
            return False
        max_assistant_turns = int(self.max_assistant_turns)
        if max_assistant_turns <= 0:
            return False
        return agent_data.assistant_turns + 1 >= max_assistant_turns

    @staticmethod
    def _message_content_len(message: dict[str, Any]) -> int:
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content)
        try:
            return len(json.dumps(content, ensure_ascii=False, default=str))
        except Exception:
            return len(str(content))

    def _maybe_add_final_policy_turn_message(
        self,
        agent_data: AgentData,
        add_messages: list[dict[str, Any]],
    ) -> Optional[dict[str, str]]:
        if agent_data.extra_fields.get("final_policy_turn_instruction_appended"):
            return None
        if not self.inject_final_policy_turn_instruction:
            return None
        if not self._is_final_policy_turn(agent_data):
            return None

        message = self._build_final_policy_turn_message()
        add_messages.append(message)
        return message

    def _mark_final_policy_turn_instruction_appended(
        self,
        agent_data: AgentData,
        message: dict[str, str],
        token_count: Optional[int],
    ) -> None:
        agent_data.extra_fields["final_policy_turn_instruction_appended"] = True
        self._append_policy_judge_event(
            agent_data,
            {
                "stage": "final_policy_turn_instruction",
                "assistant_turn": agent_data.assistant_turns + 1,
                "message": message["content"],
                "token_count": token_count,
            },
        )

    def _build_policy_budget_debug_payload(
        self,
        agent_data: AgentData,
        orchestrator_trace: dict[str, Any],
        *,
        assistant_turn_id: Optional[int] = None,
        tool_call_count: Optional[int] = None,
        original_response_len: Optional[int] = None,
        append_budget: Optional[int] = None,
        tool_response_len: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload = {
            "request_id": agent_data.request_id,
            "trace_id": orchestrator_trace.get("trace_id", ""),
            "orchestrator_round": orchestrator_trace.get("round", None),
            "source_uid": agent_data.extra_fields.get("source_uid", None)
            or agent_data.tools_kwargs.get("source_uid", None)
            or orchestrator_trace.get("source_uid", None),
            "global_step": orchestrator_trace.get(
                "global_step",
                agent_data.extra_fields.get("global_step", agent_data.extra_fields.get("global_steps", None)),
            ),
            "assistant_turn_id": assistant_turn_id,
            "tool_call_count": tool_call_count,
            "max_assistant_turns": self.max_assistant_turns,
            "original_response_len": original_response_len,
            "append_budget": append_budget,
            "prompt_len": len(agent_data.prompt_ids),
            "response_tokens_so_far": self._total_response_tokens(agent_data),
            "assistant_tokens_so_far": self._assistant_response_tokens(agent_data),
            "tool_response_len": tool_response_len,
            "remaining_response_budget": self._remaining_assistant_budget(agent_data),
            "finish_reason": agent_data.extra_fields.get(
                "last_finish_reason", agent_data.extra_fields.get("finish_reason")
            ),
            "stop_reason": agent_data.extra_fields.get(
                "last_stop_reason", agent_data.extra_fields.get("stop_reason")
            ),
        }
        if extra:
            payload.update(extra)
        return payload

    @staticmethod
    def _extract_last_answer_evidence_pair(text: str) -> str:
        merged = str(text or "").strip()
        if not merged:
            return ""
        pair_pattern = re.compile(
            r"(<answer>.*?</answer>\s*<evidence>.*?</evidence>)",
            flags=re.DOTALL | re.IGNORECASE,
        )
        pairs = pair_pattern.findall(merged)
        if not pairs:
            return ""
        return pairs[-1].strip()

    @classmethod
    def _has_complete_answer_evidence_output(cls, text: str) -> bool:
        return bool(cls._extract_last_answer_evidence_pair(text))

    @staticmethod
    def _has_strict_final_answer_evidence_output(text: str) -> bool:
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

    async def _append_text_observation(
        self,
        agent_data: AgentData,
        message: dict[str, Any],
        *,
        metric_key: str,
        truncated_flag_key: str,
    ) -> bool:
        add_messages = [message]
        orchestrator_trace = self._extract_orchestrator_trace(agent_data.tools_kwargs)
        final_instruction_message = self._maybe_add_final_policy_turn_message(agent_data, add_messages)
        response_ids = await self.apply_chat_template(
            add_messages,
            images=None,
            videos=None,
            remove_system_prompt=True,
        )
        original_response_len = len(response_ids)
        append_budget = self._remaining_append_budget(agent_data)
        if final_instruction_message is not None and append_budget < original_response_len:
            agent_data.extra_fields["final_policy_turn_instruction_append_failed"] = True
            agent_data.metrics["final_policy_turn_instruction_append_failed"] = 1
            self._append_policy_debug_trace(
                "policy.final_turn_instruction_append_failed",
                self._build_policy_budget_debug_payload(
                    agent_data,
                    orchestrator_trace,
                    assistant_turn_id=agent_data.assistant_turns + 1,
                    tool_call_count=len(agent_data.tool_calls or []),
                    original_response_len=original_response_len,
                    append_budget=append_budget,
                    tool_response_len=None,
                    extra={
                        "reason": "insufficient_append_budget",
                        "message": final_instruction_message["content"],
                    },
                ),
            )
            return False
        response_ids, truncated = self._truncate_observation_tokens_to_budget(
            agent_data,
            response_ids,
            metric_key=metric_key,
        )
        if not response_ids:
            return False

        agent_data.messages.extend(add_messages)
        self._append_zero_mask_tokens(agent_data, response_ids)
        agent_data.user_turns += 1
        if final_instruction_message is not None:
            agent_data.user_turns += 1
            self._mark_final_policy_turn_instruction_appended(
                agent_data,
                final_instruction_message,
                token_count=None,
            )
        if truncated:
            agent_data.extra_fields[truncated_flag_key] = True
        return True

    def _truncate_tool_doc_text(self, value: Any, max_chars: int) -> str:
        text = str(value or "").strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def _truncate_structured_tool_response(self, tool_name: str, tool_response_text: str) -> Optional[str]:
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

        def _truncate_doc(doc: dict[str, Any], max_chars: int) -> dict[str, Any]:
            return {
                "doc_id": str(doc.get("doc_id", "")),
                "title": self._truncate_tool_doc_text(doc.get("title", ""), max_chars),
                "snippet": self._truncate_tool_doc_text(doc.get("snippet", ""), max_chars),
                "url": self._truncate_tool_doc_text(doc.get("url", ""), max_chars),
                "score": doc.get("score"),
            }

        def _truncate_doc_list(docs: Any, max_docs: int, max_chars: int) -> list[dict[str, Any]]:
            if not isinstance(docs, list):
                return []
            limit = len(docs) if max_docs <= 0 else max_docs
            truncated_docs: list[dict[str, Any]] = []
            for doc in docs[:limit]:
                if isinstance(doc, dict):
                    truncated_docs.append(_truncate_doc(doc, max_chars))
            return truncated_docs

        def _build_payload(max_docs: int, max_chars: int) -> dict[str, Any]:
            truncated = deepcopy(payload)

            if isinstance(truncated.get("docs"), list):
                truncated["docs"] = _truncate_doc_list(truncated["docs"], max_docs, max_chars)
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
                        new_item["docs"] = _truncate_doc_list(item["docs"], max_docs, max_chars)
                    raw_result_text = item.get("raw_result_text")
                    if raw_result_text and not new_item.get("docs"):
                        new_item["raw_result_text"] = self._truncate_tool_doc_text(raw_result_text, max_chars * 2)
                    new_round_results.append(new_item)
                truncated["round_results"] = new_round_results

            raw_result_text = truncated.get("raw_result_text")
            if raw_result_text and not truncated.get("docs"):
                truncated["raw_result_text"] = self._truncate_tool_doc_text(raw_result_text, max_chars * 2)

            return truncated

        current_max_docs = max(1, int(self.max_tool_response_docs))
        current_max_chars = max(120, int(self.max_tool_response_doc_chars))
        text = json.dumps(_build_payload(current_max_docs, current_max_chars), ensure_ascii=False)

        while len(text) > self.max_tool_response_length and (current_max_chars > 120 or current_max_docs > 1):
            if current_max_chars > 120:
                current_max_chars = max(120, current_max_chars // 2)
            elif current_max_docs > 1:
                current_max_docs -= 1
            text = json.dumps(_build_payload(current_max_docs, current_max_chars), ensure_ascii=False)

        return text

    def _truncate_text_for_backbone_judge(self, text: str) -> str:
        max_chars = int(self.backbone_judge_max_chars)
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "...(truncated)"

    def _prepare_tool_response_for_backbone_judge(self, tool_name: str, tool_response_text: Optional[str]) -> str:
        text = (tool_response_text or "").strip()
        if not text:
            return ""
        structured = self._truncate_structured_tool_response(tool_name, text)
        if structured is not None:
            return structured
        return self._truncate_text_for_backbone_judge(text)

    def _build_policy_chain_for_backbone_judge(self, agent_data: AgentData, final_response_text: str) -> str:
        context = deepcopy(agent_data.extra_fields.get("policy_judge_context", {}))
        trace = deepcopy(agent_data.extra_fields.get("policy_judge_trace", []))

        for event in trace:
            if not isinstance(event, dict):
                continue
            stage = str(event.get("stage", ""))
            if stage == "assistant_output":
                response_text = event.get("response_text")
                if isinstance(response_text, str):
                    event["response_text"] = self._truncate_text_for_backbone_judge(response_text)
            elif stage == "tool_result":
                tool_name = str(event.get("tool_name", ""))
                raw_tool_response_text = event.get("raw_tool_response_text")
                if isinstance(raw_tool_response_text, str):
                    event["raw_tool_response_text"] = self._prepare_tool_response_for_backbone_judge(
                        tool_name, raw_tool_response_text
                    )
                policy_visible_tool_response_text = event.get("policy_visible_tool_response_text")
                if isinstance(policy_visible_tool_response_text, str):
                    event["policy_visible_tool_response_text"] = self._prepare_tool_response_for_backbone_judge(
                        tool_name, policy_visible_tool_response_text
                    )

        payload = {
            "policy_prompt": context.get("initial_messages", []),
            "policy_trace": trace,
            # Use the full assistant-side accumulated round output here so the judge sees
            # the completed policy multi-turn loop rather than only the last assistant turn.
            "final_policy_output": self._truncate_text_for_backbone_judge(final_response_text),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _judge_policy_chain_with_backbone_binary(
        self,
        messages: list[dict[str, Any]],
        policy_chain_text: str,
    ) -> tuple[Optional[float], dict[str, Any]]:
        custom_cfg = self.rollout_config.get("custom", {}) or {}
        api_mode = str(custom_cfg.get("backbone_api_mode", "openai_compatible")).lower()
        api_url = str(custom_cfg.get("backbone_api_url", ""))
        api_model = str(custom_cfg.get("backbone_api_model", ""))
        api_key = str(
            custom_cfg.get("backbone_api_key", "")
            or os.environ.get("BACKBONE_API_KEY", "")
            or os.environ.get("DEEPSEEK_API_KEY", "")
        )
        if not api_url or not api_model or api_mode != "openai_compatible":
            return None, {}

        question = self._extract_forced_search_query(messages)
        judge_system = (
            "You are a strict binary judge for policy retrieval quality and evidence summarization quality. "
            "Return only JSON."
        )
        judge_user = (
            f"Question:\n{question}\n\n"
            "You are given the full policy chain from receiving the backbone search request to producing the final "
            "answer and evidence. Judge two things:\n"
            "1. retrieval_effective: whether the policy's search queries were reasonable and the retrieved content was "
            "useful for answering the question.\n"
            "2. summary_reasonable: whether the policy's final answer and evidence are faithful to and reasonably "
            "supported by the retrieved content.\n\n"
            "Set score=1 only if both retrieval_effective=1 and summary_reasonable=1. Otherwise set score=0.\n\n"
            "Return JSON in this exact schema:\n"
            "{\"score\": 0 or 1, \"retrieval_effective\": 0 or 1, \"summary_reasonable\": 0 or 1, "
            "\"reason\": \"short explanation\"}\n\n"
            "Policy chain JSON:\n"
            f"{policy_chain_text}"
        )

        endpoint = f"{api_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": api_model,
            "messages": [
                {"role": "system", "content": judge_system},
                {"role": "user", "content": judge_user},
            ],
            "temperature": 0.0,
        }

        try:
            disable_backbone_proxy = str(os.environ.get("BACKBONE_API_NO_PROXY", "")).strip().lower() not in (
                "",
                "0",
                "false",
                "no",
            )
            with requests.Session() as session:
                if disable_backbone_proxy:
                    session.trust_env = False
                resp = session.post(endpoint, json=payload, headers=headers, timeout=self.backbone_judge_timeout)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", []) if isinstance(data, dict) else []
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content", "") if isinstance(message, dict) else ""

            details: dict[str, Any] = {"raw_judge_response": content}
            try:
                parsed = json.loads(content)
                score = int(parsed.get("score", 0))
                retrieval_effective = int(parsed.get("retrieval_effective", score))
                summary_reasonable = int(parsed.get("summary_reasonable", score))
                details.update(
                    {
                        "retrieval_effective": 1 if retrieval_effective == 1 else 0,
                        "summary_reasonable": 1 if summary_reasonable == 1 else 0,
                        "reason": str(parsed.get("reason", "")).strip(),
                    }
                )
                return float(1 if score == 1 else 0), details
            except Exception:
                m = re.search(r"\b([01])\b", str(content))
                if m:
                    score = int(m.group(1))
                    details.update(
                        {
                            "retrieval_effective": 1 if score == 1 else 0,
                            "summary_reasonable": 1 if score == 1 else 0,
                            "reason": "",
                        }
                    )
                    return float(score), details
                return None, details
        except Exception as e:
            logger.warning(f"Backbone binary judge failed: {e}")
            return None, {"error": str(e)}

    async def _judge_final_policy_output_with_backbone_binary(
        self, agent_data: AgentData, final_response_text: str
    ) -> tuple[float, dict[str, Any]]:
        """Judge the full policy chain with the backbone binary scorer."""
        policy_chain_text = self._build_policy_chain_for_backbone_judge(agent_data, final_response_text)
        judged, details = await self.loop.run_in_executor(
            None,
            lambda: self._judge_policy_chain_with_backbone_binary(
                messages=agent_data.messages,
                policy_chain_text=policy_chain_text,
            ),
        )
        details = details or {}
        details["policy_chain"] = policy_chain_text
        details["policy_full_trace_output"] = policy_chain_text
        return 0.0 if judged is None else float(judged), details

    def _compute_policy_output_format_penalty(self, agent_data: AgentData) -> tuple[float, dict[str, Any]]:
        last_assistant_response = str(agent_data.extra_fields.get("last_assistant_response_text", "") or "")
        has_valid_answer_evidence = self._has_strict_final_answer_evidence_output(last_assistant_response)
        penalty = 0.0 if has_valid_answer_evidence else float(self.policy_format_penalty)
        return penalty, {
            "has_valid_answer_evidence_output": has_valid_answer_evidence,
            "policy_format_penalty": penalty,
            "last_assistant_response_text": last_assistant_response,
        }

    @staticmethod
    def _coerce_judge_bit(value: Any, fallback: float) -> float:
        try:
            return 1.0 if float(value) > 0.5 else 0.0
        except (TypeError, ValueError):
            return 1.0 if float(fallback) > 0.5 else 0.0

    def _compute_dense_backbone_judge_reward(
        self,
        *,
        final_binary_score: float,
        final_binary_details: dict[str, Any],
        format_penalty: float,
    ) -> tuple[float, dict[str, float]]:
        retrieval_effective = self._coerce_judge_bit(
            final_binary_details.get("retrieval_effective"), final_binary_score
        )
        summary_reasonable = self._coerce_judge_bit(
            final_binary_details.get("summary_reasonable"), final_binary_score
        )
        retrieval_reward = self.retrieval_effective_reward_weight * retrieval_effective
        summary_reward = self.summary_reasonable_reward_weight * summary_reasonable
        dense_reward = retrieval_reward + summary_reward + float(format_penalty)
        return dense_reward, {
            "retrieval_effective": retrieval_effective,
            "summary_reasonable": summary_reasonable,
            "retrieval_effective_reward_weight": float(self.retrieval_effective_reward_weight),
            "summary_reasonable_reward_weight": float(self.summary_reasonable_reward_weight),
            "retrieval_effective_reward": float(retrieval_reward),
            "summary_reasonable_reward": float(summary_reward),
            "dense_policy_reward": float(dense_reward),
            "policy_reward_mode": 2.0,
        }

    def _compute_discrete_backbone_judge_reward(
        self,
        *,
        final_binary_score: float,
        final_binary_details: dict[str, Any],
        format_penalty: float,
        final_response_text: str,
    ) -> tuple[float, dict[str, float]]:
        retrieval_effective = self._coerce_judge_bit(
            final_binary_details.get("retrieval_effective"), final_binary_score
        )
        summary_reasonable = self._coerce_judge_bit(
            final_binary_details.get("summary_reasonable"), final_binary_score
        )
        text = str(final_response_text or "")
        answer_matches = list(re.finditer(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE))
        evidence_matches = list(re.finditer(r"<evidence>(.*?)</evidence>", text, flags=re.DOTALL | re.IGNORECASE))
        answer = answer_matches[0].group(1).strip() if len(answer_matches) == 1 else ""
        evidence = evidence_matches[0].group(1).strip() if len(evidence_matches) == 1 else ""
        answer_too_long = len(answer) > int(self.policy_reward_max_answer_chars)
        evidence_too_long = len(evidence) > int(self.policy_reward_max_evidence_chars)
        output_too_long = len(text) > int(self.policy_reward_max_output_chars)
        strict_format_ok = False
        if len(answer_matches) == 1 and len(evidence_matches) == 1 and answer and evidence:
            answer_match = answer_matches[0]
            evidence_match = evidence_matches[0]
            answer_before_evidence = (
                answer_match.start() < answer_match.end() <= evidence_match.start() < evidence_match.end()
            )
            trailing = text[evidence_match.end() :].strip()
            search_after_answer = re.search(r"</?search\b", text[answer_match.start() :], flags=re.IGNORECASE)
            strict_format_ok = bool(
                answer_before_evidence
                and not trailing
                and not search_after_answer
                and not answer_too_long
                and not evidence_too_long
            )

        format_invalid = bool(format_penalty < 0.0 or not strict_format_ok or output_too_long)

        if format_invalid:
            score = float(self.format_invalid_reward)
            reward_case = 0.0
        elif retrieval_effective > 0.5 and summary_reasonable > 0.5:
            score = float(self.both_good_reward)
            reward_case = 1.0
        elif retrieval_effective <= 0.5 and summary_reasonable > 0.5:
            score = float(self.summary_only_reward)
            reward_case = 2.0
        elif retrieval_effective > 0.5 and summary_reasonable <= 0.5:
            score = float(self.retrieval_only_reward)
            reward_case = 3.0
        else:
            score = float(self.both_bad_reward)
            reward_case = 4.0

        return score, {
            "retrieval_effective": retrieval_effective,
            "summary_reasonable": summary_reasonable,
            "strict_format_ok": float(strict_format_ok),
            "output_too_long": float(output_too_long),
            "answer_too_long": float(answer_too_long),
            "evidence_too_long": float(evidence_too_long),
            "format_invalid": float(format_invalid),
            "discrete_policy_reward": float(score),
            "reward_case": float(reward_case),
            "policy_reward_mode": 3.0,
        }

    def _compute_backbone_judge_policy_reward(
        self,
        *,
        final_binary_score: float,
        final_binary_details: dict[str, Any],
        format_penalty: float,
        final_response_text: str,
    ) -> tuple[float, dict[str, float]]:
        if self.policy_reward_mode in {"backbone_discrete_judge", "backbone_discrete_only"}:
            return self._compute_discrete_backbone_judge_reward(
                final_binary_score=final_binary_score,
                final_binary_details=final_binary_details,
                format_penalty=format_penalty,
                final_response_text=final_response_text,
            )
        return self._compute_dense_backbone_judge_reward(
            final_binary_score=final_binary_score,
            final_binary_details=final_binary_details,
            format_penalty=format_penalty,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        messages = self._inject_policy_system_prompt(messages)

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        orchestrator_trace = self._extract_orchestrator_trace(tools_kwargs)
        is_validation_rollout = bool(kwargs.get("validate", False))

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        agent_data.extra_fields["policy_judge_context"] = {
            "initial_messages": deepcopy(messages),
        }
        agent_data.extra_fields["policy_judge_trace"] = []
        agent_data.extra_fields["validate"] = is_validation_rollout
        agent_data.extra_fields["assistant_response_length"] = int(self.response_length)
        agent_data.extra_fields["max_response_length"] = int(self.total_response_length)
        extra_info = kwargs.get("extra_info", {}) or {}
        if isinstance(extra_info, dict):
            for key in ("source_uid", "uid", "global_step", "global_steps"):
                if key in extra_info and key not in agent_data.extra_fields:
                    agent_data.extra_fields[key] = extra_info[key]
        self._append_io_trace(
            "policy.run_start",
            {
                "request_id": request_id,
                "trace_id": orchestrator_trace.get("trace_id", ""),
                "orchestrator_round": orchestrator_trace.get("round", None),
                "orchestrator_global_step": orchestrator_trace.get("global_step", None),
                "raw_prompt": messages,
                "tools_kwargs": tools_kwargs,
                "sampling_params": sampling_params,
                "assistant_response_length": self.response_length,
                "max_total_response_length": self.total_response_length,
            },
        )

        try:
            # State machine loop
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)
                else:
                    logger.error(f"Invalid state: {state}")
                    state = AgentState.TERMINATED

            # Finalize output
            response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
            prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
            multi_modal_data = {}
            if agent_data.image_data is not None:
                multi_modal_data["images"] = agent_data.image_data
            if agent_data.video_data is not None:
                multi_modal_data["videos"] = agent_data.video_data

            output: AgentLoopOutput = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.total_response_length],
                response_mask=agent_data.response_mask[: self.total_response_length],
                multi_modal_data=multi_modal_data,
                response_logprobs=agent_data.response_logprobs[: self.total_response_length]
                if agent_data.response_logprobs
                else None,
                num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                metrics=agent_data.metrics,
                routed_experts=agent_data.routed_experts,
                extra_fields=agent_data.extra_fields,
            )
            if self.tool_reward_source == "backbone_binary":
                if is_validation_rollout:
                    output.extra_fields["final_backbone_binary_score"] = None
                    output.extra_fields["final_backbone_binary_details"] = {
                        "skipped": True,
                        "skip_reason": "validation_rollout",
                    }
                    self._append_io_trace(
                        "policy.final_binary_judge_skipped",
                        {
                            "request_id": agent_data.request_id,
                            "trace_id": orchestrator_trace.get("trace_id", ""),
                            "orchestrator_round": orchestrator_trace.get("round", None),
                            "reason": "validation_rollout",
                        },
                    )
                else:
                    format_penalty, format_details = self._compute_policy_output_format_penalty(agent_data)
                    valid_generated_ids = [
                        int(tok)
                        for tok, mask in zip(output.response_ids, output.response_mask, strict=False)
                        if int(mask) == 1
                    ]
                    final_response_text = (
                        await self.loop.run_in_executor(
                            None, lambda: self.tokenizer.decode(valid_generated_ids, skip_special_tokens=True)
                        )
                        if valid_generated_ids
                        else ""
                    )
                    final_binary_score, final_binary_details = await self._judge_final_policy_output_with_backbone_binary(
                        agent_data=agent_data,
                        final_response_text=final_response_text,
                    )
                    reward_value, reward_details = self._compute_backbone_judge_policy_reward(
                        final_binary_score=final_binary_score,
                        final_binary_details=final_binary_details,
                        format_penalty=format_penalty,
                        final_response_text=final_response_text,
                    )
                    final_reward = float(reward_value)
                    agent_data.tool_rewards = [final_reward]
                    output.extra_fields["final_backbone_binary_score"] = final_binary_score
                    output.extra_fields["policy_format_penalty"] = format_penalty
                    output.extra_fields["final_policy_reward"] = final_reward
                    final_binary_details.update(format_details)
                    final_binary_details.update(reward_details)
                    final_binary_details["final_policy_reward"] = final_reward
                    final_binary_details["policy_reward_mode_name"] = self.policy_reward_mode
                    output.extra_fields["final_backbone_binary_details"] = final_binary_details
                    self._append_io_trace(
                        "policy.final_binary_judge",
                        {
                            "request_id": agent_data.request_id,
                            "trace_id": orchestrator_trace.get("trace_id", ""),
                            "orchestrator_round": orchestrator_trace.get("round", None),
                            "score": final_binary_score,
                            "reward_mode": self.policy_reward_mode,
                            "retrieval_effective": reward_details["retrieval_effective"],
                            "summary_reasonable": reward_details["summary_reasonable"],
                            "policy_format_penalty": format_penalty,
                            "final_policy_reward": final_reward,
                            "details": final_binary_details,
                            "final_response_text": final_response_text,
                        },
                    )
            # Export the latest dialog state so orchestrator can continue multi-round reasoning.
            output.extra_fields["request_id"] = agent_data.request_id
            output.extra_fields["raw_prompt"] = deepcopy(agent_data.messages)
            output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
            return output
        finally:
            await self._release_tool_instances(agent_data)

    async def _release_tool_instances(self, agent_data: AgentData) -> None:
        """Release per-trajectory tool instances safely at rollout end."""
        for tool_name, instance_id in list(agent_data.tool_instances.items()):
            tool = self.tools.get(tool_name)
            if tool is None:
                continue
            try:
                await tool.release(instance_id)
            except Exception as e:
                logger.warning(f"Failed to release tool instance {tool_name}:{instance_id}: {e}")
        agent_data.tool_instances.clear()

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        add_messages: list[dict[str, Any]] = []
        final_instruction_message = self._maybe_add_final_policy_turn_message(agent_data, add_messages)
        if final_instruction_message is not None:
            agent_data.messages.extend(add_messages)
            agent_data.user_turns += 1
            self._mark_final_policy_turn_instruction_appended(
                agent_data,
                final_instruction_message,
                token_count=None,
            )
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self._chat_template_tool_schemas(),
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []
        orchestrator_trace = self._extract_orchestrator_trace(agent_data.tools_kwargs)

        turn_sampling_params, assistant_budget, response_window_budget, append_budget = self._build_turn_sampling_params(
            sampling_params,
            agent_data,
        )
        if assistant_budget <= 0 or append_budget <= 0:
            agent_data.metrics["budget_exhausted"] = 1
            agent_data.metrics["assistant_budget_exhausted"] = 1 if assistant_budget <= 0 else 0
            agent_data.metrics["response_window_exhausted"] = 1 if response_window_budget <= 0 else 0
            agent_data.metrics["append_budget_exhausted"] = 1 if append_budget <= 0 else 0
            self._append_io_trace(
                "policy.budget_exhausted",
                {
                    "request_id": agent_data.request_id,
                    "trace_id": orchestrator_trace.get("trace_id", ""),
                    "orchestrator_round": orchestrator_trace.get("round", None),
                    "prompt_length": len(agent_data.prompt_ids),
                    "assistant_tokens_so_far": self._assistant_response_tokens(agent_data),
                    "response_tokens_so_far": self._total_response_tokens(agent_data),
                    "prompt_budget": self.prompt_length,
                    "assistant_response_budget": self.response_length,
                    "total_response_budget": self.total_response_length,
                    "remaining_assistant_budget": assistant_budget,
                    "remaining_response_window": response_window_budget,
                    "remaining_append_budget": append_budget,
                },
            )
            logger.warning(
                "Terminating rollout for request %s because no generation budget remains "
                "(prompt_len=%s, prompt_budget=%s, assistant_budget=%s, total_response_budget=%s)",
                agent_data.request_id,
                len(agent_data.prompt_ids),
                self.prompt_length,
                self.response_length,
                self.total_response_length,
            )
            return AgentState.TERMINATED

        if self.policy_use_api:
            with simple_timer("generate_sequences", agent_data.metrics):
                assistant_message, _raw_response = await self.loop.run_in_executor(
                    None,
                    lambda: self._call_policy_api(
                        agent_data.messages,
                        max_tokens=min(assistant_budget, append_budget),
                    ),
                )
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(assistant_message, add_special_tokens=False)
                )
                response_ids = response_ids[: min(assistant_budget, append_budget)]
                output = TokenOutput(
                    token_ids=response_ids,
                    log_probs=None,
                    num_preempted=0,
                    extra_fields={
                        "policy_api_model": self.policy_api_model,
                    },
                )
        else:
            with simple_timer("generate_sequences", agent_data.metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=turn_sampling_params,
                    image_data=agent_data.image_data,
                    video_data=agent_data.video_data,
                )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if output.extra_fields:
            for key, value in output.extra_fields.items():
                if key == "max_global_steps":
                    if value:
                        agent_data.extra_fields["max_global_steps"] = value
                    continue
                if key in {"finish_reason", "stop_reason", "global_step", "global_steps"}:
                    agent_data.extra_fields[f"last_{key}"] = value
                if key not in agent_data.extra_fields:
                    agent_data.extra_fields[key] = value

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and (
            self._assistant_response_tokens(agent_data) >= self.response_length
            or self._remaining_append_budget(agent_data) <= 0
        ):
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            agent_data.response_ids,
            tools,
            parse_context={
                "request_id": agent_data.request_id,
                "assistant_turn": agent_data.assistant_turns,
                "trace_id": orchestrator_trace.get("trace_id", ""),
                "orchestrator_round": orchestrator_trace.get("round", None),
            },
        )
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )
        agent_data.extra_fields["last_assistant_response_text"] = assistant_message or ""
        if self.policy_use_api:
            agent_data.messages.append({"role": "assistant", "content": assistant_message or ""})
        if not agent_data.tool_calls and (
            "<tool_call>" in (assistant_message or "") or "<search>" in (assistant_message or "")
        ):
            parse_error = None
            if hasattr(self.tool_parser, "pop_last_decode_error"):
                parse_error = self.tool_parser.pop_last_decode_error()
            self._append_io_trace(
                "policy.tool_call_parse_empty",
                {
                    "request_id": agent_data.request_id,
                    "trace_id": orchestrator_trace.get("trace_id", ""),
                    "orchestrator_round": orchestrator_trace.get("round", None),
                    "assistant_turn": agent_data.assistant_turns,
                    "parser": self.tool_parser_name,
                    "response_text": assistant_message or "",
                    "decode_error": parse_error,
                },
            )
            if parse_error is not None:
                has_remaining_assistant_turn = (
                    self.max_assistant_turns is None or agent_data.assistant_turns < self.max_assistant_turns
                )
                if has_remaining_assistant_turn:
                    if self.tool_parser_name == "search_xml":
                        feedback_content = (
                            "Your previous search request could not be decoded. "
                            f"Decode error: {parse_error}. "
                            "Output exactly one valid search request in this format: <search>query</search>. "
                            "Do not output plain text in the same turn."
                        )
                    else:
                        feedback_content = (
                            "Your previous tool call could not be decoded. "
                            f"Decode error: {parse_error}. "
                            "Output exactly one valid tool call using the configured format. "
                            "Do not output plain text."
                        )
                    decode_error_message = {
                        "role": "user",
                        "content": feedback_content,
                    }
                    appended = await self._append_text_observation(
                        agent_data,
                        decode_error_message,
                        metric_key="decode_error_observation_truncated",
                        truncated_flag_key="decode_error_observation_truncated",
                    )
                    if appended:
                        self._append_policy_judge_event(
                            agent_data,
                            {
                                "stage": "tool_call_decode_error_feedback",
                                "assistant_turn": agent_data.assistant_turns,
                                "decode_error": parse_error,
                                "feedback_message": decode_error_message["content"],
                            },
                        )
                        self._append_io_trace(
                            "policy.tool_call_decode_error_feedback",
                            {
                                "request_id": agent_data.request_id,
                                "trace_id": orchestrator_trace.get("trace_id", ""),
                                "orchestrator_round": orchestrator_trace.get("round", None),
                                "assistant_turn": agent_data.assistant_turns,
                                "decode_error": parse_error,
                                "feedback_message": decode_error_message["content"],
                            },
                        )
                        return AgentState.GENERATING
                return AgentState.TERMINATED

        self._append_policy_judge_event(
            agent_data,
            {
                "stage": "assistant_output",
                "assistant_turn": agent_data.assistant_turns,
                "response_text": assistant_message or "",
                "tool_calls": [
                    {
                        "name": tc.name,
                        "arguments": self._parse_tool_arguments_for_trace(tc.arguments),
                    }
                    for tc in (agent_data.tool_calls[: self.max_parallel_calls] if agent_data.tool_calls else [])
                ],
            },
        )

        self._append_io_trace(
            "policy.model_output",
            {
                "request_id": agent_data.request_id,
                "trace_id": orchestrator_trace.get("trace_id", ""),
                "orchestrator_round": orchestrator_trace.get("round", None),
                "assistant_turn": agent_data.assistant_turns,
                "response_text": assistant_message or "",
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in (agent_data.tool_calls[: self.max_parallel_calls] if agent_data.tool_calls else [])
                ],
            },
        )

        # Handle interaction if needed
        if self.interaction_config_file:
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        pending_tool_followup = bool(agent_data.extra_fields.get("awaiting_tool_followup", False))
        if pending_tool_followup:
            has_complete_answer_evidence = self._has_complete_answer_evidence_output(assistant_message or "")
            if agent_data.tool_calls:
                agent_data.extra_fields["awaiting_tool_followup"] = False
            elif has_complete_answer_evidence:
                agent_data.extra_fields["awaiting_tool_followup"] = False
            else:
                agent_data.extra_fields["awaiting_tool_followup"] = False
                orchestrator_trace = self._extract_orchestrator_trace(agent_data.tools_kwargs)
                self._append_io_trace(
                    "policy.invalid_post_tool_output_terminated",
                    {
                        "request_id": agent_data.request_id,
                        "trace_id": orchestrator_trace.get("trace_id", ""),
                        "orchestrator_round": orchestrator_trace.get("round", None),
                        "assistant_turn": agent_data.assistant_turns,
                        "response_text": assistant_message or "",
                    },
                )
                return AgentState.TERMINATED

        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
        orchestrator_trace = self._extract_orchestrator_trace(agent_data.tools_kwargs)
        self._append_io_trace(
            "policy.tool_calls_completed",
            {
                "request_id": agent_data.request_id,
                "trace_id": orchestrator_trace.get("trace_id", ""),
                "orchestrator_round": orchestrator_trace.get("round", None),
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in (agent_data.tool_calls[: self.max_parallel_calls] if agent_data.tool_calls else [])
                ],
            },
        )

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_call, (tool_response, tool_reward, tool_meta) in zip(
            agent_data.tool_calls[: self.max_parallel_calls], responses, strict=False
        ):
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)
            raw_tool_response_text = tool_response.text or ""
            if isinstance(tool_meta, dict):
                raw_candidate = tool_meta.get("raw_tool_response_text")
                if isinstance(raw_candidate, str) and raw_candidate.strip():
                    raw_tool_response_text = raw_candidate
            self._append_policy_judge_event(
                agent_data,
                {
                    "stage": "tool_result",
                    "tool_name": tool_call.name,
                    "tool_args": self._parse_tool_arguments_for_trace(tool_call.arguments),
                    "raw_tool_response_text": raw_tool_response_text,
                    "policy_visible_tool_response_text": tool_response.text or "",
                },
            )

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)
            if tool_meta:
                tool_trace = agent_data.extra_fields.setdefault("tool_trace", [])
                if isinstance(tool_trace, list):
                    tool_trace.append(tool_meta)

        final_instruction_message = self._maybe_add_final_policy_turn_message(agent_data, add_messages)

        if self.tool_parser_name == "gpt-oss" and final_instruction_message is None:
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        original_response_len = len(response_ids)
        tool_response_len = sum(
            self._message_content_len(message) for message in add_messages if message.get("role") == "tool"
        )
        append_budget = self._remaining_append_budget(agent_data)
        tool_call_count = len(agent_data.tool_calls[: self.max_parallel_calls] if agent_data.tool_calls else [])
        self._append_policy_debug_trace(
            "policy.tool_observation_append_before",
            self._build_policy_budget_debug_payload(
                agent_data,
                orchestrator_trace,
                assistant_turn_id=agent_data.assistant_turns,
                tool_call_count=tool_call_count,
                original_response_len=original_response_len,
                append_budget=append_budget,
                tool_response_len=tool_response_len,
                extra={
                    "final_turn_instruction": final_instruction_message["content"]
                    if final_instruction_message is not None
                    else None,
                    "tool_message_count": len(add_messages),
                },
            ),
        )
        if final_instruction_message is not None and append_budget < original_response_len:
            agent_data.extra_fields["final_policy_turn_instruction_append_failed"] = True
            agent_data.metrics["final_policy_turn_instruction_append_failed"] = 1
            self._append_policy_debug_trace(
                "policy.final_turn_instruction_append_failed",
                self._build_policy_budget_debug_payload(
                    agent_data,
                    orchestrator_trace,
                    assistant_turn_id=agent_data.assistant_turns + 1,
                    tool_call_count=tool_call_count,
                    original_response_len=original_response_len,
                    append_budget=append_budget,
                    tool_response_len=tool_response_len,
                    extra={
                        "reason": "insufficient_append_budget",
                        "message": final_instruction_message["content"],
                        "tool_message_count": len(add_messages),
                    },
                ),
            )
            return AgentState.TERMINATED
        response_ids, truncated = self._truncate_observation_tokens_to_budget(
            agent_data,
            response_ids,
            metric_key="tool_observation_truncated",
        )
        if not response_ids:
            self._append_policy_debug_trace(
                "policy.tool_observation_append_after",
                self._build_policy_budget_debug_payload(
                    agent_data,
                    orchestrator_trace,
                    assistant_turn_id=agent_data.assistant_turns,
                    tool_call_count=tool_call_count,
                    original_response_len=original_response_len,
                    append_budget=self._remaining_append_budget(agent_data),
                    tool_response_len=tool_response_len,
                    extra={
                        "final_turn_instruction": final_instruction_message["content"]
                        if final_instruction_message is not None
                        else None,
                        "kept_response_len": 0,
                        "truncated": truncated,
                        "terminated": True,
                        "tool_message_count": len(add_messages),
                    },
                ),
            )
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.messages.extend(add_messages)
        self._append_zero_mask_tokens(agent_data, response_ids)
        agent_data.user_turns += 1
        if final_instruction_message is not None:
            agent_data.user_turns += 1
            self._mark_final_policy_turn_instruction_appended(
                agent_data,
                final_instruction_message,
                token_count=None,
            )
        agent_data.extra_fields["awaiting_tool_followup"] = True
        self._append_policy_debug_trace(
            "policy.tool_observation_append_after",
            self._build_policy_budget_debug_payload(
                agent_data,
                orchestrator_trace,
                assistant_turn_id=agent_data.assistant_turns,
                tool_call_count=tool_call_count,
                original_response_len=original_response_len,
                append_budget=self._remaining_append_budget(agent_data),
                tool_response_len=tool_response_len,
                extra={
                    "final_turn_instruction": final_instruction_message["content"]
                    if final_instruction_message is not None
                    else None,
                    "kept_response_len": len(response_ids),
                    "truncated": truncated,
                    "terminated": False,
                    "tool_message_count": len(add_messages),
                },
            ),
        )
        if truncated:
            agent_data.extra_fields["tool_observation_truncated"] = True
            self._append_io_trace(
                "policy.tool_observation_truncated",
                {
                    "request_id": agent_data.request_id,
                    "trace_id": orchestrator_trace.get("trace_id", ""),
                    "orchestrator_round": orchestrator_trace.get("round", None),
                    "original_tokens": original_response_len,
                    "kept_tokens": len(response_ids),
                    "max_total_response_length": self.total_response_length,
                },
            )
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        orchestrator_trace = self._extract_orchestrator_trace(agent_data.tools_kwargs)
        final_instruction_message = self._maybe_add_final_policy_turn_message(agent_data, add_messages)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )
        original_response_len = len(response_ids)
        append_budget = self._remaining_append_budget(agent_data)
        if final_instruction_message is not None and append_budget < original_response_len:
            agent_data.extra_fields["final_policy_turn_instruction_append_failed"] = True
            agent_data.metrics["final_policy_turn_instruction_append_failed"] = 1
            self._append_policy_debug_trace(
                "policy.final_turn_instruction_append_failed",
                self._build_policy_budget_debug_payload(
                    agent_data,
                    orchestrator_trace,
                    assistant_turn_id=agent_data.assistant_turns + 1,
                    tool_call_count=len(agent_data.tool_calls or []),
                    original_response_len=original_response_len,
                    append_budget=append_budget,
                    tool_response_len=None,
                    extra={
                        "reason": "insufficient_append_budget",
                        "message": final_instruction_message["content"],
                    },
                ),
            )
            return AgentState.TERMINATED
        response_ids, truncated = self._truncate_observation_tokens_to_budget(
            agent_data,
            response_ids,
            metric_key="interaction_observation_truncated",
        )
        if not response_ids:
            return AgentState.TERMINATED

        # Update prompt_ids and response_mask
        agent_data.messages.extend(add_messages)
        self._append_zero_mask_tokens(agent_data, response_ids)
        agent_data.user_turns += 1
        if final_instruction_message is not None:
            agent_data.user_turns += 1
            self._mark_final_policy_turn_instruction_appended(
                agent_data,
                final_instruction_message,
                token_count=None,
            )
        if truncated:
            agent_data.extra_fields["interaction_observation_truncated"] = True

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, Optional[float], dict]:
        """Call tool and return tool response."""
        tool_name = tool_call.name
        tool_args: dict[str, Any] = {}
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_args = json.loads(tool_call.arguments)
            if tool_name not in self.tools:
                raise KeyError(f"Tool {tool_name} is not registered")
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id = agent_data.tool_instances.get(tool_name)
            if instance_id is None:
                create_kwargs = kwargs.get("create_kwargs", {}) or {}
                instance_id, _ = await tool.create(**create_kwargs)
                agent_data.tool_instances[tool_name] = instance_id
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
            if self.tool_reward_source == "backbone_binary":
                # In backbone_binary mode we judge once on the final integrated policy answer,
                # rather than per tool call.
                tool_reward = None
            else:
                model_score = self._extract_model_score(tool_args, tool_execution_response.text, res)
                if model_score is not None:
                    # Use model-side relevance score as step reward signal for GRPO.
                    tool_reward = model_score if tool_reward is None else float(tool_reward) + model_score
                    if isinstance(res, dict):
                        res["model_score"] = model_score
            # Optional extra step-level reward judged by external judge model.
            if self.tool_reward_source != "backbone_binary" and self.step_reward_judge_url:
                payload = {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_response": tool_execution_response.text,
                    "messages": agent_data.messages,
                }
                try:
                    judge_resp = await self.loop.run_in_executor(
                        None,
                        lambda: requests.post(
                            self.step_reward_judge_url, json=payload, timeout=self.step_reward_judge_timeout
                        ),
                    )
                    judge_resp.raise_for_status()
                    judge_data = judge_resp.json()
                    judged_reward = float(judge_data.get("score", 0.0))
                    tool_reward = judged_reward if tool_reward is None else float(tool_reward) + judged_reward
                    if isinstance(res, dict):
                        res["step_judge_score"] = judged_reward
                except Exception as e:
                    logger.warning(f"Step reward judge failed: {e}")
        except Exception as e:
            orchestrator_trace = self._extract_orchestrator_trace(agent_data.tools_kwargs)
            self._append_io_trace(
                "policy.tool_call_error",
                {
                    "request_id": agent_data.request_id,
                    "trace_id": orchestrator_trace.get("trace_id", ""),
                    "orchestrator_round": orchestrator_trace.get("round", None),
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "error": str(e),
                },
            )
            logger.warning(f"Error when executing tool: {e}")
            error_tool_reward = None if self.tool_reward_source == "backbone_binary" else 0.0
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                error_tool_reward,
                {},
            )

        tool_response_text = self._prepare_tool_response_for_policy(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_response_text=tool_execution_response.text,
        )
        structured_tool_response_text = None
        if tool_response_text:
            structured_tool_response_text = self._truncate_structured_tool_response(tool_name, tool_response_text)
        if structured_tool_response_text is not None:
            tool_response_text = structured_tool_response_text
        elif tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        orchestrator_trace = self._extract_orchestrator_trace(agent_data.tools_kwargs)
        if isinstance(res, dict):
            res["raw_tool_response_text"] = tool_execution_response.text
            res["policy_visible_tool_response_text"] = tool_response_text
        trace_payload = {
            "request_id": agent_data.request_id,
            "trace_id": orchestrator_trace.get("trace_id", ""),
            "orchestrator_round": orchestrator_trace.get("round", None),
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_meta": res,
            "tool_response": tool_response_text,
        }
        if self.tool_reward_source != "backbone_binary":
            trace_payload["tool_reward"] = tool_reward
        self._append_io_trace("policy.tool_call_result", trace_payload)

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
