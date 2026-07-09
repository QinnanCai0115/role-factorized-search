# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
from datetime import datetime, timezone
import json
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        custom_cfg = self.rollout_config.get("custom", {}) or {}
        self.io_trace_log_path = str(custom_cfg.get("io_trace_log_path", os.getenv("VERL_IO_TRACE_LOG_PATH", "")) or "")
        self.io_trace_max_chars = int(custom_cfg.get("io_trace_max_chars", 4000))
        self.io_trace_max_items = int(custom_cfg.get("io_trace_max_items", 6))

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

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        request_id = uuid4().hex
        self._append_io_trace(
            "backbone.run_start",
            {
                "request_id": request_id,
                "raw_prompt": messages,
                "sampling_params": sampling_params,
            },
        )

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
        )
        self._append_io_trace(
            "backbone.model_output",
            {
                "request_id": request_id,
                "response_text": response_text,
                "num_turns": output.num_turns,
            },
        )

        # Keep request_id in extra_fields so downstream non_tensor_batch and io traces can join reliably.
        output.extra_fields.update({"request_id": request_id, "turn_scores": [], "tool_rewards": []})

        return output
