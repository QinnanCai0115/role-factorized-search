# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import logging
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat
from typing import Any

import numpy as np
import ray
import torch

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger

logger = logging.getLogger(__name__)


class _NoopCheckpointManager:
    def sleep_replicas(self):
        return None


class _BackboneAPIRolloutAdapter:
    """Driver-side backbone rollout adapter for fully async two-stage orchestration."""

    world_size = 1

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def _api_config(self) -> tuple[str, float, str, str, str, int, int, float]:
        custom_cfg = self.config.actor_rollout_ref.rollout.get("custom", {}) or {}
        api_url = str(custom_cfg.get("backbone_api_url", ""))
        timeout = float(custom_cfg.get("backbone_api_timeout", 30.0))
        api_mode = str(custom_cfg.get("backbone_api_mode", "custom")).lower()
        api_key = str(custom_cfg.get("backbone_api_key", ""))
        if not api_key:
            api_key = os.environ.get("BACKBONE_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
        api_model = str(custom_cfg.get("backbone_api_model", "deepseek-reasoner"))
        api_max_concurrent = max(1, int(custom_cfg.get("backbone_api_max_concurrent", 1)))
        api_max_retries = max(0, int(custom_cfg.get("backbone_api_max_retries", 3)))
        api_retry_backoff = max(0.0, float(custom_cfg.get("backbone_api_retry_backoff", 1.0)))
        return api_url, timeout, api_mode, api_key, api_model, api_max_concurrent, api_max_retries, api_retry_backoff

    @staticmethod
    def _raw_prompt_to_text(raw_prompt: Any) -> str:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if isinstance(raw_prompt, (list, tuple)):
            chunks = []
            for msg in raw_prompt:
                if isinstance(msg, dict):
                    role = str(msg.get("role", "")).strip()
                    content = str(msg.get("content", ""))
                    chunks.append(f"{role}\n{content}" if role else content)
                else:
                    chunks.append(str(msg))
            return "\n\n".join(chunks)
        if isinstance(raw_prompt, dict):
            return str(raw_prompt.get("content", ""))
        return str(raw_prompt)

    def _build_dataproto_from_api_texts(
        self,
        prompts: DataProto,
        response_texts: list[str],
        has_tool_calls: list[bool],
        prompt_texts: list[str] | None = None,
        api_usages: list[dict[str, Any] | None] | None = None,
    ) -> DataProto:
        meta_info = prompts.meta_info or {}
        pad_token_id = meta_info.get("pad_token_id", self.tokenizer.pad_token_id)
        batch_tensors = getattr(prompts, "batch", None)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        max_prompt_len = int(rollout_cfg.get("prompt_length", self.config.data.get("max_prompt_length", 512)))

        def _normalize_prompt_tensors(
            prompt_ids: torch.Tensor, prompt_attn: torch.Tensor, target_len: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if prompt_ids.shape[1] > target_len:
                prompt_ids = prompt_ids[:, -target_len:]
                prompt_attn = prompt_attn[:, -target_len:]
            elif prompt_ids.shape[1] < target_len:
                pad_cols = target_len - prompt_ids.shape[1]
                prompt_ids_pad = torch.full(
                    (prompt_ids.shape[0], pad_cols), fill_value=pad_token_id, dtype=prompt_ids.dtype
                )
                prompt_attn_pad = torch.zeros((prompt_attn.shape[0], pad_cols), dtype=prompt_attn.dtype)
                prompt_ids = torch.cat([prompt_ids_pad, prompt_ids], dim=1)
                prompt_attn = torch.cat([prompt_attn_pad, prompt_attn], dim=1)
            return prompt_ids.cpu(), prompt_attn.cpu()

        prompt_ids = None
        prompt_attn = None
        if batch_tensors is not None:
            prompt_ids = batch_tensors.get("prompts")
            if prompt_ids is None:
                prompt_ids = batch_tensors.get("input_ids")
            prompt_attn = batch_tensors.get("attention_mask")

        if prompt_ids is None:
            if prompt_texts is None:
                raise ValueError("API backbone rollout requires raw_prompt or prompt tensors.")
            tokenized = [self.tokenizer.encode(t, add_special_tokens=False) for t in prompt_texts]
            prompt_ids = torch.full((len(tokenized), max_prompt_len), fill_value=pad_token_id, dtype=torch.long)
            prompt_attn = torch.zeros((len(tokenized), max_prompt_len), dtype=torch.long)
            for i, tok in enumerate(tokenized):
                if not tok:
                    continue
                tok = tok[-max_prompt_len:]
                t = torch.tensor(tok, dtype=torch.long)
                prompt_ids[i, -len(tok) :] = t
                prompt_attn[i, -len(tok) :] = 1
        else:
            prompt_ids = prompt_ids.cpu()
            if prompt_attn is None:
                prompt_attn = torch.ones_like(prompt_ids, dtype=torch.long)
            else:
                prompt_attn = prompt_attn.cpu()
                if prompt_attn.shape[1] > prompt_ids.shape[1]:
                    prompt_attn = prompt_attn[:, -prompt_ids.shape[1] :]
            prompt_ids, prompt_attn = _normalize_prompt_tensors(prompt_ids, prompt_attn, max_prompt_len)

        batch_size = prompt_ids.shape[0]
        max_resp_len = int(rollout_cfg.get("response_length", self.config.data.get("max_response_length", 512)))
        responses = torch.full((batch_size, max_resp_len), fill_value=pad_token_id, dtype=prompt_ids.dtype)
        response_mask = torch.zeros((batch_size, max_resp_len), dtype=torch.long)
        for i, text in enumerate(response_texts):
            token_ids = self.tokenizer.encode(str(text or ""), add_special_tokens=False)[:max_resp_len]
            if not token_ids:
                continue
            tok = torch.tensor(token_ids, dtype=prompt_ids.dtype)
            responses[i, : len(token_ids)] = tok
            response_mask[i, : len(token_ids)] = 1

        input_ids = torch.cat([prompt_ids, responses], dim=1)
        attention_mask = torch.cat([prompt_attn, response_mask], dim=1)
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids = torch.clamp(position_ids, min=0)

        non_tensors = {}
        for key, value in prompts.non_tensor_batch.items():
            non_tensors[key] = value
        non_tensors["has_tool_call"] = np.array(has_tool_calls, dtype=object)
        if api_usages is not None:
            non_tensors["backbone_api_usage"] = np.array(api_usages, dtype=object)

        return DataProto.from_dict(
            tensors={
                "prompts": prompt_ids,
                "responses": responses,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "rm_scores": torch.zeros((batch_size, max_resp_len), dtype=torch.float32),
            },
            non_tensors=non_tensors,
            meta_info=dict(meta_info),
        )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        (
            api_url,
            timeout,
            api_mode,
            api_key,
            api_model,
            api_max_concurrent,
            api_max_retries,
            api_retry_backoff,
        ) = self._api_config()
        if not api_url:
            raise ValueError("backbone_api_url must be set when fully async backbone API rollout is enabled.")

        import requests

        raw_prompts = prompts.non_tensor_batch.get("raw_prompt", None)
        if raw_prompts is None or len(raw_prompts) == 0:
            raise ValueError("API backbone rollout cannot find raw_prompt.")
        prompt_text_rows = [self._raw_prompt_to_text(raw_prompts[i]) for i in range(len(raw_prompts))]
        response_texts: list[str] = [""] * len(prompt_text_rows)
        has_tool_calls: list[bool] = [False] * len(prompt_text_rows)
        api_usages: list[dict[str, Any] | None] = [None] * len(prompt_text_rows)

        disable_backbone_proxy = str(os.environ.get("BACKBONE_API_NO_PROXY", "")).strip().lower() not in (
            "",
            "0",
            "false",
            "no",
        )

        def _build_request_session():
            request_session = requests.Session()
            if disable_backbone_proxy:
                request_session.trust_env = False
            return request_session

        def _is_retryable_backbone_error(exc: Exception) -> bool:
            if isinstance(exc, requests.HTTPError):
                response = getattr(exc, "response", None)
                status_code = response.status_code if response is not None else None
                return status_code == 429 or (status_code is not None and 500 <= status_code < 600)
            return isinstance(exc, (requests.ConnectionError, requests.Timeout))

        def _backbone_retry_delay(attempt: int, exc: Exception) -> float:
            response = getattr(exc, "response", None)
            retry_after = response.headers.get("Retry-After") if response is not None else None
            if retry_after:
                try:
                    return max(0.0, min(float(retry_after), 60.0))
                except ValueError:
                    pass
            return min(api_retry_backoff * (2**attempt), 8.0)

        def _request_one(i: int, prompt_text: str) -> dict[str, Any]:
            request_session = _build_request_session()
            try:
                for attempt in range(api_max_retries + 1):
                    try:
                        if api_mode == "openai_compatible":
                            endpoint = f"{api_url.rstrip('/')}/chat/completions"
                            headers = {"Content-Type": "application/json"}
                            if api_key:
                                headers["Authorization"] = f"Bearer {api_key}"
                            payload = {
                                "model": api_model,
                                "messages": [{"role": "user", "content": prompt_text}],
                                "temperature": 0.0,
                            }
                            resp = request_session.post(endpoint, json=payload, headers=headers, timeout=timeout)
                            resp.raise_for_status()
                            data = resp.json()
                            api_usage = data.get("usage", None) if isinstance(data, dict) else None
                            choices = data.get("choices", []) if isinstance(data, dict) else []
                            message = choices[0].get("message", {}) if choices else {}
                            response_text = message.get("content", "") if isinstance(message, dict) else ""
                            has_tool_call = bool((message.get("tool_calls") if isinstance(message, dict) else None))
                        else:
                            resp = request_session.post(
                                api_url,
                                json={"prompt": prompt_text, "index": i},
                                timeout=timeout,
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            api_usage = data.get("usage", None) if isinstance(data, dict) else None
                            response_text = data.get("text", data.get("response", ""))
                            has_tool_call = bool(data.get("has_tool_call", False))
                        break
                    except Exception as exc:
                        if attempt >= api_max_retries or not _is_retryable_backbone_error(exc):
                            raise
                        delay = _backbone_retry_delay(attempt, exc)
                        logger.warning(
                            "Backbone API request failed for row %s; retrying %s/%s in %.1fs: %s",
                            i,
                            attempt + 1,
                            api_max_retries,
                            delay,
                            exc,
                        )
                        time.sleep(delay)
                else:
                    raise RuntimeError("Backbone API retry loop exhausted without an exception")
            finally:
                request_session.close()

            if not has_tool_call:
                resp_text = str(response_text).lower()
                has_tool_call = (
                    "<tool_call>" in resp_text
                    or "<tool_calls>" in resp_text
                    or '"tool_calls"' in resp_text
                    or "<search>" in resp_text
                )
            return {
                "index": i,
                "response_text": response_text,
                "has_tool_call": has_tool_call,
                "api_usage": api_usage if isinstance(api_usage, dict) else None,
            }

        start = time.time()
        max_workers = min(api_max_concurrent, len(prompt_text_rows))
        if max_workers <= 1:
            for i, prompt_text in enumerate(prompt_text_rows):
                row = _request_one(i, prompt_text)
                response_texts[i] = row["response_text"]
                has_tool_calls[i] = row["has_tool_call"]
                api_usages[i] = row.get("api_usage")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_request_one, i, prompt_text) for i, prompt_text in enumerate(prompt_text_rows)
                ]
                for future in futures:
                    row = future.result()
                    i = row["index"]
                    response_texts[i] = row["response_text"]
                    has_tool_calls[i] = row["has_tool_call"]
                    api_usages[i] = row.get("api_usage")

        output = self._build_dataproto_from_api_texts(
            prompts,
            response_texts,
            has_tool_calls,
            prompt_texts=prompt_text_rows,
            api_usages=api_usages,
        )
        output.meta_info["timing"] = {"backbone_api/generate": time.time() - start}
        return output


@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(SeparateRayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger or equal than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = False

        self.use_rm = False

        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # ==================== fully async config ====================

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        if self.config.async_training.use_trainer_do_validate:
            rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
            train_gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node
            total_gpus = rollout_gpus + train_gpus
            print(f"[FullyAsyncRollouter] split before val_dataset total len: {len(val_dataset)}")
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[:rollout_gpus]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            print(f"[FullyAsyncRollouter] split after val_dataset total len: {len(val_dataset)}")
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None
        self.backbone_rollout_wg = None
        self.backbone_async_rollout_manager = None
        self.checkpoint_manager = _NoopCheckpointManager()
        self._latest_rollout_trajectory_batch = None
        self._ensure_io_trace_defaults()
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        self.backbone_api_continue_on_failure = self._config_bool(
            rollout_custom.get("backbone_api_continue_on_failure", True), default=True
        )

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.total_generated_samples = 0
        self.total_generated_train_points = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.failed_samples = 0
        self.backbone_api_failed_samples = 0
        self.processed_sample_count = 0
        # we start from step 1
        self.global_steps = 1
        self.idle_start_time = time.time()
        self.step_start_time = time.time()

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Initialize async queues
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()

        cpu_cores = multiprocessing.cpu_count()
        # cpu case use cpu_cores; io case use cpu_cores*2
        self.validate_executor = ThreadPoolExecutor(max_workers=cpu_cores)
        self.validate_task = None

    @staticmethod
    def _config_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _is_recoverable_backbone_rollout_error(exc: Exception) -> bool:
        seen = set()
        current: BaseException | None = exc
        recoverable_names = {
            "ConnectionError",
            "ConnectTimeout",
            "JSONDecodeError",
            "MaxRetryError",
            "NameResolutionError",
            "ReadTimeout",
            "Timeout",
            "gaierror",
        }
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            exc_name = current.__class__.__name__
            if exc_name in recoverable_names:
                return True
            if exc_name == "HTTPError":
                response = getattr(current, "response", None)
                status_code = getattr(response, "status_code", None)
                if status_code == 429 or (status_code is not None and 500 <= status_code < 600):
                    return True
                return False
            current = current.__cause__ if current.__cause__ is not None else current.__context__
        return False

    def _estimated_train_points_per_rollout_sample(self) -> int:
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        try:
            policy_rollout_n = int(rollout_custom.get("policy_rollout_n", 1))
        except (TypeError, ValueError):
            policy_rollout_n = 1
        return max(1, policy_rollout_n)

    @staticmethod
    def _count_train_points(batch: DataProto | None) -> int:
        return len(batch) if batch is not None else 0

    def _queue_group_key(self, batch: DataProto) -> str | None:
        algorithm_cfg = self.config.get("algorithm", {}) or {}
        group_key = algorithm_cfg.get("grpo_group_key", None)
        candidates = [group_key, "pair_group_id", "uid"]
        for key in candidates:
            if key and key in batch.non_tensor_batch:
                return str(key)
        return None

    @staticmethod
    def _slice_batch_for_queue(batch: DataProto, indices: list[int]) -> DataProto:
        chunk = batch[indices]
        chunk.meta_info = dict(chunk.meta_info or {})
        metrics = chunk.meta_info.get("metrics")
        if isinstance(metrics, list) and len(metrics) == len(batch):
            chunk.meta_info["metrics"] = [metrics[i] for i in indices]
        return chunk

    def _split_rollout_sample_for_queue(self, rollout_sample: RolloutSample) -> list[RolloutSample]:
        batch = rollout_sample.full_batch
        if batch is None or len(batch) == 0:
            return []

        group_key = self._queue_group_key(batch)
        if group_key is None:
            return [rollout_sample]

        group_values = batch.non_tensor_batch.get(group_key)
        if group_values is None or len(group_values) != len(batch):
            return [rollout_sample]

        ordered_groups: dict[str, list[int]] = {}
        for idx, value in enumerate(group_values):
            group_id = str(value if value is not None else "")
            if not group_id:
                group_id = f"__ungrouped_{idx}"
            ordered_groups.setdefault(group_id, []).append(idx)

        if len(ordered_groups) <= 1:
            return [rollout_sample]

        chunks: list[RolloutSample] = []
        for group_idx, indices in enumerate(ordered_groups.values()):
            chunk_batch = self._slice_batch_for_queue(batch, indices)
            chunks.append(
                RolloutSample(
                    full_batch=chunk_batch,
                    sample_id=f"{rollout_sample.sample_id}::group{group_idx}",
                    epoch=rollout_sample.epoch,
                    rollout_status=rollout_sample.rollout_status,
                    trajectory_batch=None,
                )
            )
        return chunks

    def _record_failed_rollout_sample(self, rollout_sample: RolloutSample, exc: Exception, source: str):
        self.failed_samples += 1
        if source == "backbone_api":
            self.backbone_api_failed_samples += 1
        self.processed_sample_count += 1
        self.staleness_samples = max(0, self.staleness_samples - self._estimated_train_points_per_rollout_sample())
        sample_id = getattr(rollout_sample, "sample_id", "<unknown>")
        logger.exception(
            "[FullyAsyncRollouter][SampleFailure] Dropping rollout sample %s after %s failure; continuing.",
            sample_id,
            source,
        )
        print(
            "[FullyAsyncRollouter][SampleFailure] "
            f"Dropped rollout sample {sample_id} after {source} failure; continuing. "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )

    def _init_async_objects(self):
        # Initialize asyncio synchronization primitives.
        # We let asyncio.Condition create the Lock internally to ensure they share the same Event Loop.
        # This avoids 'ValueError: loop argument must agree with lock' which can occur in Ray environments
        # where the lock's captured loop (get_running_loop) differs from Condition's default loop check.
        # Explicitly passing the loop is deprecated/removed in Python 3.10+, so this reverse-initialization
        # is the most robust workaround.
        self.condition = asyncio.Condition()
        self.lock = self.condition._lock

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_required_samples(self):
        async with self.lock:
            self.max_required_samples = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            estimated_train_points = self._estimated_train_points_per_rollout_sample()
            max_concurrent_rollout_samples = max(
                1,
                (self.max_required_samples + estimated_train_points - 1) // estimated_train_points,
            )
            self.max_concurrent_samples = len(self.async_rollout_manager.server_handles) * 16
            self.max_concurrent_samples = min(self.max_concurrent_samples, max_concurrent_rollout_samples)
            self.max_queue_size = self.max_required_samples

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size(train_points): {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
                f"estimated_train_points_per_rollout_sample: {estimated_train_points} "
            )

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_replicas(self):
        """Get rollout worker group"""
        return self.async_rollout_manager.rollout_replicas

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def reset_staleness(self):
        """
        Reset staleness samples after parameter update.
        Returns timing_raw dictionary for metrics.
        """
        async with self.lock:
            self.paused = False
            self.condition.notify_all()
            # every time param change, reset staleness_samples
            self.staleness_samples = self._estimated_active_task_train_points() + await self.message_queue_client.get_queue_size()
            timing_raw = {}
            rollout_active_time = self.idle_start_time - self.step_start_time
            rollout_version_time = time.time() - self.step_start_time
            idle_ratio = 1 - rollout_active_time / rollout_version_time
            timing_raw["fully_async/rollouter/active_time"] = rollout_active_time
            timing_raw["fully_async/rollouter/version_time"] = rollout_version_time
            timing_raw["fully_async/rollouter/idle_ratio"] = idle_ratio

            print(
                f"[FullyAsyncRollouter][Public][reset_staleness] "
                f"reset staleness_samples to: {self.staleness_samples} "
                f"idle_ratio: {timing_raw['fully_async/rollouter/idle_ratio']:.4f}"
            )
            self.step_start_time = time.time()
        return timing_raw

    def _estimated_active_task_train_points(self) -> int:
        unfinished_tasks = sum(1 for task in self.active_tasks if not task.done())
        return unfinished_tasks * self._estimated_train_points_per_rollout_sample()

    def _refresh_staleness_samples_from_runtime_state(self, queue_size: int) -> int:
        """Refresh stale outstanding samples from the live task and queue state.

        The trainer consumes and may drop entries from MessageQueue without notifying the
        rollouter. If we only mutate staleness_samples when producing samples or syncing
        parameters, the rollouter can stay paused forever after the queue has drained.
        """
        queue_size = max(0, int(queue_size))
        active_estimated = self._estimated_active_task_train_points()
        refreshed_staleness = active_estimated + queue_size
        if refreshed_staleness != self.staleness_samples:
            print(
                "[FullyAsyncRollouter][Staleness] "
                f"refresh staleness_samples {self.staleness_samples} -> {refreshed_staleness} "
                f"(active_estimated={active_estimated}, mq_train_points={queue_size})",
                flush=True,
            )
            self.staleness_samples = refreshed_staleness
        return self.staleness_samples

    def do_validate(self) -> ValidateMetrics:
        """Run validation and return metrics"""
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate()
        return ValidateMetrics(timing_raw=timing_raw, metrics=val_metrics)

    async def save_checkpoint(self, local_global_step_folder: str):
        # WARNING!: Due to the asynchronous nature, there are some in-flight samples
        # (pending/cancel/result queue and message queue).
        # Therefore, directly saving the state of the dataloader will result in losing these
        # samples when resuming training.
        # TODO: Implement dataloader recovery without losing in-flight samples.
        from verl.utils.fs import local_mkdir_safe

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        async with self.dataloader_lock:
            dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        print(f"[FullyAsyncRollouter] Saved dataloader checkpoint to {dataloader_local_path}")

    def load_checkpoint(self):
        """Load checkpoint including dataloader state based on resume mode"""

        if self.config.trainer.resume_mode == "disable":
            print("[FullyAsyncRollouter] Resume mode is disabled, starting from scratch")
            return 0

        # Determine checkpoint folder path
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("[FullyAsyncRollouter] Load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # Find and validate global_step_folder based on resume mode
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncRollouter] Training from scratch (no checkpoint found)")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), (
                "[FullyAsyncRollouter] resume_from_path must be str type"
            )
            assert "global_step_" in self.config.trainer.resume_from_path, (
                "[FullyAsyncRollouter] resume_from_path must specify the global_steps"
            )
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            raise ValueError(f"[FullyAsyncRollouter] Unknown resume_mode: {self.config.trainer.resume_mode}")

        print(f"[FullyAsyncRollouter] Loading checkpoint from: {global_step_folder}")

        # Extract and set global step
        trainer_global_steps = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = (
            trainer_global_steps * self.required_samples * self.config.async_training.trigger_parameter_sync_step + 1
        )
        print(f"[FullyAsyncRollouter] Setting global_steps to {self.global_steps}")

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"[FullyAsyncRollouter] Loaded dataloader state from {dataloader_local_path}")
        else:
            print(
                f"[FullyAsyncRollouter] Warning: No dataloader state found at {dataloader_local_path}, "
                f"will start from scratch"
            )

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_async_objects()
        self._create_worker_classes()
        self._init_reward_loop()
        await self._init_async_rollout_manager()

    def _create_actor_rollout_classes(self):
        # Skip rollout creation and let agentloop handle it
        pass

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config, worker_group=self.rollout_wg, reward_loop_worker_handles=reward_loop_worker_handles
        )

        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        if bool(rollout_custom.get("enable_backbone_rollout", False)):
            if bool(rollout_custom.get("backbone_use_api", False)):
                self.backbone_rollout_wg = _BackboneAPIRolloutAdapter(self.config, self.tokenizer)
                print("[FullyAsyncRollouter] Backbone API rollout adapter initialized")
            else:
                raise NotImplementedError(
                    "Fully async two-stage rollout currently supports API-backed backbone rollout only. "
                    "Set actor_rollout_ref.rollout.custom.backbone_use_api=true."
                )

    # Add samples to the pending_queue
    async def _feed_samples(self):
        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:
            # Similar to _prepare_generate_batch: Separate data
            full_batch = prepare_single_generation_data(batch_dict, self.config)

            sample_id = f"sample_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                sample_id=sample_id,
                epoch=epoch,
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples: "
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put(None)
        print(f"[FullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    async def _processor_worker(self):
        """
        Streaming worker coroutines, a sample is submitted for processing without waiting for batches
        """
        while True:
            if self.paused or await self._should_pause_generation():
                print(
                    "[FullyAsyncRollouter][Processor] Received pause signal, waiting for remaining tasks to return..."
                )
                async with self.lock:
                    self.paused = True
                while self.active_tasks:
                    async with self.lock:
                        # After acquiring the lock, the number of active_tasks may change, need to be verified again
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done_tasks:
                                await task

                async with self.lock:
                    while self.paused:
                        self.idle_start_time = time.time()
                        await self.condition.wait()
                continue
            # Get sample from appropriate queue and immediately mark task as done
            rollout_sample = await self.pending_queue.get()
            self.pending_queue.task_done()
            self.staleness_samples += self._estimated_train_points_per_rollout_sample()

            if rollout_sample is None:
                print(
                    "[FullyAsyncRollouter][Processor] Received end signal, waiting for remaining tasks to complete..."
                )
                while self.active_tasks:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done_tasks:
                                await task
                break

            # Check whether the number of concurrent tasks exceeds the limit
            while len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done_tasks:
                            await task

            # Submit single sample processing
            async with self.lock:
                # After the pause is over, the lock is acquired and it is necessary
                # to determine whether it is the pause phase, otherwise continue to wait
                while self.paused:
                    await self.condition.wait()
                task = safe_create_task(
                    self._process_single_sample_streaming(rollout_sample),
                    name=rollout_sample.sample_id,
                    task_set=self.active_tasks,
                )

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample streamingly"""
        # Calling asynchronous generation methods
        trajectory_batch = None
        self._attach_sample_identity(rollout_sample)
        uses_backbone_rollout = self._is_backbone_rollout_enabled()
        try:
            if uses_backbone_rollout:
                timing_raw = {}
                self._latest_rollout_trajectory_batch = None
                ret, trajectory_batch = await asyncio.to_thread(
                    self._run_two_stage_orchestrator_rollout,
                    rollout_sample.full_batch,
                    timing_raw,
                    False,
                    return_trajectory_batch=True,
                )
                if "metrics" not in ret.meta_info:
                    generate_time = sum(float(v) for v in timing_raw.values() if isinstance(v, (int, float)))
                    ret.meta_info["metrics"] = [
                        {"generate_sequences": generate_time, "tool_calls": 0.0, "num_preempted": 0}
                        for _ in range(len(ret))
                    ]
            else:
                ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
        except Exception as exc:
            if (
                uses_backbone_rollout
                and self.backbone_api_continue_on_failure
                and self._is_recoverable_backbone_rollout_error(exc)
            ):
                self._record_failed_rollout_sample(rollout_sample, exc, "backbone_api")
                return
            raise
        rollout_sample.full_batch = ret
        rollout_sample.trajectory_batch = trajectory_batch
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        if rollout_sample.trajectory_batch is not None:
            rollout_sample.trajectory_batch.non_tensor_batch["uid"] = np.array(
                [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.trajectory_batch), dtype=object
            )
        rollout_sample.rollout_status = await self.get_statistics()

        queue_samples = self._split_rollout_sample_for_queue(rollout_sample)
        queued_train_points = sum(self._count_train_points(sample.full_batch) for sample in queue_samples)
        self.staleness_samples = max(
            0,
            self.staleness_samples + queued_train_points - self._estimated_train_points_per_rollout_sample(),
        )

        queue_dropped_old_sample = False
        queued_entries = 0
        for queue_sample in queue_samples:
            train_points = self._count_train_points(queue_sample.full_batch)
            if train_points <= 0:
                continue
            success = await self.message_queue_client.put_sample(
                sample=ray.cloudpickle.dumps(queue_sample),
                item_count=train_points,
            )
            queue_dropped_old_sample = queue_dropped_old_sample or not success
            queued_entries += 1

        if queued_entries > 0:
            self.total_generated_samples += 1
            self.total_generated_train_points += queued_train_points
        if queue_dropped_old_sample:
            self.dropped_stale_samples += 1
        self.processed_sample_count += 1

    def _attach_sample_identity(self, rollout_sample: RolloutSample):
        sample_uid = f"uid_{rollout_sample.sample_id}"
        batch_size = len(rollout_sample.full_batch)
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array([sample_uid] * batch_size, dtype=object)
        rollout_sample.full_batch.non_tensor_batch["orchestrator_trace_id"] = np.array(
            [f"{rollout_sample.sample_id}::root{i}" for i in range(batch_size)], dtype=object
        )

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed coroutine, streaming process coroutine
        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        self.processor_task = safe_create_task(self._processor_worker(), name="processor_task")

        try:
            # Wait for sample feed to complete
            # Use asyncio.wait to monitor all tasks. If processor exits early,
            # detect it instead of blocking on feed_task (it might be stuck on a full queue).
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[FullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await self.processor_task
            print("[FullyAsyncRollouter] Streaming process completed")

            await self.pending_queue.join()
            print("[FullyAsyncRollouter] pending_queue joined")

        except Exception as e:
            print(f"[FullyAsyncRollouter] Streaming process exception: {e}")
            raise e

        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            self.feed_task = None
            self.processor_task = None

            # Send a finish signal
            await self.message_queue_client.put_sample(sample=None)

        async with self.lock:
            self.running = False

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True

        # Create the main asynchronous task
        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            # The monitor loop is intentionally long-lived. Wait for the generation
            # task itself so generation failures are propagated instead of being
            # hidden while the monitor keeps printing stale statistics.
            await generation_task
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
            raise
        finally:
            async with self.lock:
                self.running = False

            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # Trigger rollout recovery
            if self.paused and not await self._should_pause_generation():
                async with self.lock:
                    self.paused = False
                    print("[FullyAsyncRollouter][ShouldPause] notify all wait tasks.")
                    self.condition.notify_all()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = self.message_queue_client.get_statistics_sync()
        queue_size = queue_stats["queue_size"]
        staleness_samples = self._refresh_staleness_samples_from_runtime_state(queue_size)

        if queue_size >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: size={queue_size}, max={self.max_queue_size}"
                )
            return True

        if staleness_samples >= self.max_required_samples:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"staleness_samples {staleness_samples} >= max_required_samples {self.max_required_samples} "
                )
            return True

        return False

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            # counting stats
            "count/total_generated_samples": self.total_generated_samples,
            "count/total_generated_train_points": self.total_generated_train_points,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            "count/failed_samples": self.failed_samples,
            "count/backbone_api_failed_samples": self.backbone_api_failed_samples,
            # static stats
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        return stats
