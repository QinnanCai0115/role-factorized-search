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
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    rollout_status: dict[str, Any]
    trajectory_batch: Any | None = None


@dataclass
class ValidateMetrics:
    """Metrics for validation"""

    timing_raw: dict[str, Any]
    metrics: Optional[dict[str, Any]] = None


def prepare_single_generation_data(batch_dict, config) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    batch_keys_to_pop = []
    non_tensor_batch_keys_to_pop = []

    existing_batch_keys = [k for k in batch_keys_to_pop if k in full_batch.batch.keys()]
    existing_non_tensor_keys = [k for k in non_tensor_batch_keys_to_pop if k in full_batch.non_tensor_batch.keys()]

    if existing_batch_keys or existing_non_tensor_keys:
        full_batch.pop(
            batch_keys=existing_batch_keys,
            non_tensor_batch_keys=existing_non_tensor_keys,
        )

    # Setting selected agent, that supports partial
    if config.actor_rollout_ref.rollout.multi_turn.enable:
        full_batch.non_tensor_batch["agent_name"] = np.array(["tool_agent"] * len(full_batch), dtype=object)
    else:
        full_batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(full_batch), dtype=object)

    # Add global step count to generated data
    full_batch = full_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
    return full_batch


def addition_process(output: DataProto):
    """collect metirics"""
    metrics = output.meta_info.pop("metrics", None)  # List[Dict[str, str]]
    if metrics:
        processing_times_list = [item.get("generate_sequences", 0.0) for item in metrics]
        tool_calls_times_list = [item.get("tool_calls", 0.0) for item in metrics]
    else:
        timing = output.meta_info.pop("timing", {}) or {}
        generate_time = timing.get("agent_loop/generate_sequences/mean", timing.get("generate_sequences", 0.0))
        tool_calls_time = timing.get("agent_loop/tool_calls/mean", timing.get("tool_calls", 0.0))
        processing_times_list = [generate_time] * len(output)
        tool_calls_times_list = [tool_calls_time] * len(output)
    output.non_tensor_batch["processing_times"] = processing_times_list
    output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list
    return output


def _as_1d_object_array(values, batch_len: int) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.shape == (batch_len,):
        return values.astype(object, copy=False)

    normalized = np.empty(batch_len, dtype=object)
    for index in range(batch_len):
        try:
            normalized[index] = values[index]
        except (IndexError, KeyError, TypeError):
            normalized[index] = None
    return normalized


def _normalize_non_tensor_value_for_concat(values, batch_len: int) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.ndim == 1 and values.shape[0] == batch_len:
        return values
    return _as_1d_object_array(values, batch_len)


def _default_non_tensor_value(key: str, batch_len: int) -> np.ndarray:
    return np.array([None] * batch_len, dtype=object)


def _normalize_non_tensor_batches_for_concat(batches: list[DataProto]) -> list[DataProto]:
    if not batches:
        return batches

    all_non_tensor_keys: set[str] = set()
    for batch in batches:
        all_non_tensor_keys.update(batch.non_tensor_batch.keys())

    missing_counts: dict[str, int] = {}
    for batch in batches:
        batch_len = len(batch)
        normalized_non_tensor_batch = {}
        for key in sorted(all_non_tensor_keys):
            if key in batch.non_tensor_batch:
                normalized_non_tensor_batch[key] = _normalize_non_tensor_value_for_concat(
                    batch.non_tensor_batch[key], batch_len
                )
            else:
                normalized_non_tensor_batch[key] = _default_non_tensor_value(key, batch_len)
                missing_counts[key] = missing_counts.get(key, 0) + batch_len
        batch.non_tensor_batch = normalized_non_tensor_batch

    if missing_counts:
        print(f"[BatchUtils] Filled missing non_tensor fields before concat: {missing_counts}")

    return batches


def _coerce_param_version(value: Any, fallback: int = 0) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_trajectory_batches_for_logging(trajectory_batches: list[DataProto], tokenizer) -> list[DataProto]:
    """Keep trajectory logging batches concat-safe without changing train batches."""
    if not trajectory_batches:
        return []

    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    all_non_tensor_keys: set[str] = set()
    max_response_len = 0
    response_dtype = torch.long
    for trajectory_batch in trajectory_batches:
        all_non_tensor_keys.update(trajectory_batch.non_tensor_batch.keys())
        if trajectory_batch.batch is not None and "responses" in trajectory_batch.batch.keys():
            responses = trajectory_batch.batch["responses"]
            max_response_len = max(max_response_len, int(responses.shape[-1]))
            response_dtype = responses.dtype

    normalized_batches: list[DataProto] = []
    for trajectory_batch in trajectory_batches:
        batch_len = len(trajectory_batch)
        tensors: dict[str, torch.Tensor] = {}
        if max_response_len > 0:
            if trajectory_batch.batch is not None and "responses" in trajectory_batch.batch.keys():
                responses = trajectory_batch.batch["responses"].detach().cpu()
                if responses.shape[-1] < max_response_len:
                    pad_width = max_response_len - responses.shape[-1]
                    pad = torch.full(
                        (*responses.shape[:-1], pad_width),
                        fill_value=pad_token_id,
                        dtype=responses.dtype,
                    )
                    responses = torch.cat([responses, pad], dim=-1)
                elif responses.shape[-1] > max_response_len:
                    responses = responses[..., :max_response_len]
                tensors["responses"] = responses
            else:
                tensors["responses"] = torch.full(
                    (batch_len, max_response_len),
                    fill_value=pad_token_id,
                    dtype=response_dtype,
                )

        non_tensors = {}
        for key in sorted(all_non_tensor_keys):
            if key in trajectory_batch.non_tensor_batch:
                non_tensors[key] = _as_1d_object_array(trajectory_batch.non_tensor_batch[key], batch_len)
            else:
                non_tensors[key] = np.array([None] * batch_len, dtype=object)

        normalized_batches.append(DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={}))

    return normalized_batches


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_samples_batch = []
    rollout_status = rollout_samples[0].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        batch = addition_process(rs.full_batch)
        if len(batch) > 0:
            rollout_samples_batch.append(batch)
    if not rollout_samples_batch:
        raise ValueError("No trainable policy samples provided for batch assembly")
    rollout_samples_batch = _normalize_non_tensor_batches_for_concat(rollout_samples_batch)
    final_batch = DataProto.concat(rollout_samples_batch)

    trajectory_batches = [
        rs.trajectory_batch for rs in rollout_samples if rs.trajectory_batch is not None and len(rs.trajectory_batch) > 0
    ]
    if trajectory_batches:
        final_batch.meta_info["rollout_trajectory_batch"] = DataProto.concat(
            _normalize_trajectory_batches_for_logging(trajectory_batches, tokenizer)
        )

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    # Collect statistics
    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    param_version_start = []
    param_version_end = []
    for start_value, end_value in zip(
        final_batch.non_tensor_batch["min_global_steps"],
        final_batch.non_tensor_batch["max_global_steps"],
        strict=False,
    ):
        end = _coerce_param_version(end_value, fallback=0)
        start = _coerce_param_version(start_value, fallback=end)
        end = _coerce_param_version(end_value, fallback=start)
        param_version_start.append(start)
        param_version_end.append(end)
    final_batch.non_tensor_batch["min_global_steps"] = np.array(param_version_start, dtype=np.int64)
    final_batch.non_tensor_batch["max_global_steps"] = np.array(param_version_end, dtype=np.int64)
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    trajectory_param_versions = final_batch.non_tensor_batch["max_global_steps"]

    final_batch.meta_info.update(
        {
            "param_version_diversity": len(set(trajectory_param_versions)),
            "trajectory_param_versions": trajectory_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


class MetricsAggregator:
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int):
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
            "min": ["timing_s/agent_loop/tool_calls/min"],
            "avg": ["timing_s/agent_loop/tool_calls/mean"],
            "max": ["timing_s/agent_loop/tool_calls/max"],
            "last": [
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples_processed",
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_stale_samples",
                "training/global_step",  # TODO change name to: total_step
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int, timestamp: float = None):
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(v * c for v, c in zip(values, self.sample_counts, strict=False))
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t:.4f} seconds.")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["fully_async/trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }


def task_exception_handler(task: asyncio.Task):
    """Handle task exceptions and log them"""
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task was cancelled, this is expected
    except Exception as e:
        print(f"Task {task.get_name()} failed with exception: {e}")
        raise e


def safe_create_task(coro, name: str, task_set: set = None):
    """Safely create a task with exception handling

    Args:
        coro: The coroutine to run
        name: Name for the task
        task_set: Optional set to add the task to

    Returns:
        The created asyncio.Task
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(task_exception_handler)
    if task_set is not None:
        task_set.add(task)
    return task
