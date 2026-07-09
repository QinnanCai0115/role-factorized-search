# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import ast
import json
import logging
import os
import re
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.reward_score import search_r1_like_qa_em
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.two_stage_prompts import (
    build_final_backbone_message,
    build_initial_backbone_messages,
    build_next_backbone_message,
    build_policy_failure_backbone_message,
    extract_policy_output_from_backbone_followup,
)
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__file__)


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        grpo_group_key = config.get("grpo_group_key", "uid") if config is not None else "uid"
        if grpo_group_key not in data.non_tensor_batch:
            raise KeyError(f"GRPO group key '{grpo_group_key}' not found in non_tensor_batch")
        grpo_group_index = data.non_tensor_batch[grpo_group_key]
        if grpo_group_key != "uid" and "uid" in data.non_tensor_batch:
            uid_index = data.non_tensor_batch["uid"]
            grpo_group_index = np.array(
                [
                    uid_index[i] if group_id is None or str(group_id) == "" else group_id
                    for i, group_id in enumerate(grpo_group_index)
                ],
                dtype=object,
            )

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=grpo_group_index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    _VALIDATION_CONTROL_KEYS = {"enabled", "test_freq"}

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None
        self.backbone_rollout_wg = None
        self.backbone_async_rollout_manager = None
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        self.io_trace_log_path = str(rollout_custom.get("io_trace_log_path", os.getenv("VERL_IO_TRACE_LOG_PATH", "")) or "")
        self.io_trace_max_chars = int(rollout_custom.get("io_trace_max_chars", 4000))
        self.io_trace_max_items = int(rollout_custom.get("io_trace_max_items", 6))
        self.io_trace_max_samples = int(rollout_custom.get("io_trace_max_samples", 3))
        self.io_trace_record_sample_chain = bool(rollout_custom.get("io_trace_record_sample_chain", True))
        self._init_runtime_validation_control()

    def _init_runtime_validation_control(self) -> None:
        trainer_cfg = self.config.trainer
        try:
            default_test_freq = int(trainer_cfg.get("test_freq", -1))
        except (TypeError, ValueError):
            default_test_freq = -1

        self._runtime_validation_enabled = default_test_freq > 0
        self._runtime_validation_test_freq = default_test_freq
        self._validation_control_file = str(
            trainer_cfg.get("validation_control_file", os.getenv("VERL_VALIDATION_CONTROL_FILE", "")) or ""
        )
        self._validation_control_last_mtime: float | None = None
        self._validation_control_missing_warned = False

        if self._validation_control_file:
            print(
                "[validation control] watching "
                f"{self._validation_control_file} "
                f"(enabled={self._runtime_validation_enabled}, "
                f"test_freq={self._runtime_validation_test_freq})"
            )

    def _refresh_runtime_validation_control(self) -> None:
        if not self._validation_control_file:
            return

        try:
            stat_result = os.stat(self._validation_control_file)
        except FileNotFoundError:
            if not self._validation_control_missing_warned:
                print(
                    "[validation control] file not found, keeping startup validation settings: "
                    f"{self._validation_control_file}"
                )
                self._validation_control_missing_warned = True
            return
        except OSError as exc:
            logger.warning("Failed to stat validation control file %s: %s", self._validation_control_file, exc)
            return

        self._validation_control_missing_warned = False
        if self._validation_control_last_mtime == stat_result.st_mtime:
            return

        try:
            with open(self._validation_control_file, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read validation control file %s: %s", self._validation_control_file, exc)
            return

        if not isinstance(payload, dict):
            logger.warning(
                "Validation control file %s must contain a JSON object, got %s",
                self._validation_control_file,
                type(payload).__name__,
            )
            return

        unknown_keys = sorted(set(payload.keys()) - self._VALIDATION_CONTROL_KEYS)
        if unknown_keys:
            logger.warning(
                "Ignoring unknown validation control keys in %s: %s",
                self._validation_control_file,
                ", ".join(unknown_keys),
            )

        enabled = payload.get("enabled", self._runtime_validation_enabled)
        test_freq = payload.get("test_freq", self._runtime_validation_test_freq)

        try:
            enabled = bool(enabled)
            test_freq = int(test_freq)
        except (TypeError, ValueError) as exc:
            logger.warning("Invalid validation control payload in %s: %s", self._validation_control_file, exc)
            return

        self._runtime_validation_enabled = enabled
        self._runtime_validation_test_freq = test_freq
        self._validation_control_last_mtime = stat_result.st_mtime

        print(
            "[validation control] reloaded "
            f"{self._validation_control_file}: "
            f"enabled={self._runtime_validation_enabled}, "
            f"test_freq={self._runtime_validation_test_freq}"
        )

    def _should_run_validation(self, is_last_step: bool) -> bool:
        self._refresh_runtime_validation_control()
        if not self._runtime_validation_enabled:
            return False
        if self._runtime_validation_test_freq <= 0:
            return False
        return is_last_step or self.global_steps % self._runtime_validation_test_freq == 0

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    @staticmethod
    def _dump_jsonl_records(records: list[dict[str, Any]], dump_path: str, filename: str) -> str:
        """Dump structured rollout records as JSONL and return the written path."""
        os.makedirs(dump_path, exist_ok=True)
        output_path = os.path.join(dump_path, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                jsonable_record = RayPPOTrainer._to_jsonable_log_value(record)
                f.write(json.dumps(jsonable_record, ensure_ascii=False, default=str) + "\n")
        return output_path

    @staticmethod
    def _parse_orchestrator_round_branch(trace_id: Any, pair_group_id: Any = None) -> tuple[Optional[int], Optional[int]]:
        round_idx = None
        branch_idx = None
        trace_text = str(trace_id or "")
        trace_match = re.search(r"::r(\d+)b(\d+)$", trace_text)
        if trace_match:
            round_idx = int(trace_match.group(1))
            branch_idx = int(trace_match.group(2))

        if round_idx is None:
            group_text = str(pair_group_id or "")
            group_match = re.search(r"::round(\d+)$", group_text)
            if group_match:
                round_idx = int(group_match.group(1))
        return round_idx, branch_idx

    def _get_batch_tensor_row_stat(
        self,
        batch: DataProto,
        key: str,
        idx: int,
        *,
        reduce: str = "sum",
    ) -> Optional[float]:
        if batch.batch is None or key not in batch.batch.keys():
            return None
        tensor = batch.batch[key]
        if not isinstance(tensor, torch.Tensor):
            return None
        row = tensor[idx].detach()
        if row.ndim == 0:
            return float(row.cpu().item())
        if "response_mask" in batch.batch.keys() and row.shape == batch.batch["response_mask"][idx].shape:
            mask = batch.batch["response_mask"][idx].detach().to(row.device).bool()
            row = row[mask]
        if row.numel() == 0:
            return 0.0
        if reduce == "mean":
            return float(row.float().mean().cpu().item())
        return float(row.float().sum().cpu().item())

    def _get_rollout_ground_truth_at(self, batch: DataProto, idx: int) -> Any:
        reward_model = self._get_non_tensor_value_at(batch, "reward_model", idx)
        targets = self._collect_ground_truth_targets(reward_model)
        if len(targets) == 1:
            return targets[0]
        if len(targets) > 1:
            return targets
        answer = self._get_non_tensor_value_at(batch, "answer", idx)
        return self._to_jsonable_log_value(answer)

    def _extract_policy_round_full_trace_output_at(self, batch: DataProto, idx: int, fallback_text: str = "") -> str:
        details = self._get_non_tensor_value_at(batch, "final_backbone_binary_details", idx)
        if isinstance(details, dict):
            final_output = details.get("policy_full_trace_output", None)
            if isinstance(final_output, str) and final_output.strip():
                return final_output.strip()

        policy_output = self._get_non_tensor_value_at(batch, "policy_full_trace_output", idx)
        if isinstance(policy_output, str) and policy_output.strip():
            return policy_output.strip()

        fallback = str(fallback_text or "").strip()
        if fallback:
            return fallback
        return ""

    def _build_rollout_train_point_records(
        self,
        batch: DataProto,
        *,
        timing_raw: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del timing_raw
        if (
            len(batch) == 0
            or batch.batch is None
            or "responses" not in batch.batch.keys()
            or "token_level_scores" not in batch.batch.keys()
        ):
            return []

        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        records: list[dict[str, Any]] = []
        for i in range(len(batch)):
            policy_full_trace_output = self._extract_policy_round_full_trace_output_at(batch, i, outputs[i])
            policy_input = self._get_non_tensor_value_at(batch, "raw_prompt", i)
            if policy_input is None:
                policy_input = self._get_non_tensor_value_at(batch, "policy_prompt", i)
            policy_request_ids = self._get_non_tensor_value_at(batch, "policy_request_ids", i)
            request_id = self._get_non_tensor_value_at(batch, "request_id", i)
            if request_id is None and isinstance(policy_request_ids, (list, tuple)) and policy_request_ids:
                request_id = policy_request_ids[0]
            backbone_judge_details = self._get_non_tensor_value_at(batch, "final_backbone_binary_details", i)
            if not isinstance(backbone_judge_details, dict):
                backbone_judge_details = {}
            policy_format_penalty = self._get_non_tensor_value_at(batch, "policy_format_penalty", i)
            if policy_format_penalty is None:
                policy_format_penalty = backbone_judge_details.get("policy_format_penalty")
            final_policy_reward = self._get_non_tensor_value_at(batch, "final_policy_reward", i)
            if final_policy_reward is None:
                final_policy_reward = backbone_judge_details.get("final_policy_reward")
            backbone_judge_log = {
                "reason": backbone_judge_details.get("reason"),
                "retrieval_effective": backbone_judge_details.get("retrieval_effective"),
                "summary_reasonable": backbone_judge_details.get("summary_reasonable"),
                "raw_judge_response": backbone_judge_details.get("raw_judge_response"),
            }
            backbone_judge_log = {k: v for k, v in backbone_judge_log.items() if v not in (None, "", [])}
            trace_id = self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "orchestrator_trace_id", i))
            pair_group_id = self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "pair_group_id", i))
            record = {
                "record_type": "train_point",
                "step": int(self.global_steps),
                "train_point_index": i,
                "trace_id": trace_id,
                "trajectory_id": trace_id,
                "uid": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "uid", i)),
                "source_uid": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "source_uid", i)),
                "pair_group_id": pair_group_id,
                "backbone_search_id": pair_group_id,
                "policy_request_id": self._to_jsonable_log_value(request_id),
                "policy_request_ids": self._to_jsonable_log_value(policy_request_ids),
                "data_source": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "data_source", i)),
                "backbone_search_query": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "policy_prompt", i)
                ),
                "policy_input": self._to_jsonable_log_value(policy_input),
                "policy_output": self._to_jsonable_log_value(policy_full_trace_output) or outputs[i],
                "policy_internal_trace": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "tool_trace", i)
                ),
                "backbone_judge_binary_score": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "final_backbone_binary_score", i)
                    if self._get_non_tensor_value_at(batch, "final_backbone_binary_score", i) is not None
                    else self._get_non_tensor_value_at(batch, "backbone_binary_score", i)
                ),
                "backbone_judge_reason": self._to_jsonable_log_value(backbone_judge_details.get("reason")),
                "backbone_judge_retrieval_effective": self._to_jsonable_log_value(
                    backbone_judge_details.get("retrieval_effective")
                ),
                "backbone_judge_summary_reasonable": self._to_jsonable_log_value(
                    backbone_judge_details.get("summary_reasonable")
                ),
                "backbone_judge_raw_judge_response": self._to_jsonable_log_value(
                    backbone_judge_details.get("raw_judge_response")
                ),
                "backbone_judge": self._to_jsonable_log_value(backbone_judge_log) if backbone_judge_log else None,
                "policy_format_penalty": self._to_jsonable_log_value(policy_format_penalty),
                "final_policy_reward": self._to_jsonable_log_value(final_policy_reward),
            }
            records.append({k: v for k, v in record.items() if v not in (None, "", [])})
        return records

    @staticmethod
    def _extract_policy_final_answer_text(response_text: str) -> Optional[str]:
        text = str(response_text or "")
        matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE))
        if not matches:
            return None
        answer = matches[-1].group(1).strip()
        return answer or None

    def _extract_final_answer_text(self, response_text: str, final_source: str) -> str:
        source = str(final_source or "").strip().lower()
        if source == "policy":
            policy_answer = self._extract_policy_final_answer_text(response_text)
            if policy_answer:
                return policy_answer

        backbone_answer = self._extract_backbone_final_answer_text(response_text)
        if backbone_answer:
            return backbone_answer

        fallback_answer = search_r1_like_qa_em.extract_solution(str(response_text or ""))
        if isinstance(fallback_answer, str) and fallback_answer.strip():
            return fallback_answer.strip()
        return str(response_text or "").strip()

    @staticmethod
    def _build_trajectory_interaction_process(chain: Any) -> list[dict[str, Any]]:
        if not isinstance(chain, list):
            return []

        interactions: list[dict[str, Any]] = []
        for raw_event in chain:
            if not isinstance(raw_event, dict):
                continue

            stage = str(raw_event.get("stage", "") or "")
            if stage == "backbone_output":
                token_usage = raw_event.get("backbone_token_usage", None)
                interactions.append(
                    {
                        "actor": "backbone",
                        "raw_input": raw_event.get("raw_prompt", None),
                        "input": raw_event.get("question", ""),
                        "output": raw_event.get("response", ""),
                        "search_requested": bool(raw_event.get("has_tool_call", False)),
                        "token_usage": token_usage if isinstance(token_usage, dict) else None,
                    }
                )
            elif stage == "policy_input":
                interactions.append(
                    {
                        "actor": "policy",
                        "input": raw_event.get("query", ""),
                        "backbone_context": raw_event.get("backbone_response", ""),
                    }
                )
            elif stage == "policy_output":
                policy_entry = None
                for item in reversed(interactions):
                    if item.get("actor") == "policy" and "output" not in item:
                        policy_entry = item
                        break
                if policy_entry is None:
                    policy_entry = {"actor": "policy"}
                    interactions.append(policy_entry)
                policy_entry.update(
                    {
                        "request_id": raw_event.get("request_id", ""),
                        "output": raw_event.get("response", ""),
                        "internal_trace": raw_event.get("full_trace_output", ""),
                    }
                )
            elif stage == "policy_decision":
                policy_entry = None
                for item in reversed(interactions):
                    if item.get("actor") == "policy" and "backbone_judge_binary_score" not in item:
                        policy_entry = item
                        break
                if policy_entry is None:
                    policy_entry = {"actor": "policy"}
                    interactions.append(policy_entry)
                policy_entry.update(
                    {
                        "backbone_judge_binary_score": raw_event.get("binary_score", None),
                    }
                )
            elif stage == "backbone_final_output":
                token_usage = raw_event.get("backbone_token_usage", None)
                interactions.append(
                    {
                        "actor": "backbone",
                        "raw_input": raw_event.get("raw_prompt", None),
                        "input": raw_event.get("question", ""),
                        "output": raw_event.get("response", ""),
                        "final": True,
                        "token_usage": token_usage if isinstance(token_usage, dict) else None,
                    }
                )
        return [
            {k: v for k, v in item.items() if v not in (None, "", [])}
            for item in interactions
            if isinstance(item, dict)
        ]

    def _build_rollout_trajectory_records(
        self,
        batch: DataProto,
    ) -> list[dict[str, Any]]:
        if len(batch) == 0 or batch.batch is None or "responses" not in batch.batch.keys():
            return []

        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        records: list[dict[str, Any]] = []
        for i in range(len(batch)):
            trajectory_id = self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "orchestrator_trace_id", i))
            final_source = self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "backbone_final_source", i))
            if final_source in (None, ""):
                final_source = self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "response_source", i))
            policy_full_trace_output = self._extract_policy_round_full_trace_output_at(batch, i, outputs[i])
            final_output = (
                policy_full_trace_output
                if str(final_source or "").strip().lower() == "policy" and policy_full_trace_output
                else outputs[i]
            )
            final_answer = self._extract_final_answer_text(final_output, str(final_source or ""))
            final_answer_em = None
            final_answer_f1 = None
            em_from_batch = self._get_non_tensor_value_at(batch, "backbone_final_em", i)
            f1_from_batch = self._get_non_tensor_value_at(batch, "backbone_final_f1", i)
            if (
                str(final_source or "").strip().lower() == "backbone"
                and em_from_batch is not None
                and f1_from_batch is not None
            ):
                try:
                    final_answer_em = float(em_from_batch)
                    final_answer_f1 = float(f1_from_batch)
                except (TypeError, ValueError):
                    final_answer_em = None
                    final_answer_f1 = None
            if final_answer_em is None or final_answer_f1 is None:
                targets = self._collect_ground_truth_targets(self._get_non_tensor_value_at(batch, "reward_model", i))
                if not targets:
                    targets = self._collect_ground_truth_targets(self._get_non_tensor_value_at(batch, "answer", i))
                if targets and final_answer:
                    final_answer_em = float(search_r1_like_qa_em.em_check(final_answer, targets))
                    final_answer_f1 = float(max(self._token_f1_score(final_answer, t) for t in targets))
                else:
                    final_answer_em = 0.0
                    final_answer_f1 = 0.0

            chain = self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "orchestrator_chain", i))
            interactions = self._build_trajectory_interaction_process(chain)
            initial_question = None
            if isinstance(chain, list):
                for event in chain:
                    if isinstance(event, dict) and event.get("stage") == "backbone_output":
                        question = str(event.get("question", "")).strip()
                        if question:
                            initial_question = question
                            break

            record = {
                "record_type": "trajectory",
                "step": int(self.global_steps),
                "trajectory_index": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "trajectory_index", i)
                ),
                "trajectory_id": trajectory_id,
                "uid": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "uid", i)),
                "source_uid": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "source_uid", i)),
                "request_id": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "request_id", i)),
                "data_source": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "data_source", i)),
                "question": initial_question,
                "ground_truth": self._get_rollout_ground_truth_at(batch, i),
                "interactions": interactions,
                "final_answer_source": final_source,
                "final_output": final_output,
                "final_answer": final_answer,
                "backbone_deepseek_token_usage": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_deepseek_token_usage", i)
                ),
                "backbone_deepseek_input_tokens": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_deepseek_input_tokens", i)
                ),
                "backbone_deepseek_output_tokens": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_deepseek_output_tokens", i)
                ),
                "backbone_deepseek_total_tokens": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_deepseek_total_tokens", i)
                ),
                "final_em": final_answer_em,
                "final_f1": final_answer_f1,
            }
            records.append({k: v for k, v in record.items() if v not in (None, "", [])})
        return records

    def _log_rollout_data(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        timing_raw: dict,
        rollout_data_dir: str,
        *,
        trajectory_batch: Optional[DataProto] = None,
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Unused legacy argument kept for call-site compatibility.
            timing_raw (dict): Timing information for profiling the dump operation only.
            rollout_data_dir (str): Directory path to save the rollout data
            trajectory_batch (Optional[DataProto]): Final per-sample trajectory batch for trajectory logging
        """
        del reward_extra_infos_dict
        with marked_timer("dump_rollout_data", timing_raw, color="green"):
            train_point_records = self._build_rollout_train_point_records(batch, timing_raw=timing_raw)
            trajectory_log_batch = trajectory_batch if trajectory_batch is not None else batch
            trajectory_records = self._build_rollout_trajectory_records(trajectory_log_batch)
            train_points_path = self._dump_jsonl_records(
                train_point_records,
                os.path.join(rollout_data_dir, "train_points"),
                f"{self.global_steps}.jsonl",
            )
            trajectories_path = self._dump_jsonl_records(
                trajectory_records,
                os.path.join(rollout_data_dir, "trajectories"),
                f"{self.global_steps}.jsonl",
            )

            print(f"Dumped rollout train points to {train_points_path}")
            print(f"Dumped rollout trajectories to {trajectories_path}")
            self._append_io_trace(
                "train.rollout_data_dump",
                {
                    "train_points_path": train_points_path,
                    "trajectories_path": trajectories_path,
                    "num_train_points": len(train_point_records),
                    "num_trajectories": len(trajectory_records),
                },
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _drop_conflicting_rollout_non_tensor_keys(self, base_batch: DataProto, rollout_batch: DataProto) -> None:
        """Keep base-batch metadata as source-of-truth when unioning rollout outputs."""
        overlap_non_tensor_keys = set(base_batch.non_tensor_batch.keys()) & set(rollout_batch.non_tensor_batch.keys())
        for key in list(overlap_non_tensor_keys):
            lhs = base_batch.non_tensor_batch.get(key)
            rhs = rollout_batch.non_tensor_batch.get(key)
            same = False
            if isinstance(lhs, np.ndarray) and isinstance(rhs, np.ndarray):
                try:
                    same = np.array_equal(lhs, rhs)
                except Exception:
                    same = False
            if not same:
                rollout_batch.non_tensor_batch.pop(key, None)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _is_backbone_rollout_enabled(self) -> bool:
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        enabled_flag = bool(rollout_custom.get("enable_backbone_rollout", False))
        has_worker = self.backbone_rollout_wg is not None
        has_manager = self.backbone_async_rollout_manager is not None
        enabled = enabled_flag and (has_manager or has_worker)
        self._append_io_trace(
            "orchestrator.backbone_enabled_check",
            {
                "enabled": enabled,
                "enabled_flag": enabled_flag,
                "has_backbone_worker": has_worker,
                "has_backbone_manager": has_manager,
            },
        )
        return enabled

    def _infer_tool_call_mask(self, output: DataProto) -> np.ndarray:
        """Infer which samples should be forwarded to policy tool-agent rollout."""
        if "has_tool_call" in output.non_tensor_batch:
            has_tool_call = output.non_tensor_batch["has_tool_call"]
            return np.array([bool(x) for x in has_tool_call], dtype=bool)

        tool_call_patterns = ("<tool_call>", "<tool_calls>", "\"tool_calls\"", "<function_call>", "<search>")
        mask = np.zeros(len(output), dtype=bool)
        for i in range(len(output)):
            response_ids = output.batch["responses"][i]
            if "response_mask" in output.batch.keys():
                response_mask = output.batch["response_mask"][i].bool()
                valid_ids = response_ids[response_mask]
            else:
                valid_ids = response_ids
            response_str = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            mask[i] = any(p in response_str for p in tool_call_patterns)
        return mask

    def _merge_two_stage_outputs(
        self,
        backbone_output: DataProto,
        policy_output: DataProto,
        tool_call_mask: np.ndarray,
    ) -> DataProto:
        """Merge outputs from backbone/policy branches and restore original sample order."""
        no_tool_call_mask = ~tool_call_mask
        parts = []
        idx_parts = []

        if no_tool_call_mask.any():
            parts.append(backbone_output[no_tool_call_mask])
            idx_parts.append(np.where(no_tool_call_mask)[0])
        if tool_call_mask.any():
            parts.append(policy_output)
            idx_parts.append(np.where(tool_call_mask)[0])

        if len(parts) == 1:
            merged = parts[0]
            current_idx_order = idx_parts[0]
        else:
            merged = self._concat_dataproto_parts_with_seq_padding(parts)
            current_idx_order = np.concatenate(idx_parts, axis=0)

        restore_order = np.argsort(current_idx_order)
        return merged[restore_order]

    def _concat_dataproto_parts_with_seq_padding(self, parts: list[DataProto]) -> DataProto:
        """Concat DataProto parts, auto-padding tensor sequence dim (dim=1) when needed.

        In multi-round orchestration, different branches may produce different prompt lengths.
        TensorDict concat requires same non-batch shape, so we align variable-length sequence
        tensors (e.g. prompts/attention_mask) before calling DataProto.concat.
        """
        if len(parts) <= 1:
            return parts[0]

        # Work on deep copies to avoid mutating caller-visible references.
        normalized_parts = [p.select(deepcopy=True) for p in parts]
        all_batch_keys: set[str] = set()
        for p in normalized_parts:
            if p.batch is not None:
                all_batch_keys.update(p.batch.keys())

        if all_batch_keys:
            for key in all_batch_keys:
                ref_tensor = None
                for p in normalized_parts:
                    if p.batch is not None and key in p.batch.keys() and isinstance(p.batch[key], torch.Tensor):
                        ref_tensor = p.batch[key]
                        break
                if ref_tensor is None:
                    continue

                for p in normalized_parts:
                    if p.batch is None or key in p.batch.keys():
                        continue
                    fill_shape = (len(p), *ref_tensor.shape[1:])
                    p.batch[key] = torch.zeros(fill_shape, dtype=ref_tensor.dtype, device=ref_tensor.device)

                tensors = [p.batch[key] for p in normalized_parts]
                if any(not isinstance(t, torch.Tensor) for t in tensors):
                    continue
                if any(t.ndim < 2 for t in tensors):
                    continue

                seq_lens = [int(t.shape[1]) for t in tensors]
                max_seq_len = max(seq_lens)
                if min(seq_lens) == max_seq_len:
                    continue

                ref_ndim = tensors[0].ndim
                ref_tail_shape = tuple(tensors[0].shape[2:])
                for t in tensors:
                    if t.ndim != ref_ndim:
                        raise RuntimeError(
                            f"Cannot concat key '{key}': rank mismatch across parts "
                            f"{[tuple(x.shape) for x in tensors]}"
                        )
                    if tuple(t.shape[2:]) != ref_tail_shape:
                        raise RuntimeError(
                            f"Cannot concat key '{key}': non-sequence dims mismatch across parts "
                            f"{[tuple(x.shape) for x in tensors]}"
                        )

                for p, t in zip(normalized_parts, tensors, strict=False):
                    pad_len = max_seq_len - int(t.shape[1])
                    if pad_len <= 0:
                        continue
                    pad_shape = (t.shape[0], pad_len, *t.shape[2:])
                    pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
                    p.batch[key] = torch.cat([t, pad_tensor], dim=1)

        # Align non-tensor columns: DataProto.concat expects identical key sets.
        all_non_tensor_keys: set[str] = set()
        for p in normalized_parts:
            all_non_tensor_keys.update(p.non_tensor_batch.keys())

        for p in normalized_parts:
            part_len = len(p)
            for key in all_non_tensor_keys:
                if key not in p.non_tensor_batch:
                    if key in {"__num_turns__", "tool_call_counts"}:
                        p.non_tensor_batch[key] = np.zeros(part_len, dtype=np.int32)
                    else:
                        p.non_tensor_batch[key] = np.array([None] * part_len, dtype=object)
                    continue

                val = p.non_tensor_batch[key]
                if not isinstance(val, np.ndarray):
                    val = np.array(val, dtype=object)

                # Structured metadata sometimes arrives as a 2D+ object array when NumPy
                # auto-stacks equal-length inner lists. Concat expects one item per batch row.
                if val.dtype == object and val.ndim > 1 and val.shape[0] == part_len:
                    collapsed = np.empty(part_len, dtype=object)
                    for row_idx in range(part_len):
                        row_value = val[row_idx]
                        collapsed[row_idx] = row_value.tolist() if isinstance(row_value, np.ndarray) else row_value
                    val = collapsed

                # Defensive fix: ensure each non-tensor column is batch-aligned.
                if val.shape[0] != part_len:
                    if val.shape[0] == 1 and part_len > 1:
                        val = np.repeat(val, part_len, axis=0)
                    else:
                        if key in {"__num_turns__", "tool_call_counts"}:
                            fixed = np.zeros(part_len, dtype=np.int32)
                        else:
                            fixed = np.array([None] * part_len, dtype=object)
                        copy_len = min(part_len, val.shape[0])
                        if copy_len > 0:
                            fixed[:copy_len] = val[:copy_len]
                        val = fixed
                p.non_tensor_batch[key] = val

        return DataProto.concat(normalized_parts)

    def _get_max_orchestrator_rounds(self, *, is_validation: bool = False) -> int:
        """Maximum number of backbone->policy orchestration rounds per step."""
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        key = "validation_max_orchestrator_rounds" if is_validation else "max_orchestrator_rounds"
        try:
            rounds = int(rollout_custom.get(key, rollout_custom.get("max_orchestrator_rounds", 1)))
        except (TypeError, ValueError):
            rounds = 1
        return max(1, rounds)

    def _get_policy_rollout_n(self, *, is_validation: bool = False) -> int:
        """Branch factor for policy stage under two-stage orchestration."""
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        key = "validation_policy_rollout_n" if is_validation else "policy_rollout_n"
        try:
            n = int(rollout_custom.get(key, rollout_custom.get("policy_rollout_n", 1)))
        except (TypeError, ValueError):
            n = 1
        return max(1, n)

    def _get_max_backbone_search_queries(self, *, is_validation: bool = False) -> int:
        """Maximum number of atomic queries accepted from one backbone <search> block."""
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        key = "validation_max_backbone_search_queries" if is_validation else "max_backbone_search_queries"
        try:
            n = int(rollout_custom.get(key, rollout_custom.get("max_backbone_search_queries", 1)))
        except (TypeError, ValueError):
            n = 1
        return max(0, n)

    @staticmethod
    def _binary_reward_from_tool_rewards(tool_rewards: Any) -> int:
        """Convert tool reward list into binary continuation score."""
        if not isinstance(tool_rewards, (list, tuple)) or len(tool_rewards) == 0:
            return 0
        vals: list[float] = []
        for item in tool_rewards:
            try:
                vals.append(float(item))
            except (TypeError, ValueError):
                continue
        if not vals:
            return 0
        return int(max(vals) > 0.5)

    @staticmethod
    def _coerce_binary_judge_score(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            value = value.item() if value.size == 1 else value.tolist()
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.detach().cpu().flatten()[0].item()
        if isinstance(value, dict):
            for key in ("score", "binary_score", "final_backbone_binary_score", "backbone_binary_score"):
                score = RayPPOTrainer._coerce_binary_judge_score(value.get(key))
                if score is not None:
                    return score
            raw = value.get("raw_judge_response", None)
            if raw is not None:
                return RayPPOTrainer._coerce_binary_judge_score(raw)
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return RayPPOTrainer._coerce_binary_judge_score(json.loads(text))
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

    def _extract_existing_backbone_judge_score_at(self, batch: DataProto, idx: int) -> tuple[Optional[float], dict[str, Any]]:
        score = self._coerce_binary_judge_score(self._get_non_tensor_value_at(batch, "final_backbone_binary_score", idx))
        details = self._get_non_tensor_value_at(batch, "final_backbone_binary_details", idx)
        details_dict = deepcopy(details) if isinstance(details, dict) else {}
        if score is not None:
            return score, details_dict

        score = self._coerce_binary_judge_score(details_dict)
        if score is not None:
            return score, details_dict
        return None, details_dict

    def _truncate_backbone_judge_text(self, text: str) -> str:
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        try:
            max_chars = int(rollout_custom.get("backbone_judge_max_chars", 4000))
        except (TypeError, ValueError):
            max_chars = 4000
        text = str(text or "")
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "...(truncated)"

    def _build_policy_chain_for_orchestrator_backbone_judge(
        self,
        policy_output: DataProto,
        idx: int,
        *,
        policy_question: str,
        final_response_text: str,
    ) -> str:
        context = self._to_jsonable_log_value(self._get_non_tensor_value_at(policy_output, "policy_judge_context", idx))
        trace = self._to_jsonable_log_value(self._get_non_tensor_value_at(policy_output, "policy_judge_trace", idx))
        if not isinstance(context, dict):
            raw_prompt = self._get_non_tensor_value_at(policy_output, "raw_prompt", idx)
            context = {"initial_messages": self._normalize_messages(raw_prompt)}
        if not isinstance(trace, list):
            trace = []

        payload = {
            "policy_prompt": context.get("initial_messages", [{"role": "user", "content": policy_question}]),
            "policy_trace": trace,
            "final_policy_output": self._truncate_backbone_judge_text(final_response_text),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _call_orchestrator_backbone_binary_judge(
        self,
        *,
        question: str,
        policy_chain_text: str,
    ) -> tuple[Optional[float], dict[str, Any]]:
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        api_mode = str(rollout_custom.get("backbone_api_mode", "openai_compatible")).lower()
        api_url = str(rollout_custom.get("backbone_api_url", ""))
        api_model = str(rollout_custom.get("backbone_api_model", "deepseek-reasoner"))
        api_key = str(
            rollout_custom.get("backbone_api_key", "")
            or os.environ.get("BACKBONE_API_KEY", "")
            or os.environ.get("DEEPSEEK_API_KEY", "")
        )
        try:
            timeout = float(rollout_custom.get("backbone_judge_timeout", rollout_custom.get("backbone_api_timeout", 30.0)))
        except (TypeError, ValueError):
            timeout = 30.0
        if not api_url or not api_model or api_mode != "openai_compatible":
            return None, {"error": "backbone judge requires openai_compatible backbone_api_url/backbone_api_model"}

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
            f"{self._truncate_backbone_judge_text(policy_chain_text)}"
        )

        try:
            import requests

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
            disable_backbone_proxy = str(os.environ.get("BACKBONE_API_NO_PROXY", "")).strip().lower() not in (
                "",
                "0",
                "false",
                "no",
            )
            with requests.Session() as session:
                if disable_backbone_proxy:
                    session.trust_env = False
                resp = session.post(endpoint, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", []) if isinstance(data, dict) else []
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content", "") if isinstance(message, dict) else ""
            details: dict[str, Any] = {"raw_judge_response": content}
            score = self._coerce_binary_judge_score(content)
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
        except Exception as e:
            logger.warning(f"Orchestrator backbone binary judge failed: {e}")
            return None, {"error": str(e)}

    def _judge_policy_output_with_backbone_binary_at(
        self,
        policy_output: DataProto,
        idx: int,
        *,
        policy_question: str,
        final_response_text: str,
    ) -> tuple[float, dict[str, Any], bool]:
        score, details = self._extract_existing_backbone_judge_score_at(policy_output, idx)
        if score is not None:
            return score, details, False

        policy_chain_text = self._build_policy_chain_for_orchestrator_backbone_judge(
            policy_output,
            idx,
            policy_question=policy_question,
            final_response_text=final_response_text,
        )
        judged, details = self._call_orchestrator_backbone_binary_judge(
            question=policy_question,
            policy_chain_text=policy_chain_text,
        )
        details = details or {}
        details["policy_chain"] = policy_chain_text
        details["policy_full_trace_output"] = policy_chain_text
        if judged is None:
            return 0.0, details, True
        return float(1.0 if judged > 0.5 else 0.0), details, True

    @staticmethod
    def _accumulate_round_timing(timing_raw: dict[str, Any], timing: dict[str, Any], prefix: str) -> None:
        """Accumulate timing metrics under round-aware prefix keys."""
        for key, value in timing.items():
            metric_key = f"{prefix}/{key}"
            if isinstance(value, (int, float, np.integer, np.floating)):
                timing_raw[metric_key] = float(timing_raw.get(metric_key, 0.0)) + float(value)
            else:
                timing_raw[metric_key] = value

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
                out.append(f"...(truncated_items={len(value) - self.io_trace_max_items})")
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
                "global_step": int(getattr(self, "global_steps", -1)),
                "payload": self._truncate_trace_value(payload),
            }
            with open(self.io_trace_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _to_jsonable_log_value(value: Any, depth: int = 0) -> Any:
        if depth > 6:
            return "...(max_depth)"
        if isinstance(value, np.ndarray):
            return RayPPOTrainer._to_jsonable_log_value(value.tolist(), depth + 1)
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return RayPPOTrainer._to_jsonable_log_value(value.detach().cpu().tolist(), depth + 1)
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): RayPPOTrainer._to_jsonable_log_value(v, depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [RayPPOTrainer._to_jsonable_log_value(v, depth + 1) for v in value]
        return str(value)

    @staticmethod
    def _make_object_array(values: list[Any]) -> np.ndarray:
        out = np.empty(len(values), dtype=object)
        for i, value in enumerate(values):
            out[i] = value
        return out

    @staticmethod
    def _extract_numeric_timing_summary(timing_raw: dict[str, Any]) -> dict[str, float]:
        summary: dict[str, float] = {}
        for key in sorted(timing_raw.keys()):
            value = timing_raw[key]
            if isinstance(value, (int, float, np.integer, np.floating)):
                summary[str(key)] = float(value)
        return summary

    @staticmethod
    def _extract_orchestrator_timing_summary(timing_raw: dict[str, Any]) -> dict[str, float]:
        prefixes = ("backbone_round_", "policy_round_", "backbone_final_round_", "validation_rollout/")
        return {k: v for k, v in RayPPOTrainer._extract_numeric_timing_summary(timing_raw).items() if k.startswith(prefixes)}

    @staticmethod
    def _extract_update_timing_summary(timing_raw: dict[str, Any]) -> dict[str, float]:
        exact_keys = {
            "reward",
            "old_log_prob",
            str(Role.RefPolicy),
            "values",
            "adv",
            "update_critic",
            "update_actor",
            "update_weights",
            "save_checkpoint",
            "validation_reward",
            "validation_update_weights",
            "validation_reward_extract",
        }
        return {
            k: v
            for k, v in RayPPOTrainer._extract_numeric_timing_summary(timing_raw).items()
            if k in exact_keys or k.startswith("update_") or k.startswith("validation_update_")
        }

    @staticmethod
    def _get_non_tensor_value_at(batch: DataProto, key: str, idx: int) -> Any:
        values = batch.non_tensor_batch.get(key, None)
        if values is None:
            return None
        try:
            return values[idx]
        except Exception:
            return None

    def _build_sample_interaction_logs(self, batch: DataProto) -> list[dict[str, Any]]:
        logs: list[dict[str, Any]] = []
        for i in range(len(batch)):
            record = {
                "uid": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "uid", i)),
                "source_uid": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "source_uid", i)),
                "trace_id": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "orchestrator_trace_id", i)),
                "request_id": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "request_id", i)),
                "data_source": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "data_source", i)),
                "orchestrator_round_count": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "orchestrator_round_count", i)
                ),
                "response_source": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "response_source", i)
                ),
                "backbone_final_source": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_final_source", i)
                ),
                "final_backbone_binary_score": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "final_backbone_binary_score", i)
                ),
                "policy_request_ids": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "policy_request_ids", i)
                ),
                "policy_prompt": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "policy_prompt", i)),
                "policy_full_trace_output": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "policy_full_trace_output", i)
                ),
                "backbone_final_answer": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_final_answer", i)
                ),
                "backbone_deepseek_input_tokens": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_deepseek_input_tokens", i)
                ),
                "backbone_deepseek_output_tokens": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_deepseek_output_tokens", i)
                ),
                "backbone_deepseek_total_tokens": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_deepseek_total_tokens", i)
                ),
                "backbone_deepseek_token_usage": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_deepseek_token_usage", i)
                ),
                "backbone_final_em": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_final_em", i)
                ),
                "backbone_final_f1": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_final_f1", i)
                ),
                "orchestrator_chain": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "orchestrator_chain", i)
                ),
            }
            logs.append({k: v for k, v in record.items() if v not in (None, "", [])})
        return logs

    def _build_validation_backbone_deepseek_token_records(
        self,
        batch: DataProto,
        *,
        phase_batch_index: int,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for i in range(len(batch)):
            chain = self._get_non_tensor_value_at(batch, "orchestrator_chain", i)
            backbone_calls: list[dict[str, Any]] = []
            if isinstance(chain, list):
                for event in chain:
                    if not isinstance(event, dict):
                        continue
                    if event.get("stage") not in {"backbone_output", "backbone_final_output"}:
                        continue
                    backbone_calls.append(
                        {
                            "round": event.get("round"),
                            "stage": event.get("stage"),
                            "input_source": event.get("input_source"),
                            "input_raw_prompt": event.get("raw_prompt"),
                            "input_question": event.get("question"),
                            "output": event.get("response"),
                            "has_tool_call": event.get("has_tool_call"),
                            "deepseek_usage": event.get("backbone_token_usage"),
                        }
                    )

            record = {
                "record_type": "validation_backbone_deepseek_tokens",
                "step": int(self.global_steps),
                "validation_batch_index": phase_batch_index,
                "sample_index": i,
                "uid": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "uid", i)),
                "source_uid": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "source_uid", i)),
                "trace_id": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "orchestrator_trace_id", i)),
                "data_source": self._to_jsonable_log_value(self._get_non_tensor_value_at(batch, "data_source", i)),
                "final_answer_source": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_final_source", i)
                ),
                "final_backbone_answer": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_final_answer", i)
                ),
                "token_summary": self._to_jsonable_log_value(
                    self._get_non_tensor_value_at(batch, "backbone_deepseek_token_usage", i)
                ),
                "backbone_calls": self._to_jsonable_log_value(backbone_calls),
            }
            records.append({k: v for k, v in record.items() if v not in (None, "", [])})
        return records

    def _build_per_sample_log_fields(
        self,
        batch: DataProto,
        *,
        phase: str,
        timing_raw: Optional[dict[str, Any]] = None,
        phase_batch_index: Optional[int] = None,
    ) -> dict[str, list[Any]]:
        batch_size = len(batch)
        timing_summary = self._extract_numeric_timing_summary(timing_raw or {})
        orchestrator_timing_summary = self._extract_orchestrator_timing_summary(timing_raw or {})
        update_timing_summary = self._extract_update_timing_summary(timing_raw or {})
        return {
            "log_phase": [phase] * batch_size,
            "log_phase_batch_index": [phase_batch_index] * batch_size,
            "log_timing_summary": [deepcopy(timing_summary) for _ in range(batch_size)],
            "log_orchestrator_timing_summary": [deepcopy(orchestrator_timing_summary) for _ in range(batch_size)],
            "log_update_timing_summary": [deepcopy(update_timing_summary) for _ in range(batch_size)],
            "log_sample_interaction": self._build_sample_interaction_logs(batch),
        }

    def _append_batch_log_trace(
        self,
        *,
        phase: str,
        batch: DataProto,
        timing_raw: Optional[dict[str, Any]] = None,
        phase_batch_index: Optional[int] = None,
    ) -> None:
        backbone_em_preview = self._extract_non_tensor_preview(batch, "backbone_final_em")
        backbone_f1_preview = self._extract_non_tensor_preview(batch, "backbone_final_f1")
        backbone_em_values = batch.non_tensor_batch.get("backbone_final_em", None)
        backbone_f1_values = batch.non_tensor_batch.get("backbone_final_f1", None)
        backbone_em_mean = None
        backbone_f1_mean = None
        if isinstance(backbone_em_values, np.ndarray) and backbone_em_values.shape[0] == len(batch):
            try:
                backbone_em_mean = float(np.mean(backbone_em_values.astype(np.float32)))
            except Exception:
                backbone_em_mean = None
        if isinstance(backbone_f1_values, np.ndarray) and backbone_f1_values.shape[0] == len(batch):
            try:
                backbone_f1_mean = float(np.mean(backbone_f1_values.astype(np.float32)))
            except Exception:
                backbone_f1_mean = None
        self._append_io_trace(
            f"{phase}.batch_log_summary",
            {
                "phase_batch_index": phase_batch_index,
                "batch_size": len(batch),
                "uids_preview": self._extract_non_tensor_preview(batch, "uid"),
                "trace_ids_preview": self._extract_non_tensor_preview(batch, "orchestrator_trace_id"),
                "backbone_final_em_preview": backbone_em_preview,
                "backbone_final_f1_preview": backbone_f1_preview,
                "backbone_final_em_mean": backbone_em_mean,
                "backbone_final_f1_mean": backbone_f1_mean,
                "timing_summary": self._extract_numeric_timing_summary(timing_raw or {}),
                "orchestrator_timing_summary": self._extract_orchestrator_timing_summary(timing_raw or {}),
                "update_timing_summary": self._extract_update_timing_summary(timing_raw or {}),
                "sample_interaction_preview": self._build_sample_interaction_logs(batch)[: self.io_trace_max_samples],
            },
        )

    def _extract_raw_prompt_preview(self, batch: DataProto) -> list[Any]:
        previews: list[Any] = []
        raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
        if raw_prompts is None:
            return previews
        max_samples = min(len(batch), max(1, self.io_trace_max_samples))
        for i in range(max_samples):
            try:
                previews.append(raw_prompts[i])
            except Exception:
                previews.append(str(raw_prompts))
                break
        return previews

    def _extract_response_preview(self, batch: DataProto) -> list[str]:
        previews: list[str] = []
        max_samples = min(len(batch), max(1, self.io_trace_max_samples))
        for i in range(max_samples):
            response_ids = batch.batch["responses"][i]
            if "response_mask" in batch.batch.keys():
                response_mask = batch.batch["response_mask"][i].bool()
                valid_ids = response_ids[response_mask]
            else:
                valid_ids = response_ids
            previews.append(self.tokenizer.decode(valid_ids, skip_special_tokens=True))
        return previews

    def _extract_non_tensor_preview(self, batch: DataProto, key: str) -> list[Any]:
        previews: list[Any] = []
        values = batch.non_tensor_batch.get(key, None)
        if values is None:
            return previews
        max_samples = min(len(batch), max(1, self.io_trace_max_samples))
        for i in range(max_samples):
            try:
                previews.append(values[i])
            except Exception:
                previews.append(None)
        return previews

    def _get_orchestrator_trace_ids(self, batch: DataProto) -> list[str]:
        trace_ids_arr = batch.non_tensor_batch.get("orchestrator_trace_id", None)
        trace_ids: list[str] = []

        if isinstance(trace_ids_arr, np.ndarray) and trace_ids_arr.shape[0] == len(batch):
            for i in range(len(batch)):
                trace_ids.append(str(trace_ids_arr[i]))
            return trace_ids

        uid_arr = batch.non_tensor_batch.get("uid", None)
        for i in range(len(batch)):
            uid_part = ""
            if isinstance(uid_arr, np.ndarray) and uid_arr.shape[0] == len(batch):
                uid_part = str(uid_arr[i])
            if not uid_part:
                uid_part = f"sample{i}"
            trace_ids.append(f"gs{int(getattr(self, 'global_steps', -1))}_i{i}_{uid_part}")

        batch.non_tensor_batch["orchestrator_trace_id"] = np.array(trace_ids, dtype=object)
        return trace_ids

    def _extract_last_user_content(self, raw_prompt: Any) -> str:
        for msg in reversed(self._normalize_messages(raw_prompt)):
            if msg.get("role") == "user":
                content = str(msg.get("content", "")).strip()
                if content:
                    return content
        return ""

    @staticmethod
    def _summarize_tool_reward_lengths(batch: DataProto) -> dict[str, Any]:
        tool_rewards_arr = batch.non_tensor_batch.get("tool_rewards", None)
        if tool_rewards_arr is None:
            return {
                "num_trajectories": len(batch),
                "num_with_tool_rewards": 0,
                "num_without_tool_rewards": len(batch),
                "total_reward_points": len(batch),
                "length_histogram": {},
            }

        lengths: list[int] = []
        for i in range(len(batch)):
            rewards = tool_rewards_arr[i]
            if isinstance(rewards, (list, tuple)):
                lengths.append(len(rewards))
            else:
                lengths.append(0)

        length_histogram: dict[str, int] = {}
        for length in lengths:
            key = str(length)
            length_histogram[key] = length_histogram.get(key, 0) + 1

        total_reward_points = 0
        for length in lengths:
            total_reward_points += length if length > 0 else 1

        return {
            "num_trajectories": len(batch),
            "num_with_tool_rewards": sum(1 for length in lengths if length > 0),
            "num_without_tool_rewards": sum(1 for length in lengths if length == 0),
            "total_reward_points": total_reward_points,
            "length_histogram": length_histogram,
            "length_preview": lengths[:8],
        }

    def _strip_backbone_instruction_from_user_content(self, content: str) -> str:
        marker = "You are a frozen backbone reasoning model."
        if marker not in content:
            return content

        question_match = re.search(r"<question>\s*(.*?)\s*</question>", content, flags=re.DOTALL)
        if question_match:
            question = question_match.group(1).strip()
            if question:
                return question

        if "Question:" in content:
            tail = content.split("Question:", 1)[1].strip()
            if tail:
                return tail

        return content

    def _sanitize_policy_raw_prompt(self, raw_prompt: Any) -> Any:
        if not isinstance(raw_prompt, (list, tuple)):
            return raw_prompt

        sanitized_messages = []
        for msg in raw_prompt:
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                new_msg = dict(msg)
                new_msg["content"] = self._strip_backbone_instruction_from_user_content(msg["content"])
                sanitized_messages.append(new_msg)
            else:
                sanitized_messages.append(msg)
        return sanitized_messages

    def _sanitize_policy_batch_raw_prompts(self, batch: DataProto) -> None:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
        if raw_prompts is None:
            return

        sanitized = [self._sanitize_policy_raw_prompt(raw_prompts[i]) for i in range(len(batch))]
        batch.non_tensor_batch["raw_prompt"] = np.array(sanitized, dtype=object)

    @staticmethod
    def _maybe_parse_prompt_messages_from_string(value: str) -> Optional[list[dict[str, str]]]:
        text = str(value or "").strip()
        if not text or text[0] not in "[{":
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            messages = RayPPOTrainer._normalize_messages(parsed)
            if messages:
                return messages
        return None

    @staticmethod
    def _extract_initial_backbone_question(raw_prompt: Any) -> str:
        messages = RayPPOTrainer._normalize_messages(raw_prompt)
        if len(messages) == 1:
            parsed_messages = RayPPOTrainer._maybe_parse_prompt_messages_from_string(messages[0].get("content", ""))
            if parsed_messages:
                messages = parsed_messages

        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            question_match = re.search(r"<question>\s*(.*?)\s*</question>", content, flags=re.DOTALL)
            if question_match and question_match.group(1).strip():
                return question_match.group(1).strip()
            return content
        return ""

    @staticmethod
    def _extract_last_policy_output_for_final_backbone(messages: list[dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            policy_output = extract_policy_output_from_backbone_followup(msg.get("content", ""))
            if policy_output:
                return policy_output
        return ""

    @staticmethod
    def _ensure_backbone_final_answer_instruction(raw_prompt: Any, force_final_answer: bool = False) -> list[dict[str, str]]:
        messages = RayPPOTrainer._normalize_messages(raw_prompt)
        if force_final_answer:
            if messages and str(messages[-1].get("content", "")).startswith("Final round. Do not output another <search>."):
                return messages
            policy_output = RayPPOTrainer._extract_last_policy_output_for_final_backbone(messages)
            messages.append(build_final_backbone_message(policy_output))
            return messages

        question = RayPPOTrainer._extract_initial_backbone_question(raw_prompt)
        return build_initial_backbone_messages(question)

    def _decode_batch_responses(self, batch: DataProto) -> list[str]:
        responses: list[str] = []
        for i in range(len(batch)):
            response_ids = batch.batch["responses"][i]
            if "response_mask" in batch.batch.keys():
                response_mask = batch.batch["response_mask"][i].bool()
                response_ids = response_ids[response_mask]
            responses.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))
        return responses

    @staticmethod
    def _normalize_deepseek_usage(value: Any) -> Optional[dict[str, Any]]:
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            else:
                value = value.tolist()
        if not isinstance(value, dict):
            return None
        return RayPPOTrainer._to_jsonable_log_value(value)

    @staticmethod
    def _usage_int(usage: Optional[dict[str, Any]], key: str) -> Optional[int]:
        if not isinstance(usage, dict):
            return None
        value = usage.get(key, None)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _build_backbone_deepseek_token_usage(
        self,
        batch: DataProto,
        idx: int,
        *,
        round_idx: int,
        stage: str,
        trace_id: str,
    ) -> dict[str, Any]:
        return self._build_backbone_deepseek_token_usage_from_raw_usage(
            self._get_non_tensor_value_at(batch, "backbone_api_usage", idx),
            round_idx=round_idx,
            stage=stage,
            trace_id=trace_id,
        )

    def _build_backbone_deepseek_token_usage_from_raw_usage(
        self,
        raw_usage: Any,
        *,
        round_idx: int,
        stage: str,
        trace_id: str,
    ) -> dict[str, Any]:
        usage = self._normalize_deepseek_usage(raw_usage)
        prompt_tokens = self._usage_int(usage, "prompt_tokens")
        completion_tokens = self._usage_int(usage, "completion_tokens")
        total_tokens = self._usage_int(usage, "total_tokens")
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
        record = {
            "round": int(round_idx),
            "stage": stage,
            "trace_id": str(trace_id),
            "model": str(rollout_custom.get("backbone_api_model", "")),
            "source": "deepseek_api_usage",
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "usage": usage,
            "usage_missing": usage is None,
        }
        return {k: v for k, v in record.items() if v not in ("", [])}

    @staticmethod
    def _summarize_backbone_deepseek_token_usage(chain: Any) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        known_input_tokens = 0
        known_output_tokens = 0
        known_total_tokens = 0
        missing_count = 0

        if isinstance(chain, list):
            for event in chain:
                if not isinstance(event, dict):
                    continue
                if event.get("stage") not in {"backbone_output", "backbone_final_output"}:
                    continue
                usage_record = event.get("backbone_token_usage", None)
                if not isinstance(usage_record, dict):
                    usage_record = {
                        "round": event.get("round"),
                        "stage": event.get("stage"),
                        "trace_id": event.get("trace_id"),
                        "source": "deepseek_api_usage",
                        "usage_missing": True,
                    }
                calls.append(deepcopy(usage_record))
                if usage_record.get("usage_missing", False):
                    missing_count += 1
                    continue
                input_tokens = usage_record.get("input_tokens", None)
                output_tokens = usage_record.get("output_tokens", None)
                total_tokens = usage_record.get("total_tokens", None)
                if input_tokens is None or output_tokens is None or total_tokens is None:
                    missing_count += 1
                    continue
                known_input_tokens += int(input_tokens)
                known_output_tokens += int(output_tokens)
                known_total_tokens += int(total_tokens)

        has_missing = missing_count > 0
        return {
            "source": "deepseek_api_usage",
            "num_calls": len(calls),
            "num_missing_usage": missing_count,
            "input_tokens": None if has_missing else known_input_tokens,
            "output_tokens": None if has_missing else known_output_tokens,
            "total_tokens": None if has_missing else known_total_tokens,
            "known_input_tokens": known_input_tokens,
            "known_output_tokens": known_output_tokens,
            "known_total_tokens": known_total_tokens,
            "calls": calls,
        }

    @staticmethod
    def _iter_backbone_deepseek_usage_summaries(batch: DataProto) -> list[dict[str, Any]]:
        if batch is None or batch.non_tensor_batch is None:
            return []
        values = batch.non_tensor_batch.get("backbone_deepseek_token_usage", None)
        if values is None:
            return []
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, (list, tuple)):
            values = [values]
        return [value for value in values if isinstance(value, dict)]

    @staticmethod
    def _safe_usage_int(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _compute_backbone_deepseek_step_token_metrics(self, batch: DataProto) -> dict[str, float]:
        summaries = self._iter_backbone_deepseek_usage_summaries(batch)
        if not summaries:
            return {}

        rows = len(summaries)
        calls = 0
        missing_calls = 0
        known_input_tokens = 0
        known_output_tokens = 0
        known_total_tokens = 0

        for summary in summaries:
            calls += self._safe_usage_int(summary.get("num_calls", 0))
            missing_calls += self._safe_usage_int(summary.get("num_missing_usage", 0))
            known_input_tokens += self._safe_usage_int(summary.get("known_input_tokens", 0))
            known_output_tokens += self._safe_usage_int(summary.get("known_output_tokens", 0))
            known_total_tokens += self._safe_usage_int(summary.get("known_total_tokens", 0))

        known_calls = max(calls - missing_calls, 0)
        prefix = "token_usage/backbone_deepseek"
        metrics: dict[str, float] = {
            f"{prefix}/rows": float(rows),
            f"{prefix}/calls": float(calls),
            f"{prefix}/known_calls": float(known_calls),
            f"{prefix}/missing_usage_calls": float(missing_calls),
            f"{prefix}/known_input_tokens": float(known_input_tokens),
            f"{prefix}/known_output_tokens": float(known_output_tokens),
            f"{prefix}/known_total_tokens": float(known_total_tokens),
        }
        if known_calls > 0:
            metrics[f"{prefix}/known_input_tokens_per_call"] = float(known_input_tokens / known_calls)
            metrics[f"{prefix}/known_output_tokens_per_call"] = float(known_output_tokens / known_calls)
            metrics[f"{prefix}/known_total_tokens_per_call"] = float(known_total_tokens / known_calls)
        if rows > 0:
            metrics[f"{prefix}/known_total_tokens_per_row"] = float(known_total_tokens / rows)

        cumulative = getattr(self, "_backbone_deepseek_cumulative_token_usage", None)
        if not isinstance(cumulative, dict):
            cumulative = {
                "calls": 0,
                "known_calls": 0,
                "missing_usage_calls": 0,
                "known_input_tokens": 0,
                "known_output_tokens": 0,
                "known_total_tokens": 0,
            }
            self._backbone_deepseek_cumulative_token_usage = cumulative
        cumulative["calls"] += calls
        cumulative["known_calls"] += known_calls
        cumulative["missing_usage_calls"] += missing_calls
        cumulative["known_input_tokens"] += known_input_tokens
        cumulative["known_output_tokens"] += known_output_tokens
        cumulative["known_total_tokens"] += known_total_tokens

        cumulative_prefix = f"{prefix}/cumulative"
        metrics.update(
            {
                f"{cumulative_prefix}_calls": float(cumulative["calls"]),
                f"{cumulative_prefix}_known_calls": float(cumulative["known_calls"]),
                f"{cumulative_prefix}_missing_usage_calls": float(cumulative["missing_usage_calls"]),
                f"{cumulative_prefix}_known_input_tokens": float(cumulative["known_input_tokens"]),
                f"{cumulative_prefix}_known_output_tokens": float(cumulative["known_output_tokens"]),
                f"{cumulative_prefix}_known_total_tokens": float(cumulative["known_total_tokens"]),
            }
        )
        return metrics

    @staticmethod
    def _collect_ground_truth_targets(ground_truth: Any) -> list[str]:
        if isinstance(ground_truth, dict):
            target = ground_truth.get("target", None)
            if target is None:
                target = ground_truth.get("ground_truth", None)
        else:
            target = ground_truth

        if isinstance(target, str):
            text = target.strip()
            return [text] if text else []

        if isinstance(target, (list, tuple)):
            out: list[str] = []
            for item in target:
                text = str(item).strip()
                if text:
                    out.append(text)
            return out

        if target is None:
            return []

        text = str(target).strip()
        return [text] if text else []

    @staticmethod
    def _token_f1_score(prediction: str, gold: str) -> float:
        pred_tokens = search_r1_like_qa_em.normalize_answer(prediction).split()
        gold_tokens = search_r1_like_qa_em.normalize_answer(gold).split()
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

    @staticmethod
    def _extract_backbone_final_answer_text(response_text: str) -> Optional[str]:
        text = str(response_text or "")
        matches = list(
            re.finditer(r"<final[_ ]answer>\s*(.*?)\s*</final[_ ]answer>", text, flags=re.DOTALL | re.IGNORECASE)
        )
        if not matches:
            return None
        answer = matches[-1].group(1).strip()
        return answer or None

    def _compute_backbone_reference_scores(self, batch: DataProto) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        answers = batch.non_tensor_batch.get("backbone_final_answer", None)
        reward_model = batch.non_tensor_batch.get("reward_model", None)
        final_sources = batch.non_tensor_batch.get("backbone_final_source", None)

        if not isinstance(answers, np.ndarray) or answers.shape[0] != len(batch):
            return None, None
        if not isinstance(reward_model, np.ndarray) or reward_model.shape[0] != len(batch):
            return None, None
        has_final_sources = isinstance(final_sources, np.ndarray) and final_sources.shape[0] == len(batch)

        ems: list[float] = []
        f1s: list[float] = []
        for i in range(len(batch)):
            final_source = ""
            if has_final_sources and final_sources[i] is not None:
                final_source = str(final_sources[i]).strip().lower()
            if final_source and final_source != "backbone":
                ems.append(0.0)
                f1s.append(0.0)
                continue

            raw_answer = str(answers[i]) if answers[i] is not None else ""
            parsed_answer = self._extract_backbone_final_answer_text(raw_answer)
            if final_source == "backbone":
                final_answer = parsed_answer or ""
            else:
                fallback_answer = search_r1_like_qa_em.extract_solution(raw_answer)
                final_answer = parsed_answer or (
                    fallback_answer if isinstance(fallback_answer, str) and fallback_answer.strip() else raw_answer
                )

            targets = self._collect_ground_truth_targets(reward_model[i])
            if not targets or not final_answer.strip():
                ems.append(0.0)
                f1s.append(0.0)
                continue

            em = float(search_r1_like_qa_em.em_check(final_answer, targets))
            best_f1 = max(self._token_f1_score(final_answer, t) for t in targets)
            ems.append(em)
            f1s.append(float(best_f1))

        return np.array(ems, dtype=np.float32), np.array(f1s, dtype=np.float32)

    def _extract_backbone_search_query(self, backbone_response: str) -> str:
        text = str(backbone_response or "")
        matches = list(re.finditer(r"<search>\s*(.*?)\s*</search>", text, flags=re.DOTALL | re.IGNORECASE))
        if matches:
            query = matches[-1].group(1).strip()
            if query:
                return query

        m_tool = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL | re.IGNORECASE)
        if m_tool:
            try:
                payload = json.loads(m_tool.group(1))
                if isinstance(payload, dict):
                    args = payload.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    if isinstance(args, dict):
                        query = str(args.get("query", "")).strip()
                        if query:
                            return query
            except Exception:
                pass
        return ""

    @staticmethod
    def _normalize_backbone_search_query_list(value: Any) -> list[str]:
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, (list, tuple)):
            queries: list[str] = []
            for item in value:
                queries.extend(RayPPOTrainer._normalize_backbone_search_query_list(item))
            return queries
        return []

    @staticmethod
    def _strip_search_list_item_marker(line: str) -> str:
        text = line.strip().rstrip(",").strip()
        text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text)
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()
        return text

    @staticmethod
    def _parse_backbone_search_queries(search_block: Any) -> list[str]:
        text = str(search_block or "").strip()
        if not text:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            queries = RayPPOTrainer._normalize_backbone_search_query_list(parsed)
            if queries:
                return queries

        lines = [RayPPOTrainer._strip_search_list_item_marker(line) for line in text.splitlines()]
        line_queries = [line for line in lines if line]
        if len(line_queries) > 1:
            return line_queries
        return [text]

    @staticmethod
    def _escape_xml_text(text: Any) -> str:
        return str(text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def _normalize_messages(raw_prompt: Any) -> list[dict[str, str]]:
        if isinstance(raw_prompt, (list, tuple)):
            out: list[dict[str, str]] = []
            for msg in raw_prompt:
                if isinstance(msg, dict):
                    out.append(
                        {
                            "role": str(msg.get("role", "user")),
                            "content": str(msg.get("content", "")),
                        }
                    )
                else:
                    out.append({"role": "user", "content": str(msg)})
            return out
        if isinstance(raw_prompt, dict):
            return [{"role": str(raw_prompt.get("role", "user")), "content": str(raw_prompt.get("content", ""))}]
        return [{"role": "user", "content": str(raw_prompt)}]

    def _extract_tool_evidence_from_policy_prompt(self, raw_prompt: Any) -> str:
        if isinstance(raw_prompt, (list, tuple)):
            messages = list(raw_prompt)
        elif isinstance(raw_prompt, dict):
            messages = [raw_prompt]
        else:
            messages = []

        chunks: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                c = content.strip()
                if c:
                    chunks.append(c)
                continue
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        t = str(item.get("text", "")).strip()
                        if t:
                            text_parts.append(t)
                if text_parts:
                    chunks.append("\n".join(text_parts))
        return "\n\n".join(chunks).strip()

    @staticmethod
    def _extract_xml_tag_contents(text: str, tag: str) -> list[str]:
        return re.findall(rf"<{tag}>(.*?)</{tag}>", text or "", flags=re.DOTALL | re.IGNORECASE)

    @staticmethod
    def _has_strict_policy_answer_evidence(
        text: str,
        *,
        max_answer_chars: int = 256,
        max_evidence_chars: int = 768,
    ) -> bool:
        text = text or ""
        answers = RayPPOTrainer._extract_xml_tag_contents(text, "answer")
        evidences = RayPPOTrainer._extract_xml_tag_contents(text, "evidence")
        if len(answers) != 1 or len(evidences) != 1:
            return False
        answer = answers[0].strip()
        evidence = evidences[0].strip()
        if not answer or not evidence:
            return False
        if len(answer) > max_answer_chars or len(evidence) > max_evidence_chars:
            return False

        lower_text = text.lower()
        answer_open = lower_text.find("<answer>")
        answer_close = lower_text.find("</answer>", answer_open)
        evidence_open = lower_text.find("<evidence>", answer_close)
        evidence_close = lower_text.find("</evidence>", evidence_open)
        if min(answer_open, answer_close, evidence_open, evidence_close) < 0:
            return False
        if not (answer_open < answer_close < evidence_open < evidence_close):
            return False
        trailing = text[evidence_close + len("</evidence>") :].strip()
        if trailing:
            return False
        if "<search>" in lower_text[answer_open:]:
            return False
        return True

    def _extract_policy_answer_evidence_from_decoded_response(self, response_text: str) -> str:
        # Match reward-side parsing by extracting the final <answer>...</answer>
        # <evidence>...</evidence> pair from the decoded assistant response text.
        merged = (response_text or "").strip()
        if not merged:
            return ""
        pair_pattern = re.compile(
            r"(<answer>.*?</answer>\s*<evidence>.*?</evidence>)",
            flags=re.DOTALL | re.IGNORECASE,
        )
        pairs = pair_pattern.findall(merged)
        if pairs:
            return pairs[-1].strip()
        return ""

    def _get_policy_continue_gate_config(self) -> dict[str, int | bool]:
        custom_cfg = self.config.actor_rollout_ref.rollout.get("custom", {})
        def _get_int(key: str, default: int) -> int:
            try:
                return int(custom_cfg.get(key, default))
            except (TypeError, ValueError):
                return int(default)
        return {
            "enabled": bool(custom_cfg.get("policy_continue_require_discrete_good", True)),
            "max_output_chars": _get_int("policy_continue_max_output_chars", 1500),
            "max_answer_chars": _get_int("policy_continue_max_answer_chars", 256),
            "max_evidence_chars": _get_int("policy_continue_max_evidence_chars", 768),
        }

    def _build_policy_result_for_backbone(self, search_query: str, policy_answer_evidence: str) -> str:
        if policy_answer_evidence:
            return policy_answer_evidence
        return (
            "<evidence_unavailable>"
            "The policy model did not produce a valid answer/evidence result. "
            f"Request: {self._escape_xml_text(search_query)}"
            "</evidence_unavailable>"
        )

    def _build_parallel_policy_results_for_backbone(self, policy_runs: list[dict[str, Any]]) -> str:
        if not policy_runs:
            return "<evidence_unavailable>No policy search results were produced.</evidence_unavailable>"

        chunks = ["<search_results>"]
        for run in sorted(policy_runs, key=lambda item: int(item.get("query_index", 0))):
            query_index = int(run.get("query_index", 0))
            search_query = str(run.get("search_query", ""))
            policy_answer_evidence = str(run.get("answer_evidence", ""))
            chunks.append(
                f'<result index="{query_index}">\n'
                f"<request>{self._escape_xml_text(search_query)}</request>\n"
                f"{self._build_policy_result_for_backbone(search_query, policy_answer_evidence)}\n"
                "</result>"
            )
        chunks.append("</search_results>")
        return "\n".join(chunks)

    def _build_next_backbone_raw_prompt(
        self,
        previous_backbone_prompt: Any,
        backbone_response: str,
        tool_evidence: str,
    ) -> list[dict[str, str]]:
        messages = self._normalize_messages(previous_backbone_prompt)
        if backbone_response and (not messages or messages[-1].get("content") != backbone_response):
            messages.append({"role": "assistant", "content": backbone_response})

        evidence_text = (tool_evidence or "").strip()
        if evidence_text:
            messages.append(build_next_backbone_message(evidence_text))
        else:
            messages.append(build_policy_failure_backbone_message())
        return messages

    def _run_two_stage_orchestrator_rollout(
        self,
        gen_batch_output: DataProto,
        timing_raw: dict[str, Any],
        curr_step_profile: bool,
        *,
        return_trajectory_batch: bool = False,
    ) -> DataProto | tuple[DataProto, Optional[DataProto]]:
        """Run multi-round backbone-policy orchestration rollout.

        Flow per round:
        1) frozen backbone (single turn) proposes whether tool call is needed
        2) policy tool-agent handles tool calls for selected samples
        3) policy outputs are fed back to next round backbone
        """
        is_validation_rollout = bool(gen_batch_output.meta_info.get("validate", False))
        max_rounds = self._get_max_orchestrator_rounds(is_validation=is_validation_rollout)
        emit_policy_round_train_points = (
            not is_validation_rollout and bool(self.config.algorithm.get("tool_reward_as_grpo_point", False))
        )
        self._latest_rollout_trajectory_batch = None
        remaining_batch = gen_batch_output.select(deepcopy=True)
        remaining_indices = np.arange(len(remaining_batch))
        initial_backbone_prompt_arr = np.empty(len(remaining_batch), dtype=object)
        for i in range(len(remaining_batch)):
            initial_backbone_prompt_arr[i] = self._ensure_backbone_final_answer_instruction(
                remaining_batch.non_tensor_batch["raw_prompt"][i]
            )
        remaining_batch.non_tensor_batch["raw_prompt"] = initial_backbone_prompt_arr

        initial_trace_ids = self._get_orchestrator_trace_ids(remaining_batch)
        chain_events_by_trace: dict[str, list[dict[str, Any]]] = {tid: [] for tid in initial_trace_ids}
        policy_request_ids_by_trace: dict[str, list[str]] = {tid: [] for tid in initial_trace_ids}
        policy_prompt_by_trace: dict[str, str] = {}
        policy_full_trace_output_by_trace: dict[str, str] = {}
        final_binary_score_by_trace: dict[str, float] = {}
        backbone_final_source_by_trace: dict[str, str] = {}
        pair_group_id_by_trace: dict[str, str] = {}
        source_uid_by_trace: dict[str, str] = {}
        initial_uid_arr = remaining_batch.non_tensor_batch.get("uid", None)
        if isinstance(initial_uid_arr, np.ndarray) and initial_uid_arr.shape[0] == len(remaining_batch):
            for i, tid in enumerate(initial_trace_ids):
                source_uid_by_trace[tid] = str(initial_uid_arr[i])

        finished_parts: list[DataProto] = []
        finished_indices: list[np.ndarray] = []
        policy_round_parts: list[DataProto] = []

        def _ensure_agent_name(batch: DataProto, agent_name: str) -> None:
            """Keep non_tensor_batch['agent_name'] aligned with batch size.

            Some rollout paths may drop/keep stale non-tensor fields. This guard ensures
            downstream DataProto.concat/check_consistency always sees a valid column.
            """
            expected_len = len(batch)
            curr = batch.non_tensor_batch.get("agent_name")
            if (
                not isinstance(curr, np.ndarray)
                or curr.shape[0] != expected_len
            ):
                batch.non_tensor_batch["agent_name"] = np.array([agent_name] * expected_len, dtype=object)

        def _generate_with_optional_padding(
            batch: DataProto,
            *,
            size_divisor: int,
            generate_fn,
        ) -> DataProto:
            """Pad dynamic sub-batches before distributed dispatch, then unpad outputs."""
            if size_divisor <= 1:
                return generate_fn(batch)
            padded_batch, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
            padded_output = generate_fn(padded_batch)
            return unpad_dataproto(padded_output, pad_size=pad_size)

        for round_idx in range(max_rounds):
            if len(remaining_batch) == 0:
                break

            backbone_batch = remaining_batch.select(deepcopy=True)
            backbone_trace_ids = self._get_orchestrator_trace_ids(backbone_batch)
            backbone_questions = [
                self._extract_last_user_content(backbone_batch.non_tensor_batch["raw_prompt"][i])
                for i in range(len(backbone_batch))
            ]
            backbone_raw_prompts = [deepcopy(backbone_batch.non_tensor_batch["raw_prompt"][i]) for i in range(len(backbone_batch))]
            backbone_input_source = "initial" if round_idx == 0 else "policy_to_backbone"
            self._append_io_trace(
                "orchestrator.backbone_input",
                {
                    "round": round_idx,
                    "input_source": backbone_input_source,
                    "num_samples": len(backbone_batch),
                    "trace_ids_preview": backbone_trace_ids[: self.io_trace_max_samples],
                    "question_preview": backbone_questions[: self.io_trace_max_samples],
                    "raw_prompt_preview": self._extract_raw_prompt_preview(backbone_batch),
                },
            )
            if self.io_trace_record_sample_chain:
                for i, trace_id in enumerate(backbone_trace_ids):
                    chain_events_by_trace.setdefault(trace_id, []).append(
                        {
                            "round": round_idx,
                            "stage": "backbone_input",
                            "trace_id": trace_id,
                            "input_source": backbone_input_source,
                            "question": backbone_questions[i],
                            "raw_prompt": self._to_jsonable_log_value(backbone_raw_prompts[i]),
                        }
                    )
                    self._append_io_trace(
                        "orchestrator.sample_backbone_input",
                        {
                            "round": round_idx,
                            "trace_id": trace_id,
                            "input_source": backbone_input_source,
                            "question": backbone_questions[i],
                            "raw_prompt": self._to_jsonable_log_value(backbone_raw_prompts[i]),
                        },
                    )
            backbone_batch.non_tensor_batch["agent_name"] = np.array(
                ["single_turn_agent"] * len(backbone_batch), dtype=object
            )
            if self.backbone_async_rollout_manager is not None:
                if curr_step_profile:
                    self.backbone_async_rollout_manager.start_profile()
                backbone_output = _generate_with_optional_padding(
                    backbone_batch,
                    size_divisor=len(self.backbone_async_rollout_manager.agent_loop_workers),
                    generate_fn=self.backbone_async_rollout_manager.generate_sequences,
                )
                if curr_step_profile:
                    self.backbone_async_rollout_manager.stop_profile()
            else:
                # API-backed backbone path can call worker group directly without a second rollout server.
                backbone_output = _generate_with_optional_padding(
                    backbone_batch,
                    size_divisor=self.backbone_rollout_wg.world_size,
                    generate_fn=self.backbone_rollout_wg.generate_sequences,
                )
            # Keep one stable chain id per sample across orchestrator rounds.
            backbone_output.non_tensor_batch["orchestrator_trace_id"] = np.array(backbone_trace_ids, dtype=object)
            _ensure_agent_name(backbone_output, "single_turn_agent")

            backbone_timing = backbone_output.meta_info.get("timing", {})
            self._accumulate_round_timing(timing_raw, backbone_timing, prefix=f"backbone_round_{round_idx}")
            backbone_output.meta_info.pop("timing", None)

            backbone_output_trace_ids = self._get_orchestrator_trace_ids(backbone_output)
            backbone_responses_all = self._decode_batch_responses(backbone_output)
            max_backbone_search_queries = self._get_max_backbone_search_queries(
                is_validation=is_validation_rollout
            )
            backbone_search_blocks_all: list[str] = []
            backbone_search_queries_all: list[list[str]] = []
            backbone_search_query_truncated_counts: list[int] = []
            for response_text in backbone_responses_all:
                search_block = self._extract_backbone_search_query(response_text)
                search_queries = self._parse_backbone_search_queries(search_block)
                truncated_count = 0
                if (
                    search_queries
                    and max_backbone_search_queries > 0
                    and len(search_queries) > max_backbone_search_queries
                ):
                    truncated_count = len(search_queries) - max_backbone_search_queries
                    search_queries = search_queries[:max_backbone_search_queries]
                backbone_search_blocks_all.append(search_block)
                backbone_search_queries_all.append(search_queries)
                backbone_search_query_truncated_counts.append(truncated_count)

            tool_call_mask = self._infer_tool_call_mask(backbone_output)
            no_tool_call_mask = ~tool_call_mask
            backbone_token_usages = [
                self._build_backbone_deepseek_token_usage(
                    backbone_output,
                    i,
                    round_idx=round_idx,
                    stage="backbone_output",
                    trace_id=backbone_output_trace_ids[i],
                )
                for i in range(len(backbone_output))
            ]
            self._append_io_trace(
                "orchestrator.backbone_output",
                {
                    "round": round_idx,
                    "num_samples": len(backbone_output),
                    "tool_call_count": int(tool_call_mask.sum()),
                    "trace_ids_preview": backbone_output_trace_ids[: self.io_trace_max_samples],
                    "response_preview": self._extract_response_preview(backbone_output),
                    "search_query_counts_preview": [
                        len(queries) for queries in backbone_search_queries_all[: self.io_trace_max_samples]
                    ],
                    "deepseek_token_usage_preview": backbone_token_usages[: self.io_trace_max_samples],
                },
            )
            for i, trace_id in enumerate(backbone_output_trace_ids):
                chain_events_by_trace.setdefault(trace_id, []).append(
                    {
                        "round": round_idx,
                        "stage": "backbone_output",
                        "trace_id": trace_id,
                        "input_source": backbone_input_source,
                        "raw_prompt": self._to_jsonable_log_value(backbone_raw_prompts[i])
                        if i < len(backbone_raw_prompts)
                        else None,
                        "question": backbone_questions[i] if i < len(backbone_questions) else "",
                        "response": backbone_responses_all[i],
                        "has_tool_call": bool(tool_call_mask[i]),
                        "search_query": backbone_search_blocks_all[i],
                        "search_queries": backbone_search_queries_all[i],
                        "search_query_count": len(backbone_search_queries_all[i]),
                        "search_query_truncated_count": backbone_search_query_truncated_counts[i],
                        "backbone_token_usage": backbone_token_usages[i],
                    }
                )
                if self.io_trace_record_sample_chain:
                    self._append_io_trace(
                        "orchestrator.sample_backbone_output",
                        {
                            "round": round_idx,
                            "trace_id": trace_id,
                            "has_tool_call": bool(tool_call_mask[i]),
                            "response": backbone_responses_all[i],
                            "search_query": backbone_search_blocks_all[i],
                            "search_queries": backbone_search_queries_all[i],
                            "backbone_token_usage": backbone_token_usages[i],
                        },
                    )

            if no_tool_call_mask.any():
                for i, has_no_tool in enumerate(no_tool_call_mask):
                    if has_no_tool:
                        backbone_final_source_by_trace[backbone_output_trace_ids[i]] = "backbone"
                finished_parts.append(backbone_output[no_tool_call_mask])
                finished_indices.append(remaining_indices[no_tool_call_mask])

            if not tool_call_mask.any():
                remaining_indices = np.array([], dtype=np.int64)
                break

            parent_policy_batch = backbone_output[tool_call_mask]
            parent_backbone_context_prompts = [
                deepcopy(parent_policy_batch.non_tensor_batch["raw_prompt"][i]) for i in range(len(parent_policy_batch))
            ]
            parent_backbone_responses = self._decode_batch_responses(parent_policy_batch)
            parent_positions = [i for i, has_tool in enumerate(tool_call_mask) if has_tool]
            parent_trace_ids = [backbone_output_trace_ids[i] for i in parent_positions]
            parent_search_blocks = [backbone_search_blocks_all[i] for i in parent_positions]
            parent_search_queries = [list(backbone_search_queries_all[i]) for i in parent_positions]
            parent_indices = remaining_indices[tool_call_mask]

            policy_rollout_n = self._get_policy_rollout_n(is_validation=is_validation_rollout)
            policy_parent_refs: list[int] = []
            policy_query_indices: list[int] = []
            policy_query_counts: list[int] = []
            policy_source_indices_list: list[int] = []
            policy_trace_ids: list[str] = []
            policy_composite_trace_ids: list[str] = []
            policy_search_queries: list[str] = []
            policy_search_blocks: list[str] = []

            for parent_i in range(len(parent_policy_batch)):
                queries = [str(query).strip() for query in parent_search_queries[parent_i] if str(query).strip()]
                if not queries:
                    sanitized_prompt = self._sanitize_policy_raw_prompt(parent_backbone_context_prompts[parent_i])
                    fallback_question = ""
                    for msg in reversed(self._normalize_messages(sanitized_prompt)):
                        if msg.get("role") == "user":
                            fallback_question = str(msg.get("content", "")).strip()
                            if fallback_question:
                                break
                    fallback_query = (
                        str(parent_search_blocks[parent_i] or "").strip()
                        or fallback_question
                        or "Find evidence relevant to the current question."
                    )
                    queries = [fallback_query]
                    parent_search_queries[parent_i] = queries
                query_count = len(queries)
                parent_trace = parent_trace_ids[parent_i]
                for branch_i in range(policy_rollout_n):
                    if policy_rollout_n > 1:
                        composite_trace_id = f"{parent_trace}::r{round_idx}b{branch_i}"
                    else:
                        composite_trace_id = parent_trace
                    if composite_trace_id not in chain_events_by_trace:
                        chain_events_by_trace[composite_trace_id] = deepcopy(
                            chain_events_by_trace.get(parent_trace, [])
                        )
                    if composite_trace_id not in policy_request_ids_by_trace:
                        policy_request_ids_by_trace[composite_trace_id] = deepcopy(
                            policy_request_ids_by_trace.get(parent_trace, [])
                        )
                    if parent_trace in source_uid_by_trace:
                        source_uid_by_trace[composite_trace_id] = source_uid_by_trace[parent_trace]

                    for query_index, query in enumerate(queries):
                        policy_trace_id = (
                            f"{composite_trace_id}::q{query_index}" if query_count > 1 else composite_trace_id
                        )
                        policy_parent_refs.append(parent_i)
                        policy_query_indices.append(query_index)
                        policy_query_counts.append(query_count)
                        policy_source_indices_list.append(int(parent_indices[parent_i]))
                        policy_trace_ids.append(policy_trace_id)
                        policy_composite_trace_ids.append(composite_trace_id)
                        policy_search_queries.append(query)
                        policy_search_blocks.append(parent_search_blocks[parent_i])

                        pair_group_id = f"{parent_trace}::round{round_idx}"
                        if query_count > 1:
                            pair_group_id = f"{pair_group_id}::q{query_index}"
                        pair_group_id_by_trace[policy_trace_id] = pair_group_id
                        if parent_trace in source_uid_by_trace:
                            source_uid_by_trace[policy_trace_id] = source_uid_by_trace[parent_trace]
                        if policy_trace_id not in chain_events_by_trace:
                            chain_events_by_trace[policy_trace_id] = deepcopy(
                                chain_events_by_trace.get(parent_trace, [])
                            )
                        if policy_trace_id not in policy_request_ids_by_trace:
                            policy_request_ids_by_trace[policy_trace_id] = deepcopy(
                                policy_request_ids_by_trace.get(parent_trace, [])
                            )

            if not policy_parent_refs:
                remaining_indices = np.array([], dtype=np.int64)
                break

            parent_ref = np.array(policy_parent_refs, dtype=np.int64)
            policy_source_indices = np.array(policy_source_indices_list, dtype=np.int64)
            policy_batch = parent_policy_batch[parent_ref]

            backbone_context_prompts = [
                parent_backbone_context_prompts[int(parent_ref[i])] for i in range(len(policy_batch))
            ]
            backbone_responses = [parent_backbone_responses[int(parent_ref[i])] for i in range(len(policy_batch))]

            policy_prompts: list[list[dict[str, str]]] = []
            for i in range(len(policy_batch)):
                query = policy_search_queries[i]
                policy_prompts.append([{"role": "user", "content": query}])
                policy_prompt_by_trace[policy_trace_ids[i]] = query
                chain_events_by_trace.setdefault(policy_trace_ids[i], []).append(
                    {
                        "round": round_idx,
                        "stage": "policy_input",
                        "query": query,
                        "query_index": int(policy_query_indices[i]),
                        "query_count": int(policy_query_counts[i]),
                        "parallel_group_trace_id": policy_composite_trace_ids[i],
                        "backbone_response": backbone_responses[i],
                    }
                )
            policy_prompt_arr = np.empty(len(policy_prompts), dtype=object)
            for i, prompt in enumerate(policy_prompts):
                policy_prompt_arr[i] = prompt
            policy_batch.non_tensor_batch["raw_prompt"] = policy_prompt_arr

            tools_kwargs_arr = np.empty(len(policy_batch), dtype=object)
            existing_tools_kwargs = policy_batch.non_tensor_batch.get("tools_kwargs", None)
            for i in range(len(policy_batch)):
                base_kwargs: dict[str, Any] = {}
                if isinstance(existing_tools_kwargs, np.ndarray) and existing_tools_kwargs.shape[0] == len(policy_batch):
                    if isinstance(existing_tools_kwargs[i], dict):
                        base_kwargs = dict(existing_tools_kwargs[i])
                base_kwargs["_orchestrator_trace"] = {
                    "trace_id": policy_trace_ids[i],
                    "parallel_group_trace_id": policy_composite_trace_ids[i],
                    "query_index": int(policy_query_indices[i]),
                    "query_count": int(policy_query_counts[i]),
                    "round": round_idx,
                    "global_step": int(getattr(self, "global_steps", -1)),
                }
                tools_kwargs_arr[i] = base_kwargs
            policy_batch.non_tensor_batch["tools_kwargs"] = tools_kwargs_arr

            self._append_io_trace(
                "orchestrator.policy_input",
                {
                    "round": round_idx,
                    "num_samples": len(policy_batch),
                    "trace_ids_preview": policy_trace_ids[: self.io_trace_max_samples],
                    "parallel_group_trace_ids_preview": policy_composite_trace_ids[: self.io_trace_max_samples],
                    "query_indices_preview": policy_query_indices[: self.io_trace_max_samples],
                    "query_counts_preview": policy_query_counts[: self.io_trace_max_samples],
                    "raw_prompt_preview": self._extract_raw_prompt_preview(policy_batch),
                    "backbone_response_preview": self._extract_response_preview(policy_batch),
                },
            )
            if self.io_trace_record_sample_chain:
                for i, trace_id in enumerate(policy_trace_ids):
                    self._append_io_trace(
                        "orchestrator.sample_policy_input",
                        {
                            "round": round_idx,
                            "trace_id": trace_id,
                            "parallel_group_trace_id": policy_composite_trace_ids[i],
                            "query_index": int(policy_query_indices[i]),
                            "query_count": int(policy_query_counts[i]),
                            "query": policy_prompts[i][0].get("content", ""),
                        },
                    )
            policy_batch.non_tensor_batch["agent_name"] = np.array(["tool_agent"] * len(policy_batch), dtype=object)

            if curr_step_profile:
                self.async_rollout_manager.start_profile()
            policy_output = _generate_with_optional_padding(
                policy_batch,
                size_divisor=len(self.async_rollout_manager.agent_loop_workers),
                generate_fn=self.async_rollout_manager.generate_sequences,
            )
            # Enforce trace-id inheritance from backbone->policy for robust chain reconstruction.
            policy_output.non_tensor_batch["orchestrator_trace_id"] = np.array(policy_trace_ids, dtype=object)
            policy_output.non_tensor_batch["orchestrator_source_index"] = np.array(policy_source_indices, dtype=np.int64)
            policy_output.non_tensor_batch["parallel_group_trace_id"] = np.array(
                policy_composite_trace_ids, dtype=object
            )
            policy_output.non_tensor_batch["parallel_query_index"] = np.array(policy_query_indices, dtype=np.int32)
            policy_output.non_tensor_batch["parallel_query_count"] = np.array(policy_query_counts, dtype=np.int32)
            policy_output.non_tensor_batch["backbone_search_block"] = np.array(policy_search_blocks, dtype=object)
            policy_output.non_tensor_batch["backbone_search_queries"] = self._make_object_array(
                [
                    [
                        policy_search_queries[j]
                        for j in range(len(policy_search_queries))
                        if policy_composite_trace_ids[j] == group_id
                    ]
                    for group_id in policy_composite_trace_ids
                ]
            )
            self.checkpoint_manager.sleep_replicas()
            if curr_step_profile:
                self.async_rollout_manager.stop_profile()

            policy_timing = policy_output.meta_info.get("timing", {})
            self._accumulate_round_timing(timing_raw, policy_timing, prefix=f"policy_round_{round_idx}")
            policy_output.meta_info.pop("timing", None)
            _ensure_agent_name(policy_output, "tool_agent")

            policy_output_trace_ids = self._get_orchestrator_trace_ids(policy_output)
            policy_output_responses = self._decode_batch_responses(policy_output)
            policy_round_full_trace_outputs = [
                self._extract_policy_round_full_trace_output_at(policy_output, i, policy_output_responses[i])
                for i in range(len(policy_output))
            ]
            for i, trace_id in enumerate(policy_output_trace_ids):
                policy_full_trace_output_by_trace[trace_id] = policy_round_full_trace_outputs[i]
            policy_request_ids: list[str] = []
            req_arr = policy_output.non_tensor_batch.get("request_id", None)
            for i in range(len(policy_output)):
                req_id = ""
                if isinstance(req_arr, np.ndarray) and req_arr.shape[0] == len(policy_output):
                    req_id = str(req_arr[i])
                policy_request_ids.append(req_id)
                if req_id:
                    policy_request_ids_by_trace.setdefault(policy_output_trace_ids[i], []).append(req_id)
                chain_events_by_trace.setdefault(policy_output_trace_ids[i], []).append(
                    {
                        "round": round_idx,
                        "stage": "policy_output",
                        "request_id": req_id,
                        "query_index": int(policy_query_indices[i]),
                        "query_count": int(policy_query_counts[i]),
                        "parallel_group_trace_id": policy_composite_trace_ids[i],
                        "response": policy_output_responses[i],
                        "full_trace_output": policy_round_full_trace_outputs[i],
                    }
                )

            self._append_io_trace(
                "orchestrator.policy_output",
                {
                    "round": round_idx,
                    "num_samples": len(policy_output),
                    "trace_ids_preview": policy_output_trace_ids[: self.io_trace_max_samples],
                    "parallel_group_trace_ids_preview": policy_composite_trace_ids[: self.io_trace_max_samples],
                    "policy_request_ids_preview": policy_request_ids[: self.io_trace_max_samples],
                    "response_preview": self._extract_response_preview(policy_output),
                    "tool_rewards_preview": list(policy_output.non_tensor_batch.get("tool_rewards", [])[: self.io_trace_max_samples]),
                },
            )
            if self.io_trace_record_sample_chain:
                for i, trace_id in enumerate(policy_output_trace_ids):
                    self._append_io_trace(
                        "orchestrator.sample_policy_output",
                        {
                            "round": round_idx,
                            "trace_id": trace_id,
                            "parallel_group_trace_id": policy_composite_trace_ids[i],
                            "query_index": int(policy_query_indices[i]),
                            "query_count": int(policy_query_counts[i]),
                            "request_id": policy_request_ids[i],
                            "response": policy_output_responses[i],
                        },
                    )

            missing_evidence_mask = np.zeros(len(policy_output), dtype=bool)
            policy_answer_evidences: list[str] = []
            policy_backbone_inputs: list[str] = []
            policy_binary_scores: list[int] = []
            policy_binary_score_values: list[Optional[float]] = []
            policy_binary_details: list[dict[str, Any]] = []
            orchestrator_judge_call_trace_ids: list[str] = []
            for i in range(len(policy_output)):
                policy_answer_evidence = self._extract_policy_answer_evidence_from_decoded_response(
                    policy_output_responses[i]
                )
                policy_answer_evidences.append(policy_answer_evidence)
                if not policy_answer_evidence:
                    missing_evidence_mask[i] = True
                policy_backbone_inputs.append(
                    self._build_policy_result_for_backbone(policy_search_queries[i], policy_answer_evidence)
                )
                policy_question = policy_prompt_by_trace.get(policy_output_trace_ids[i], "")
                if not policy_question:
                    policy_question = self._extract_last_user_content(policy_output.non_tensor_batch["raw_prompt"][i])
                if is_validation_rollout:
                    binary_score_float, binary_details = self._extract_existing_backbone_judge_score_at(
                        policy_output, i
                    )
                    binary_details = binary_details if isinstance(binary_details, dict) else {}
                    if binary_score_float is None:
                        binary_details = {
                            **binary_details,
                            "skipped": True,
                            "skip_reason": "validation_rollout",
                        }
                    orchestrator_judge_called = False
                else:
                    binary_score_float, binary_details, orchestrator_judge_called = (
                        self._judge_policy_output_with_backbone_binary_at(
                            policy_output,
                            i,
                            policy_question=policy_question,
                            final_response_text=policy_output_responses[i],
                        )
                    )
                if orchestrator_judge_called:
                    orchestrator_judge_call_trace_ids.append(policy_output_trace_ids[i])
                if isinstance(binary_details, dict):
                    full_trace_output = binary_details.get("policy_full_trace_output", None)
                    if isinstance(full_trace_output, str) and full_trace_output.strip():
                        policy_round_full_trace_outputs[i] = full_trace_output.strip()
                        policy_full_trace_output_by_trace[policy_output_trace_ids[i]] = full_trace_output.strip()
                        for event in reversed(chain_events_by_trace.get(policy_output_trace_ids[i], [])):
                            if isinstance(event, dict) and event.get("stage") == "policy_output":
                                event["full_trace_output"] = full_trace_output.strip()
                                break
                binary_score = int(float(binary_score_float) > 0.5) if binary_score_float is not None else 0
                policy_binary_details.append(binary_details if isinstance(binary_details, dict) else {})
                policy_binary_scores.append(binary_score)
                policy_binary_score_values.append(float(binary_score) if binary_score_float is not None else None)
                if binary_score_float is not None:
                    final_binary_score_by_trace[policy_output_trace_ids[i]] = float(binary_score)
            if orchestrator_judge_call_trace_ids:
                self._append_io_trace(
                    "orchestrator.backbone_binary_judge",
                    {
                        "round": round_idx,
                        "num_judged": len(orchestrator_judge_call_trace_ids),
                        "trace_ids_preview": orchestrator_judge_call_trace_ids[: self.io_trace_max_samples],
                    },
                )
            policy_binary_scores_arr = np.array(policy_binary_scores, dtype=np.int32)
            policy_output.non_tensor_batch["backbone_binary_score"] = policy_binary_scores_arr
            policy_output.non_tensor_batch["final_backbone_binary_score"] = np.array(
                policy_binary_score_values, dtype=object
            )
            policy_output.non_tensor_batch["final_backbone_binary_details"] = self._make_object_array(
                policy_binary_details
            )
            policy_output.non_tensor_batch["response_source"] = np.array(["policy"] * len(policy_output), dtype=object)
            policy_output.non_tensor_batch["policy_backbone_input"] = np.array(policy_backbone_inputs, dtype=object)
            policy_output.non_tensor_batch["pair_group_id"] = np.array(
                [pair_group_id_by_trace.get(trace_id, "") for trace_id in policy_output_trace_ids], dtype=object
            )
            policy_output.non_tensor_batch["source_uid"] = np.array(
                [source_uid_by_trace.get(trace_id, "") for trace_id in policy_output_trace_ids], dtype=object
            )
            policy_output.non_tensor_batch["policy_prompt"] = np.array(
                [policy_prompt_by_trace.get(trace_id, "") for trace_id in policy_output_trace_ids], dtype=object
            )
            policy_output.non_tensor_batch["policy_full_trace_output"] = np.array(
                policy_round_full_trace_outputs, dtype=object
            )
            policy_output.non_tensor_batch["policy_request_ids"] = self._make_object_array(
                [[req_id] if req_id else [] for req_id in policy_request_ids]
            )
            policy_output.non_tensor_batch["orchestrator_chain"] = self._make_object_array(
                [deepcopy(chain_events_by_trace.get(trace_id, [])) for trace_id in policy_output_trace_ids]
            )
            policy_output.non_tensor_batch["orchestrator_round_count"] = np.array(
                [round_idx + 1] * len(policy_output), dtype=np.int32
            )

            if emit_policy_round_train_points:
                policy_round_parts.append(policy_output.select(deepcopy=True))

            grouped_policy_indices: dict[str, list[int]] = {}
            group_order: list[str] = []
            for i, group_trace_id in enumerate(policy_composite_trace_ids):
                if group_trace_id not in grouped_policy_indices:
                    grouped_policy_indices[group_trace_id] = []
                    group_order.append(group_trace_id)
                grouped_policy_indices[group_trace_id].append(i)

            continue_indices: list[int] = []
            continue_source_indices: list[int] = []
            continue_trace_ids: list[str] = []
            continue_prompts: list[list[dict[str, str]]] = []
            terminated_indices: list[int] = []
            group_would_continue_flags: dict[str, bool] = {}
            group_binary_continue_flags: dict[str, bool] = {}
            group_discrete_continue_flags: dict[str, bool] = {}
            group_continue_flags: dict[str, bool] = {}
            policy_continue_checks_by_idx: dict[int, dict[str, bool]] = {}
            policy_continue_gate = self._get_policy_continue_gate_config()
            continue_gate_enabled = bool(policy_continue_gate["enabled"])
            continue_max_output_chars = int(policy_continue_gate["max_output_chars"])
            continue_max_answer_chars = int(policy_continue_gate["max_answer_chars"])
            continue_max_evidence_chars = int(policy_continue_gate["max_evidence_chars"])

            for group_trace_id in group_order:
                member_indices = grouped_policy_indices[group_trace_id]
                member_indices = sorted(member_indices, key=lambda idx: int(policy_query_indices[idx]))
                expected_query_count = max(int(policy_query_counts[idx]) for idx in member_indices)
                seen_query_indices = {int(policy_query_indices[idx]) for idx in member_indices}
                has_all_queries = seen_query_indices == set(range(expected_query_count))
                member_continue_checks: dict[int, dict[str, bool]] = {}
                for idx in member_indices:
                    details = policy_binary_details[idx] if idx < len(policy_binary_details) else {}
                    details = details if isinstance(details, dict) else {}
                    retrieval_effective = float(details.get("retrieval_effective", policy_binary_scores[idx])) > 0.5
                    summary_reasonable = float(details.get("summary_reasonable", policy_binary_scores[idx])) > 0.5
                    policy_full_text = policy_round_full_trace_outputs[idx] if idx < len(policy_round_full_trace_outputs) else ""
                    policy_answer_evidence = policy_answer_evidences[idx]
                    strict_format_ok = self._has_strict_policy_answer_evidence(
                        policy_answer_evidence,
                        max_answer_chars=continue_max_answer_chars,
                        max_evidence_chars=continue_max_evidence_chars,
                    )
                    output_too_long = len(policy_full_text or policy_answer_evidence or "") > continue_max_output_chars
                    discrete_good = (
                        retrieval_effective
                        and summary_reasonable
                        and strict_format_ok
                        and not output_too_long
                    )
                    member_continue_checks[idx] = {
                        "retrieval_effective": bool(retrieval_effective),
                        "summary_reasonable": bool(summary_reasonable),
                        "strict_format_ok": bool(strict_format_ok),
                        "output_too_long": bool(output_too_long),
                        "discrete_good": bool(discrete_good),
                    }
                    policy_continue_checks_by_idx[idx] = member_continue_checks[idx]
                group_binary_continue = has_all_queries and all(
                    int(policy_binary_scores_arr[idx]) > 0 for idx in member_indices
                )
                group_discrete_continue = has_all_queries and all(
                    member_continue_checks[idx]["discrete_good"] for idx in member_indices
                )
                group_would_continue = group_discrete_continue if continue_gate_enabled else group_binary_continue
                group_continue = has_all_queries and (group_would_continue or is_validation_rollout)
                group_would_continue_flags[group_trace_id] = bool(group_would_continue)
                group_binary_continue_flags[group_trace_id] = bool(group_binary_continue)
                group_discrete_continue_flags[group_trace_id] = bool(group_discrete_continue)
                group_continue_flags[group_trace_id] = bool(group_continue)

                policy_runs_for_backbone = [
                    {
                        "query_index": int(policy_query_indices[idx]),
                        "query_count": int(policy_query_counts[idx]),
                        "search_query": policy_search_queries[idx],
                        "answer_evidence": policy_answer_evidences[idx],
                        "policy_trace_id": policy_output_trace_ids[idx],
                        "request_id": policy_request_ids[idx],
                        "binary_score": float(policy_binary_scores[idx]),
                        "has_answer_evidence": not bool(missing_evidence_mask[idx]),
                        "continue_gate": member_continue_checks.get(idx, {}),
                    }
                    for idx in member_indices
                ]
                combined_backbone_input = self._build_parallel_policy_results_for_backbone(
                    policy_runs_for_backbone
                )
                representative_idx = member_indices[0]
                parent_trace = parent_trace_ids[int(parent_ref[representative_idx])]
                if group_trace_id not in chain_events_by_trace:
                    chain_events_by_trace[group_trace_id] = deepcopy(chain_events_by_trace.get(parent_trace, []))
                chain_events_by_trace.setdefault(group_trace_id, []).append(
                    {
                        "round": round_idx,
                        "stage": "policy_outputs",
                        "parallel_query_count": expected_query_count,
                        "has_all_queries": bool(has_all_queries),
                        "all_binary_score_one": bool(group_binary_continue),
                        "all_discrete_good": bool(group_discrete_continue),
                        "continue_gate_enabled": bool(continue_gate_enabled),
                        "results": policy_runs_for_backbone,
                        "backbone_input": combined_backbone_input,
                    }
                )
                chain_events_by_trace.setdefault(group_trace_id, []).append(
                    {
                        "round": round_idx,
                        "stage": "policy_group_decision",
                        "parallel_query_count": expected_query_count,
                        "has_all_queries": bool(has_all_queries),
                        "all_binary_score_one": bool(group_binary_continue),
                        "all_discrete_good": bool(group_discrete_continue),
                        "continue_gate_enabled": bool(continue_gate_enabled),
                        "continue_to_backbone": bool(group_continue),
                        "terminated_after_policy": not bool(group_continue),
                        "validation_forced_continue": bool(is_validation_rollout and not group_would_continue),
                    }
                )
                existing_group_request_ids = list(policy_request_ids_by_trace.get(group_trace_id, []))
                current_group_request_ids = [
                    req_id
                    for idx in member_indices
                    for req_id in ([policy_request_ids[idx]] if policy_request_ids[idx] else [])
                ]
                policy_request_ids_by_trace[group_trace_id] = existing_group_request_ids + [
                    req_id for req_id in current_group_request_ids if req_id not in existing_group_request_ids
                ]
                policy_prompt_by_trace[group_trace_id] = policy_search_blocks[representative_idx] or json.dumps(
                    [policy_search_queries[idx] for idx in member_indices], ensure_ascii=False
                )
                policy_full_trace_output_by_trace[group_trace_id] = combined_backbone_input
                final_binary_score_by_trace[group_trace_id] = float(1.0 if group_would_continue else 0.0)

                if group_continue:
                    continue_indices.append(representative_idx)
                    continue_source_indices.append(int(policy_source_indices[representative_idx]))
                    continue_trace_ids.append(group_trace_id)
                    continue_prompts.append(
                        self._build_next_backbone_raw_prompt(
                            previous_backbone_prompt=backbone_context_prompts[representative_idx],
                            backbone_response=backbone_responses[representative_idx],
                            tool_evidence=combined_backbone_input,
                        )
                    )
                else:
                    terminated_indices.extend(member_indices)

            for i, trace_id in enumerate(policy_output_trace_ids):
                group_trace_id = policy_composite_trace_ids[i]
                group_continue = group_continue_flags.get(group_trace_id, False)
                group_would_continue = group_would_continue_flags.get(group_trace_id, False)
                group_binary_continue = group_binary_continue_flags.get(group_trace_id, False)
                group_discrete_continue = group_discrete_continue_flags.get(group_trace_id, False)
                current_gate = policy_continue_checks_by_idx.get(i, {})
                chain_events_by_trace.setdefault(trace_id, []).append(
                    {
                        "round": round_idx,
                        "stage": "policy_decision",
                        "binary_score": float(policy_binary_scores[i]),
                        "has_answer_evidence": not bool(missing_evidence_mask[i]),
                        "parallel_group_trace_id": group_trace_id,
                        "group_all_binary_score_one": bool(group_binary_continue),
                        "group_all_discrete_good": bool(group_discrete_continue),
                        "group_would_continue": bool(group_would_continue),
                        "continue_gate": current_gate,
                        "continue_gate_enabled": bool(continue_gate_enabled),
                        "continue_to_backbone": bool(group_continue and policy_query_counts[i] == 1),
                        "combined_with_parallel_queries": bool(group_continue and policy_query_counts[i] > 1),
                        "terminated_after_policy": not bool(group_continue),
                        "validation_forced_continue": bool(is_validation_rollout and not group_would_continue),
                    }
                )
            missing_evidence_trace_ids = [
                policy_output_trace_ids[i] for i in range(len(policy_output)) if missing_evidence_mask[i]
            ]
            if missing_evidence_trace_ids:
                self._append_io_trace(
                    "orchestrator.validation_missing_policy_evidence"
                    if is_validation_rollout
                    else "orchestrator.policy_missing_answer_evidence",
                    {
                        "round": round_idx,
                        "num_samples": len(policy_output),
                        "missing_count": len(missing_evidence_trace_ids),
                        "trace_ids_preview": missing_evidence_trace_ids[: self.io_trace_max_samples],
                    },
                )
            self._append_io_trace(
                "orchestrator.policy_binary_decision",
                {
                    "round": round_idx,
                    "num_samples": len(policy_output),
                    "parallel_group_count": len(group_order),
                    "continue_group_count": len(continue_indices),
                    "terminate_group_count": len(group_order) - len(continue_indices),
                    "continue_count": len(continue_indices),
                    "terminate_count": len(terminated_indices),
                    "would_continue_count": sum(1 for flag in group_would_continue_flags.values() if flag),
                    "would_terminate_count": sum(1 for flag in group_would_continue_flags.values() if not flag),
                    "continue_gate_enabled": bool(continue_gate_enabled),
                    "validation_no_prune": bool(is_validation_rollout),
                    "binary_scores_preview": policy_binary_scores[: self.io_trace_max_samples],
                },
            )
            if terminated_indices:
                terminated_indices_arr = np.array(terminated_indices, dtype=np.int64)
                for idx in terminated_indices:
                    backbone_final_source_by_trace[policy_output_trace_ids[idx]] = "policy"
                finished_parts.append(policy_output[terminated_indices_arr])
                finished_indices.append(policy_source_indices[terminated_indices_arr])

            if not continue_indices:
                remaining_indices = np.array([], dtype=np.int64)
                break

            continue_indices_arr = np.array(continue_indices, dtype=np.int64)
            remaining_batch = policy_output[continue_indices_arr]
            remaining_indices = np.array(continue_source_indices, dtype=np.int64)
            next_backbone_prompt_arr = np.empty(len(continue_prompts), dtype=object)
            for i, prompt in enumerate(continue_prompts):
                next_backbone_prompt_arr[i] = prompt
            remaining_batch.non_tensor_batch["raw_prompt"] = next_backbone_prompt_arr
            remaining_batch.non_tensor_batch["orchestrator_trace_id"] = np.array(continue_trace_ids, dtype=object)
            remaining_batch.non_tensor_batch["orchestrator_source_index"] = remaining_indices

            # Reached the orchestration cap: finalize remaining policy trajectories.
            if round_idx == max_rounds - 1:
                final_backbone_batch = remaining_batch.select(deepcopy=True)
                forced_final_prompts: list[list[dict[str, str]]] = []
                for i in range(len(final_backbone_batch)):
                    forced_final_prompts.append(
                        self._ensure_backbone_final_answer_instruction(
                            final_backbone_batch.non_tensor_batch["raw_prompt"][i], force_final_answer=True
                        )
                    )
                forced_final_prompt_arr = np.empty(len(forced_final_prompts), dtype=object)
                for i, prompt in enumerate(forced_final_prompts):
                    forced_final_prompt_arr[i] = prompt
                final_backbone_batch.non_tensor_batch["raw_prompt"] = forced_final_prompt_arr
                final_trace_ids = self._get_orchestrator_trace_ids(final_backbone_batch)
                final_questions = [
                    self._extract_last_user_content(final_backbone_batch.non_tensor_batch["raw_prompt"][i])
                    for i in range(len(final_backbone_batch))
                ]
                final_raw_prompts = [
                    deepcopy(final_backbone_batch.non_tensor_batch["raw_prompt"][i])
                    for i in range(len(final_backbone_batch))
                ]
                self._append_io_trace(
                    "orchestrator.backbone_final_input",
                    {
                        "round": round_idx,
                        "input_source": "policy_to_backbone_final",
                        "num_samples": len(final_backbone_batch),
                        "trace_ids_preview": final_trace_ids[: self.io_trace_max_samples],
                        "question_preview": final_questions[: self.io_trace_max_samples],
                        "raw_prompt_preview": self._extract_raw_prompt_preview(final_backbone_batch),
                    },
                )
                if self.io_trace_record_sample_chain:
                    for i, trace_id in enumerate(final_trace_ids):
                        chain_events_by_trace.setdefault(trace_id, []).append(
                            {
                                "round": round_idx,
                                "stage": "backbone_final_input",
                                "trace_id": trace_id,
                                "input_source": "policy_to_backbone_final",
                                "question": final_questions[i],
                                "raw_prompt": self._to_jsonable_log_value(final_raw_prompts[i]),
                            }
                        )
                        self._append_io_trace(
                            "orchestrator.sample_backbone_final_input",
                            {
                                "round": round_idx,
                                "trace_id": trace_id,
                                "input_source": "policy_to_backbone_final",
                                "question": final_questions[i],
                                "raw_prompt": self._to_jsonable_log_value(final_raw_prompts[i]),
                            },
                        )

                final_backbone_batch.non_tensor_batch["agent_name"] = np.array(
                    ["single_turn_agent"] * len(final_backbone_batch), dtype=object
                )
                if self.backbone_async_rollout_manager is not None:
                    if curr_step_profile:
                        self.backbone_async_rollout_manager.start_profile()
                    final_backbone_output = _generate_with_optional_padding(
                        final_backbone_batch,
                        size_divisor=len(self.backbone_async_rollout_manager.agent_loop_workers),
                        generate_fn=self.backbone_async_rollout_manager.generate_sequences,
                    )
                    if curr_step_profile:
                        self.backbone_async_rollout_manager.stop_profile()
                else:
                    final_backbone_output = _generate_with_optional_padding(
                        final_backbone_batch,
                        size_divisor=self.backbone_rollout_wg.world_size,
                        generate_fn=self.backbone_rollout_wg.generate_sequences,
                    )
                final_backbone_output.non_tensor_batch["orchestrator_trace_id"] = np.array(final_trace_ids, dtype=object)
                _ensure_agent_name(final_backbone_output, "single_turn_agent")

                final_backbone_timing = final_backbone_output.meta_info.get("timing", {})
                self._accumulate_round_timing(timing_raw, final_backbone_timing, prefix=f"backbone_final_round_{round_idx}")
                final_backbone_output.meta_info.pop("timing", None)

                final_backbone_out_trace_ids = self._get_orchestrator_trace_ids(final_backbone_output)
                final_backbone_responses = self._decode_batch_responses(final_backbone_output)
                final_backbone_tool_call_mask = self._infer_tool_call_mask(final_backbone_output)
                final_backbone_token_usages = [
                    self._build_backbone_deepseek_token_usage(
                        final_backbone_output,
                        i,
                        round_idx=round_idx,
                        stage="backbone_final_output",
                        trace_id=final_backbone_out_trace_ids[i],
                    )
                    for i in range(len(final_backbone_output))
                ]
                if final_backbone_tool_call_mask.any():
                    self._append_io_trace(
                        "orchestrator.backbone_final_unexpected_tool_call",
                        {
                            "round": round_idx,
                            "count": int(final_backbone_tool_call_mask.sum()),
                            "trace_ids_preview": [
                                final_backbone_out_trace_ids[i]
                                for i, has_tool in enumerate(final_backbone_tool_call_mask)
                                if has_tool
                            ][: self.io_trace_max_samples],
                        },
                    )

                self._append_io_trace(
                    "orchestrator.backbone_final_output",
                    {
                        "round": round_idx,
                        "num_samples": len(final_backbone_output),
                        "trace_ids_preview": final_backbone_out_trace_ids[: self.io_trace_max_samples],
                        "response_preview": final_backbone_responses[: self.io_trace_max_samples],
                        "tool_call_count": int(final_backbone_tool_call_mask.sum()),
                        "deepseek_token_usage_preview": final_backbone_token_usages[: self.io_trace_max_samples],
                    },
                )

                for i, trace_id in enumerate(final_backbone_out_trace_ids):
                    backbone_final_source_by_trace[trace_id] = "backbone"
                    chain_events_by_trace.setdefault(trace_id, []).append(
                        {
                            "round": round_idx,
                            "stage": "backbone_final_output",
                            "trace_id": trace_id,
                            "input_source": "policy_to_backbone_final",
                            "raw_prompt": self._to_jsonable_log_value(final_raw_prompts[i])
                            if i < len(final_raw_prompts)
                            else None,
                            "question": final_questions[i] if i < len(final_questions) else "",
                            "response": final_backbone_responses[i],
                            "has_tool_call": bool(final_backbone_tool_call_mask[i]),
                            "backbone_token_usage": final_backbone_token_usages[i],
                        }
                    )
                    if is_validation_rollout and self.io_trace_record_sample_chain:
                        self._append_io_trace(
                            "orchestrator.sample_backbone_final_output",
                            {
                                "round": round_idx,
                                "trace_id": trace_id,
                                "has_tool_call": bool(final_backbone_tool_call_mask[i]),
                                "response": final_backbone_responses[i],
                                "backbone_token_usage": final_backbone_token_usages[i],
                            },
                        )

                finished_parts.append(final_backbone_output)
                finished_indices.append(remaining_indices)
                remaining_indices = np.array([], dtype=np.int64)

        if not finished_parts:
            if return_trajectory_batch:
                return gen_batch_output, None
            return gen_batch_output

        if len(finished_parts) == 1:
            merged = finished_parts[0]
            current_idx_order = finished_indices[0]
        else:
            merged = self._concat_dataproto_parts_with_seq_padding(finished_parts)
            current_idx_order = np.concatenate(finished_indices, axis=0)

        restore_order = np.argsort(current_idx_order)
        restored = merged[restore_order]
        restored.non_tensor_batch["orchestrator_source_index"] = np.array(current_idx_order[restore_order], dtype=np.int64)
        restored_trace_ids = self._get_orchestrator_trace_ids(restored)
        restored.non_tensor_batch["orchestrator_chain"] = self._make_object_array(
            [chain_events_by_trace.get(tid, []) for tid in restored_trace_ids]
        )
        backbone_deepseek_token_summaries = [
            self._summarize_backbone_deepseek_token_usage(chain_events_by_trace.get(tid, []))
            for tid in restored_trace_ids
        ]
        restored.non_tensor_batch["backbone_deepseek_token_usage"] = self._make_object_array(
            backbone_deepseek_token_summaries
        )
        restored.non_tensor_batch["backbone_deepseek_input_tokens"] = np.array(
            [summary.get("input_tokens", None) for summary in backbone_deepseek_token_summaries],
            dtype=object,
        )
        restored.non_tensor_batch["backbone_deepseek_output_tokens"] = np.array(
            [summary.get("output_tokens", None) for summary in backbone_deepseek_token_summaries],
            dtype=object,
        )
        restored.non_tensor_batch["backbone_deepseek_total_tokens"] = np.array(
            [summary.get("total_tokens", None) for summary in backbone_deepseek_token_summaries],
            dtype=object,
        )
        restored.non_tensor_batch["backbone_deepseek_usage_missing_count"] = np.array(
            [summary.get("num_missing_usage", 0) for summary in backbone_deepseek_token_summaries],
            dtype=np.int32,
        )
        if is_validation_rollout and self.io_trace_record_sample_chain:
            for trace_id, token_summary in zip(restored_trace_ids, backbone_deepseek_token_summaries):
                self._append_io_trace(
                    "orchestrator.validation_sample_backbone_deepseek_token_usage",
                    {
                        "trace_id": trace_id,
                        "token_usage": token_summary,
                    },
                )
        restored.non_tensor_batch["policy_request_ids"] = self._make_object_array(
            [policy_request_ids_by_trace.get(tid, []) for tid in restored_trace_ids]
        )
        restored.non_tensor_batch["policy_prompt"] = np.array(
            [policy_prompt_by_trace.get(tid, "") for tid in restored_trace_ids], dtype=object
        )
        restored.non_tensor_batch["policy_full_trace_output"] = np.array(
            [policy_full_trace_output_by_trace.get(tid, "") for tid in restored_trace_ids], dtype=object
        )
        restored.non_tensor_batch["final_backbone_binary_score"] = np.array(
            [final_binary_score_by_trace.get(tid, None) for tid in restored_trace_ids], dtype=object
        )
        restored.non_tensor_batch["backbone_final_source"] = np.array(
            [backbone_final_source_by_trace.get(tid, "") for tid in restored_trace_ids], dtype=object
        )
        restored.non_tensor_batch["response_source"] = np.array(
            [backbone_final_source_by_trace.get(tid, "") for tid in restored_trace_ids], dtype=object
        )
        restored.non_tensor_batch["pair_group_id"] = np.array(
            [pair_group_id_by_trace.get(tid, "") for tid in restored_trace_ids], dtype=object
        )
        restored.non_tensor_batch["source_uid"] = np.array(
            [source_uid_by_trace.get(tid, "") for tid in restored_trace_ids], dtype=object
        )
        restored.non_tensor_batch["orchestrator_round_count"] = np.array(
            [len(chain_events_by_trace.get(tid, [])) for tid in restored_trace_ids], dtype=np.int32
        )
        restored.non_tensor_batch["backbone_final_answer"] = np.array(self._decode_batch_responses(restored), dtype=object)
        backbone_em, backbone_f1 = self._compute_backbone_reference_scores(restored)
        if backbone_em is not None and backbone_f1 is not None:
            restored.non_tensor_batch["backbone_final_em"] = backbone_em
            restored.non_tensor_batch["backbone_final_f1"] = backbone_f1
        trajectory_snapshot = restored.select(deepcopy=True)
        self._latest_rollout_trajectory_batch = trajectory_snapshot

        if emit_policy_round_train_points:
            if policy_round_parts:
                if len(policy_round_parts) == 1:
                    policy_train_points = policy_round_parts[0]
                else:
                    policy_train_points = self._concat_dataproto_parts_with_seq_padding(policy_round_parts)
            else:
                policy_train_points = restored[:0]
            if return_trajectory_batch:
                return policy_train_points, trajectory_snapshot
            return policy_train_points
        if return_trajectory_batch:
            return restored, trajectory_snapshot
        return restored

    def _expand_tool_reward_points(self, batch: DataProto) -> DataProto:
        """Expand one trajectory into multiple GRPO datapoints: one per tool-call reward."""
        if "tool_rewards" not in batch.non_tensor_batch:
            return batch

        expanded_indices: list[int] = []
        expanded_rewards: list[float] = []
        expanded_step_ids: list[int] = []

        tool_rewards_arr = batch.non_tensor_batch["tool_rewards"]
        for i in range(len(batch)):
            rewards = tool_rewards_arr[i]
            if isinstance(rewards, (list, tuple)) and len(rewards) > 0:
                for step_idx, r in enumerate(rewards):
                    expanded_indices.append(i)
                    expanded_rewards.append(float(r))
                    expanded_step_ids.append(step_idx)
            else:
                # Keep at least one datapoint so this sample is not dropped.
                expanded_indices.append(i)
                expanded_rewards.append(0.0)
                expanded_step_ids.append(0)

        expanded = batch[expanded_indices]
        expanded.non_tensor_batch["trajectory_index"] = np.array(expanded_indices, dtype=np.int32)
        expanded.non_tensor_batch["tool_reward_step"] = np.array(expanded_step_ids, dtype=np.int32)

        # Build rm_scores from per-point tool reward at final valid generated token.
        response_mask = expanded.batch["response_mask"]
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        valid_response_length = response_mask.sum(dim=-1).to(torch.long) - 1
        valid_response_length = torch.clamp(valid_response_length, min=0)
        reward_tensor = torch.tensor(expanded_rewards, dtype=torch.float32, device=rm_scores.device)
        rm_scores[torch.arange(rm_scores.size(0), device=rm_scores.device), valid_response_length] = reward_tensor
        expanded.batch["rm_scores"] = rm_scores
        return expanded

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        validation_backbone_token_records: list[dict[str, Any]] = []

        show_progress_bar = self.config.trainer.get("show_progress_bar", True)
        if isinstance(show_progress_bar, str):
            show_progress_bar = show_progress_bar.strip().lower() in ("1", "true", "yes", "y", "on")
        show_progress_bar = bool(show_progress_bar)
        val_iterator = tqdm(
            self.val_dataloader,
            total=len(self.val_dataloader),
            desc="Validation Batches",
            dynamic_ncols=True,
            disable=not show_progress_bar,
        )

        for val_batch_idx, test_data in enumerate(val_iterator):
            test_batch_timing = {}
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            if not self._is_backbone_rollout_enabled():
                # In non-orchestrator mode, validation keeps the classic N-rollout repeat.
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
                )

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            with marked_timer("validation_gen", test_batch_timing, color="green"):
                if self._is_backbone_rollout_enabled():
                    # Keep validation and training rollout paths consistent so multi-round
                    # backbone/policy traces are available in val-only runs.
                    test_output_gen_batch = self._run_two_stage_orchestrator_rollout(
                        gen_batch_output=test_gen_batch,
                        timing_raw=test_batch_timing,
                        curr_step_profile=False,
                    )
                else:
                    # pad to be divisible by dp_size
                    size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
                    test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

                    # unpad
                    test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                    rollout_timing = test_output_gen_batch.meta_info.get("timing", {})
                    self._accumulate_round_timing(test_batch_timing, rollout_timing, prefix="validation_rollout")
                    test_output_gen_batch.meta_info.pop("timing", None)

            if self.use_rm and "rm_scores" not in test_output_gen_batch.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                with marked_timer("validation_reward", test_batch_timing, color="yellow"):
                    batch_reward = self._compute_reward_colocate(test_output_gen_batch)
                test_output_gen_batch = test_output_gen_batch.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                with marked_timer("validation_update_weights", test_batch_timing, color="red"):
                    self.checkpoint_manager.update_weights(self.global_steps)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            with marked_timer("validation_merge", test_batch_timing):
                if self._is_backbone_rollout_enabled():
                    source_indices = test_output_gen_batch.non_tensor_batch.get("orchestrator_source_index", None)
                    if isinstance(source_indices, np.ndarray) and source_indices.shape[0] == len(test_output_gen_batch):
                        test_batch = test_batch[source_indices.tolist()]
                    elif len(test_batch) != len(test_output_gen_batch):
                        raise RuntimeError(
                            "Validation batch size mismatch under backbone orchestrator rollout: "
                            f"base={len(test_batch)} vs output={len(test_output_gen_batch)}, "
                            "and missing/invalid orchestrator_source_index for alignment."
                        )

                self._drop_conflicting_rollout_non_tensor_keys(test_batch, test_output_gen_batch)
                test_batch = test_batch.union(test_output_gen_batch)
                test_batch.meta_info["validate"] = True

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            with marked_timer("validation_reward_extract", test_batch_timing):
                reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            def _normalize_for_val_metrics(value: Any) -> Any:
                # process_validation_metrics expects scalar-like numeric or str values.
                # Keep structured objects for dump by serializing to JSON strings so metrics code skips them.
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, (list, tuple, dict)):
                    try:
                        return json.dumps(value, ensure_ascii=False)
                    except Exception:
                        return str(value)
                return value

            for key in [
                "orchestrator_trace_id",
                "orchestrator_source_index",
                "policy_request_ids",
                "policy_prompt",
                "policy_full_trace_output",
                "final_backbone_binary_score",
                "backbone_final_source",
                "pair_group_id",
                "source_uid",
                "orchestrator_chain",
                "orchestrator_round_count",
                "backbone_final_answer",
                "backbone_deepseek_token_usage",
                "backbone_deepseek_input_tokens",
                "backbone_deepseek_output_tokens",
                "backbone_deepseek_total_tokens",
                "backbone_deepseek_usage_missing_count",
            ]:
                if key in test_batch.non_tensor_batch:
                    val = test_batch.non_tensor_batch[key]
                    if isinstance(val, np.ndarray):
                        reward_extra_infos_dict.setdefault(key, []).extend(
                            [_normalize_for_val_metrics(x) for x in val.tolist()]
                        )
                    else:
                        vals = val if isinstance(val, list) else [val] * reward_tensor.shape[0]
                        reward_extra_infos_dict.setdefault(key, []).extend(
                            [_normalize_for_val_metrics(x) for x in vals]
                        )
            if "source_uid" not in reward_extra_infos_dict and "uid" in test_batch.non_tensor_batch:
                uid_vals = test_batch.non_tensor_batch["uid"]
                reward_extra_infos_dict["source_uid"] = uid_vals.tolist() if isinstance(uid_vals, np.ndarray) else list(uid_vals)

            for key, values in self._build_per_sample_log_fields(
                test_batch,
                phase="validation",
                timing_raw=test_batch_timing,
                phase_batch_index=val_batch_idx,
            ).items():
                reward_extra_infos_dict.setdefault(key, []).extend(values)

            backbone_em, backbone_f1 = self._compute_backbone_reference_scores(test_batch)
            if backbone_em is not None and backbone_f1 is not None:
                test_batch.non_tensor_batch["backbone_final_em"] = backbone_em.astype(np.float32)
                test_batch.non_tensor_batch["backbone_final_f1"] = backbone_f1.astype(np.float32)
                reward_extra_infos_dict.setdefault("backbone_final_em", []).extend(backbone_em.tolist())
                reward_extra_infos_dict.setdefault("backbone_final_f1", []).extend(backbone_f1.tolist())

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            if self._is_backbone_rollout_enabled():
                validation_backbone_token_records.extend(
                    self._build_validation_backbone_deepseek_token_records(
                        test_batch,
                        phase_batch_index=val_batch_idx,
                    )
                )
            self._append_batch_log_trace(
                phase="validation",
                batch=test_batch,
                timing_raw=test_batch_timing,
                phase_batch_index=val_batch_idx,
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )
            if validation_backbone_token_records:
                token_record_path = self._dump_jsonl_records(
                    validation_backbone_token_records,
                    os.path.join(val_data_dir, "backbone_deepseek_tokens"),
                    f"{self.global_steps}.jsonl",
                )
                self._append_io_trace(
                    "validation.backbone_deepseek_token_records_dump",
                    {
                        "path": token_record_path,
                        "num_records": len(validation_backbone_token_records),
                    },
                )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create optional frozen backbone rollout worker for two-stage orchestration
        if Role.BackboneRollout in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.BackboneRollout)
            backbone_rollout_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.BackboneRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.BackboneRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.BackboneRollout)] = backbone_rollout_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        # Emit init diagnostics to locate missing role bindings (e.g., backbone_rollout).
        planned_roles = []
        for _, class_dict in self.resource_pool_to_cls.items():
            planned_roles.extend(list(class_dict.keys()))
        self._append_io_trace(
            event="orchestrator.init_workers_planned_roles",
            payload={
                "planned_roles": sorted(set(planned_roles)),
            },
        )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self._append_io_trace(
            event="orchestrator.init_workers_spawned_roles",
            payload={
                "spawned_roles": sorted(all_wg.keys()),
            },
        )

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if str(Role.BackboneRollout) in all_wg:
            self.backbone_rollout_wg = all_wg[str(Role.BackboneRollout)]
            self.backbone_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        if self.backbone_rollout_wg is not None:
            rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None) or {}
            backbone_use_api = bool(rollout_custom.get("backbone_use_api", False))
            if not backbone_use_api:
                # Backbone manager is frozen and only used for stage-1 orchestration.
                self.backbone_async_rollout_manager = AgentLoopManager.create(
                    config=self.config,
                    worker_group=self.backbone_rollout_wg,
                    rollout_resource_pool=actor_rollout_resource_pool,
                    reward_loop_worker_handles=None,
                )
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        dropped_for_balance = batch_size % dp_size
        if dropped_for_balance != 0:
            kept_batch_size = batch_size - dropped_for_balance
            logger.warning(
                "Dropping %s sample(s) before DP sequence balancing because batch_size=%s is not divisible by dp_size=%s.",
                dropped_for_balance,
                batch_size,
                dp_size,
            )
            keep_indices = torch.arange(kept_batch_size, device=attention_mask.device)
            batch.reorder(keep_indices)
            metrics[f"{logging_prefix}/dropped_for_dp_balance"] = dropped_for_balance
            attention_mask = batch.batch["attention_mask"]
            batch_size = attention_mask.shape[0]
        else:
            metrics[f"{logging_prefix}/dropped_for_dp_balance"] = 0

        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(
                batch_td,
                compute_loss=False,
                max_response_length=int(batch.batch["responses"].shape[-1]),
            )
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {
                "calculate_entropy": False,
                "compute_loss": False,
                "max_response_length": int(batch.batch["responses"].shape[-1]),
            }
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=True,
                compute_loss=False,
                max_response_length=int(batch.batch["responses"].shape[-1]),
            )
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            routed_experts = tu.get(output, "routed_experts")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            if routed_experts is not None:
                old_log_prob = tu.get_tensordict(
                    {"old_log_probs": log_probs.float(), "entropys": entropy.float(), "routed_experts": routed_experts}
                )
            else:
                old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
                max_response_length=int(batch.batch["responses"].shape[-1]),
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
                max_response_length=int(batch.batch["responses"].shape[-1]),
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                backbone_rollout_enabled = self._is_backbone_rollout_enabled()
                self._latest_rollout_trajectory_batch = None
                if backbone_rollout_enabled:
                    # Keep backbone stage at one trajectory per input sample.
                    gen_batch_output = gen_batch.select(deepcopy=True)
                else:
                    gen_batch_output = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if backbone_rollout_enabled:
                            gen_batch_output = self._run_two_stage_orchestrator_rollout(
                                gen_batch_output=gen_batch_output,
                                timing_raw=timing_raw,
                                curr_step_profile=curr_step_profile,
                            )
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                    trajectory_batch_for_logging = (
                        getattr(self, "_latest_rollout_trajectory_batch", None) if backbone_rollout_enabled else None
                    )

                    # Rollout may carry partial/normalized non-tensor columns that overlap with
                    # original batch fields (e.g. data_source). DataProto.union requires strict
                    # equality on overlapping keys, so drop conflicting rollout-side duplicates
                    # and keep source-of-truth from `batch`.
                    self._drop_conflicting_rollout_non_tensor_keys(batch, gen_batch_output)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # Align original batch to rollout trajectories.
                    if backbone_rollout_enabled:
                        source_indices = gen_batch_output.non_tensor_batch.get("orchestrator_source_index", None)
                        if isinstance(source_indices, np.ndarray) and source_indices.shape[0] == len(gen_batch_output):
                            batch = batch[source_indices.tolist()]
                        else:
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    else:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # Optional mode: each tool-call score is treated as one GRPO datapoint.
                    # We expand trajectories before reward extraction and directly construct rm_scores.
                    if self.config.algorithm.get("tool_reward_as_grpo_point", False):
                        expansion_summary_before = self._summarize_tool_reward_lengths(batch)
                        batch = self._expand_tool_reward_points(batch)
                        metrics["tool_reward_expansion/original_batch_size"] = expansion_summary_before["num_trajectories"]
                        metrics["tool_reward_expansion/expanded_batch_size"] = len(batch)
                        metrics["tool_reward_expansion/num_with_tool_rewards"] = expansion_summary_before[
                            "num_with_tool_rewards"
                        ]
                        metrics["tool_reward_expansion/num_without_tool_rewards"] = expansion_summary_before[
                            "num_without_tool_rewards"
                        ]
                        metrics["tool_reward_expansion/total_reward_points"] = expansion_summary_before[
                            "total_reward_points"
                        ]
                        self._append_io_trace(
                            "trainer.tool_reward_expansion",
                            {
                                **expansion_summary_before,
                                "expanded_batch_size": len(batch),
                            },
                        )

                    reward_extra_infos_dict: dict[str, list] = {}
                    has_policy_train_points = len(batch) > 0
                    metrics["training/num_policy_train_points"] = int(len(batch))
                    metrics["training/skipped_no_policy_train_points"] = 0.0 if has_policy_train_points else 1.0

                    if has_policy_train_points:
                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        # get images_seqlens
                        images_seqlens_all = []
                        for multi_modal_input in batch.non_tensor_batch.get("multi_modal_inputs", []):
                            if multi_modal_input is None:
                                continue
                            if "image_grid_thw" not in multi_modal_input.keys():
                                continue
                            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                        batch.meta_info["images_seqlens"] = images_seqlens_all
                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # extract reward_tensor and reward_extra_infos_dict for training
                            reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                            backbone_em, backbone_f1 = self._compute_backbone_reference_scores(batch)
                            if backbone_em is not None and backbone_f1 is not None:
                                reward_extra_infos_dict["backbone_final_em"] = backbone_em.tolist()
                                reward_extra_infos_dict["backbone_final_f1"] = backbone_f1.tolist()
                                batch.non_tensor_batch["backbone_final_em"] = backbone_em.astype(np.float32)
                                batch.non_tensor_batch["backbone_final_f1"] = backbone_f1.astype(np.float32)
                                metrics["train-aux/backbone_final_em/mean"] = float(np.mean(backbone_em))
                                metrics["train-aux/backbone_final_f1/mean"] = float(np.mean(backbone_f1))
                                metrics["train-aux/backbone_final_em/max"] = float(np.max(backbone_em))
                                metrics["train-aux/backbone_final_f1/max"] = float(np.max(backbone_f1))

                            metrics.update(self._compute_backbone_deepseek_step_token_metrics(batch))

                        # Operating Mode Selection:
                        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                            apply_bypass_mode(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                actor_config = self.config.actor_rollout_ref.actor
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                    raise ValueError(
                                        "Detected conflicting router replay configuration: "
                                        "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                        "cannot be enabled simultaneously. "
                                        "The enable_rollout_routing_replay option is only used in R3 mode; "
                                        "it should not be set when using R2 mode."
                                    )
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                ref_log_prob = self._compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self._compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                        # update critic
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self._update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self._update_actor(batch)

                            # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                            esi_close_to_expiration = should_save_ckpt_esi(
                                max_steps_duration=self.max_steps_duration,
                                redundant_time=self.config.trainer.esi_redundant_time,
                            )
                            # Check if the conditions for saving a checkpoint are met.
                            # The conditions include a mandatory condition (1) and
                            # one of the following optional conditions (2/3/4):
                            # 1. The save frequency is set to a positive value.
                            # 2. It's the last training step.
                            # 3. The current step number is a multiple of the save frequency.
                            # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                            if self.config.trainer.save_freq > 0 and (
                                is_last_step
                                or self.global_steps % self.config.trainer.save_freq == 0
                                or esi_close_to_expiration
                            ):
                                if esi_close_to_expiration:
                                    print("Force saving checkpoint: ESI instance expiration approaching.")
                                with marked_timer("save_checkpoint", timing_raw, color="green"):
                                    self._save_checkpoint()

                            # update weights from trainer to rollout
                            with marked_timer("update_weights", timing_raw, color="red"):
                                self.checkpoint_manager.update_weights(self.global_steps)

                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)
                    else:
                        self._append_io_trace(
                            "trainer.skip_no_policy_train_points",
                            {
                                "global_step": int(self.global_steps),
                                "backbone_rollout_enabled": bool(backbone_rollout_enabled),
                            },
                        )

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(
                            batch,
                            reward_extra_infos_dict,
                            timing_raw,
                            rollout_data_dir,
                            trajectory_batch=trajectory_batch_for_logging,
                        )

                # validate
                if self._should_run_validation(is_last_step):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                if len(batch) > 0 and batch.batch is not None and "token_level_scores" in batch.batch.keys():
                    # collect metrics
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    # GDPO per-component reward metrics
                    gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                    if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                        for key in gdpo_reward_keys:
                            if key in batch.non_tensor_batch:
                                vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                                metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                                metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                                metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                                metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    # TODO: implement actual tflpo and theoretical tflpo
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    # compute variance proxy metrics
                    gradient_norm = metrics.get("actor/grad_norm", None)
                    metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                    # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                    # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
