# Copyright 2025 Amazon.com Inc and/or its affiliates
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
Dataset class that enables dynamic data generation strategies between iterations of training.
This class extends RLHFDataset and uses an AbstractDataGen instance to generate data.

This is especially useful in settings where proposer model generates new tasks based
on rollout data.
"""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Optional

import datasets
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.utils.dataset import RLHFDataset
from verl.utils.import_utils import load_extern_object

logger = logging.getLogger(__name__)


class AbstractDataGenerator(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def generate(self, dataset: Dataset, batch: Optional[DataProto] = None) -> datasets.Dataset:
        """
        Generate method must be implemented by subclasses.
        Args:
            dataset: The dataset to generate from.
            batch: The latest training batch (optional). Can be used to build
                hard-example replay or online augmentation strategies.
        Returns:
            Processed data or result as implemented by the subclass.
        """
        pass


class MockDataGenerator(AbstractDataGenerator):
    """
    A noop data gen class that only reappends the first datapoint.
    This class is useful as a placeholder and testing.
    """

    def __init__(self, config: DictConfig = None):
        super().__init__(config)

    def generate(self, dataset: Dataset, batch: Optional[DataProto] = None) -> datasets.Dataset:
        print("MockDataGenerator: No operation performed on the dataset.")
        return dataset.dataframe.select([0])


class ReplayBatchDataGenerator(AbstractDataGenerator):
    """Replay current training batch into dataset without prompt synthesis.

    This generator reuses non-tensor fields from the rollout/training batch and
    writes them back as new rows for subsequent steps. It does not construct or
    rewrite prompts.
    """

    def __init__(self, config: DictConfig = None):
        super().__init__(config)
        self.only_tool_samples = bool(config.get("only_tool_samples", False)) if config is not None else False
        self.dedupe_by_uid = bool(config.get("dedupe_by_uid", True)) if config is not None else True
        self.max_new_samples_per_step = int(config.get("max_new_samples_per_step", 0)) if config is not None else 0
        self.prompt_field_prefer_raw = bool(config.get("prompt_field_prefer_raw", True)) if config is not None else True

    @staticmethod
    def _to_py(val):
        if hasattr(val, "item"):
            try:
                return val.item()
            except Exception:
                pass
        return val

    @staticmethod
    def _is_tool_sample(non_tensor_batch: dict, idx: int) -> bool:
        tool_rewards = non_tensor_batch.get("tool_rewards", None)
        if tool_rewards is not None:
            rewards = tool_rewards[idx]
            if isinstance(rewards, (list, tuple)) and len(rewards) > 0:
                return True
        turns = non_tensor_batch.get("__num_turns__", None)
        if turns is not None:
            try:
                return int(turns[idx]) > 1
            except Exception:
                return False
        return False

    def generate(self, dataset: Dataset, batch: Optional[DataProto] = None) -> datasets.Dataset:
        if batch is None or not getattr(batch, "non_tensor_batch", None):
            return datasets.Dataset.from_list([])

        non_tensor_batch = batch.non_tensor_batch
        batch_size = len(batch)
        if batch_size == 0:
            return datasets.Dataset.from_list([])

        uid_seen = set()
        replay_indices: list[int] = []
        uid_arr = non_tensor_batch.get("uid", None)

        for i in range(batch_size):
            if self.only_tool_samples and (not self._is_tool_sample(non_tensor_batch, i)):
                continue

            if self.dedupe_by_uid and uid_arr is not None:
                uid_val = self._to_py(uid_arr[i])
                if uid_val in uid_seen:
                    continue
                uid_seen.add(uid_val)

            replay_indices.append(i)
            if self.max_new_samples_per_step > 0 and len(replay_indices) >= self.max_new_samples_per_step:
                break

        if len(replay_indices) == 0:
            return datasets.Dataset.from_list([])

        base_columns = list(getattr(dataset, "dataframe").column_names)
        prompt_field = "raw_prompt" if self.prompt_field_prefer_raw and "raw_prompt" in non_tensor_batch else "prompt"
        if prompt_field not in non_tensor_batch:
            logger.warning("ReplayBatchDataGenerator: no prompt/raw_prompt found in batch, skip replay.")
            return datasets.Dataset.from_list([])

        current_len = len(getattr(dataset, "dataframe"))
        rows = []
        for offset, i in enumerate(replay_indices):
            row = {col: None for col in base_columns}
            row["prompt"] = copy.deepcopy(self._to_py(non_tensor_batch[prompt_field][i]))

            if "data_source" in non_tensor_batch:
                row["data_source"] = copy.deepcopy(self._to_py(non_tensor_batch["data_source"][i]))
            if "reward_model" in non_tensor_batch:
                row["reward_model"] = copy.deepcopy(self._to_py(non_tensor_batch["reward_model"][i]))
            if "ability" in non_tensor_batch and "ability" in row:
                row["ability"] = copy.deepcopy(self._to_py(non_tensor_batch["ability"][i]))

            extra_info = {}
            if "extra_info" in non_tensor_batch and non_tensor_batch["extra_info"][i] is not None:
                val = copy.deepcopy(self._to_py(non_tensor_batch["extra_info"][i]))
                if isinstance(val, dict):
                    extra_info = val
            extra_info.setdefault("index", current_len + offset)
            row["extra_info"] = extra_info

            rows.append(row)

        return datasets.Dataset.from_list(rows)


class DynamicGenDataset(RLHFDataset):
    """
    A dataset class that uses a data generation strategy to process data.
    This class extends RLHFDataset and uses an AbstractDataGen instance to generate data.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(data_files, tokenizer, config, processor, max_samples=max_samples)
        self.datagen: AbstractDataGenerator = config.datagen
        assert "datagen" in config and config.datagen.get("path", None) is not None, (
            f"datagen path is not set in config: {config}"
        )
        # Dynamically load the custom datagen class
        datagen_cls = load_extern_object(config.datagen.path, config.datagen.name)

        # Verify that the custom datagen class inherits from AbstractDataGenerator
        abs_cls = AbstractDataGenerator
        if not issubclass(datagen_cls, abs_cls):
            raise TypeError(
                f"The custom datagen class '{config.datagen.name}' from '{config.datagen.path}'"
                + " must inherit from {abs_cls}"
            )

        self.data_generator = datagen_cls(config.datagen)
        # Optional warm-start generation before training starts.
        if bool(config.datagen.get("generate_on_init", False)):
            self.on_batch_end(batch=None)

    def append_dataframe(self, new_dataframe: datasets.Dataset):
        if new_dataframe is None or len(new_dataframe) == 0:
            return
        new_dataframe = self.maybe_filter_out_long_prompts(new_dataframe)
        if len(new_dataframe) == 0:
            return
        self.dataframe = datasets.concatenate_datasets([self.dataframe, new_dataframe])

        logger.info(f"new dataset len: {len(self.dataframe)}")

    def on_batch_end(self, batch: Optional[DataProto] = None) -> None:
        """
        Generate data using the provided data generation strategy.
        Note: This method is intended to change the dataset after each training batch.
        """
        new_data = self.data_generator.generate(self, batch=batch)
        self.append_dataframe(new_data)
