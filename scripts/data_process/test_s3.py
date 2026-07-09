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
"""
Preprocess the QA dataset to parquet format
"""

import os
import datasets
from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix_rl(dp, retriever, max_turns=4):
    input_str = """You are a search policy model trained with reinforcement learning.
Your objective is to improve retrieval quality through iterative tool calls.
At each step, either emit a new query between <query> and </query> or end with <search_complete>True</search_complete>.
You may reason in <think></think> and optionally mark key doc ids using <important_info></important_info>.
"""

    if retriever == "bm25":
        input_str += """Use Boolean operators (AND, OR) and parentheses when appropriate."""

    input_str += f"""

Question:
<question>
{dp['question']}
</question>

Interaction protocol:
- Query format:
<query>
{{
  "query": "[search query]"
}}
</query>
- Continue format: <search_complete>False</search_complete>
- Stop format: <search_complete>True</search_complete>
- Maximum recommended turns: {max_turns}
"""

    return input_str


def normalize_question(question: str) -> str:
    question = question.strip()
    if question and question[-1] != '?':
        question += '?'
    return question

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--retriever', default="bm25")
    parser.add_argument('--max_turns_hint', type=int, default=4)

    args = parser.parse_args()

    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:
        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

        if 'test' in dataset:
            print(f'Using the {data_source} test dataset...')
            test_dataset = dataset['test']
        elif 'dev' in dataset:
            print(f'Using the {data_source} dev dataset...')
            test_dataset = dataset['dev']
        else:
            print(f'Using the {data_source} train dataset...')
            test_dataset = dataset['train']

        def make_map_fn(split):
            def process_fn(example, idx):
                example['question'] = normalize_question(example['question'])
                question = make_prefix_rl(example, args.retriever, max_turns=args.max_turns_hint)
                solution = {
                    "question": example['question'],
                    "target": example['golden_answers'],
                    "gt_docs": example['supporting_facts'] if 'supporting_facts' in example else []
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "rl_profile": {
                        "format_variant": "rl",
                        "unit": "tool_call",
                        "grouping": "per_call",
                        "max_turns_hint": args.max_turns_hint,
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        'source_question': example['question'],
                    }
                }
                return data

            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.to_parquet(os.path.join(local_dir, f'test_{args.retriever}_s3.parquet'))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
