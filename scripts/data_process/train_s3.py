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
    """Backbone-oriented prompt for iterative search-assisted reasoning."""
    input_str = """You are a frozen backbone reasoning model.
Your goal is to solve the question correctly. If your current context is insufficient,
you can request additional retrieval by emitting a search request wrapped in <search></search>.

Interaction rules:
1) When you need retrieval, output one search request in this format:
<search>[search request]</search>
2) The search subagent will return retrieved evidence wrapped in <information></information>.
3) You must continue reasoning based on all accumulated context and newly returned information.
4) If needed, you may issue another <search> request. Multi-round search is allowed.
5) When you have enough evidence, provide the final answer and stop requesting search.

You may reason in <think></think>.
"""

    if retriever == "bm25":
        input_str += """Use Boolean operators (AND, OR) and parentheses when appropriate."""

    input_str += f"""

Question:
<question>
{dp['question']}
</question>

Interaction protocol:
- Maximum recommended turns: {max_turns}.
- Search request format:
<search>[search request]</search>
- Retrieved evidence format:
<information>[retrieved evidence]</information>
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
    parser.add_argument('--sample_per_source', type=int, default=-1)
    parser.add_argument('--sample_seed', type=int, default=42)
    parser.add_argument('--keep_binary_answers', action='store_true')
    args = parser.parse_args()
    
    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:
        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        train_dataset = dataset['train']

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

        def filter_fn(example):
            # Optionally filter out yes/no style answers.
            if not args.keep_binary_answers:
                if any(
                    word in example['golden_answers']
                    for word in ['yes', 'no', 'true', 'false', 'Yes', 'No', 'True', 'False']
                ):
                    return False

            return True

        # First filter, then optional per-source sampling, then map.
        train_dataset = train_dataset.filter(filter_fn)
        if args.sample_per_source > 0 and len(train_dataset) > args.sample_per_source:
            train_dataset = train_dataset.shuffle(seed=args.sample_seed).select(range(args.sample_per_source))
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        all_dataset.append(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset.to_parquet(os.path.join(local_dir, f'train_{args.retriever}_s3.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
