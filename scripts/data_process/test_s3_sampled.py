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
import pandas as pd
from collections import defaultdict

from verl.utils.hdfs_io import copy, makedirs
import argparse

# def make_prefix(dp, template_type):
#     question = dp['question']

#     # NOTE: also need to change reward_score/countdown.py
#     if template_type == 'base':
#         """This works for any base model"""
#         prefix = f"""Answer the given question. \
# You must conduct reasoning inside <think> and </think> first every time you get new information. \
# After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
# You can search as many times as your want. \
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
#     else:
#         raise NotImplementedError
#     return prefix


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


def build_sampled_question_set(sample_source_parquet: str, sample_per_source: int, sample_seed: int):
    if not os.path.exists(sample_source_parquet):
        raise FileNotFoundError(
            f"Sample source parquet not found: {sample_source_parquet}"
        )

    df = pd.read_parquet(sample_source_parquet)
    source_questions = defaultdict(set)

    for data_source in df['data_source'].unique():
        source_df = df[df['data_source'] == data_source]
        if len(source_df) > sample_per_source:
            source_df = source_df.sample(n=sample_per_source, random_state=sample_seed)

        for _, row in source_df.iterrows():
            q = row['reward_model']['ground_truth']['question']
            source_questions[data_source].add(normalize_question(q))

    total = sum(len(v) for v in source_questions.values())
    print(f"Total sampled questions: {total}")
    for ds in source_questions:
        print(f"{ds}: {len(source_questions[ds])}")

    return source_questions



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--retriever', default="bm25")
    parser.add_argument('--max_turns_hint', type=int, default=4)
    parser.add_argument('--sample_source_parquet', default='data/nq_hotpotqa_train/test_e5_s3.parquet')
    parser.add_argument('--sample_per_source', type=int, default=3000)
    parser.add_argument('--sample_seed', type=int, default=42)

    args = parser.parse_args()

    source_questions = build_sampled_question_set(
        sample_source_parquet=args.sample_source_parquet,
        sample_per_source=args.sample_per_source,
        sample_seed=args.sample_seed,
    )

    data_sources = [x for x in args.data_sources.split(',') if x]
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

        # Remove duplicates for popqa
        if data_source == 'popqa':
            seen_questions = set()
            unique_examples = []
            for example in test_dataset:
                question = example['question'].strip()
                if question[-1] != '?':
                    question += '?'
                if question not in seen_questions:
                    seen_questions.add(question)
                    unique_examples.append(example)
            test_dataset = datasets.Dataset.from_list(unique_examples)
            print(f"\nAfter removing duplicates, popqa has {len(test_dataset)} questions")

        # Check for duplicate questions
        question_counts = {}
        for example in test_dataset:
            question = example['question'].strip()
            if question[-1] != '?':
                question += '?'
            question_counts[question] = question_counts.get(question, 0) + 1
        
        # Print duplicates for popqa
        if data_source == 'popqa':
            print(f"\nChecking duplicates in {data_source}:")
            for question, count in question_counts.items():
                if count > 1:
                    print(f"Question appears {count} times: {question}")

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
            # Clean question for question set check
            question = normalize_question(example['question'])
            
            # Only include questions that were sampled from test_e5_ug.parquet
            return question in source_questions.get(data_source, set())

        # First filter, then map
        test_dataset = test_dataset.filter(filter_fn)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.to_parquet(os.path.join(local_dir, f'test_{args.retriever}_s3_sampled.parquet'))
    
    # print statistics of test_u1_sampled.parquet
    df = pd.read_parquet(os.path.join(local_dir, f'test_{args.retriever}_s3_sampled.parquet'))
    print(f"Total number of questions: {len(df)}")
    for data_source in df['data_source'].unique():
        print(f"{data_source}: {len(df[df['data_source'] == data_source])}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
