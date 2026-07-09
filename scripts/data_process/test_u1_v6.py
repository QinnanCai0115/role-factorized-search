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

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
import json

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


def make_prefix(dp, retriever):

    # input_str = """<|im_start|>system\nA conversation between User and Assistant. The User asks a question, and the Assistant solves it.<|im_end|>\n<|im_start|>user\n"""
    input_str = """You are a search copilot for the generation model. Based on a user's query and initial searched results, you will first determine if the searched results are enough to produce an answer.
If the searched results are enough, you will use <search_complete>True</search_complete> to indicate that you have gathered enough information for the generation model to produce an answer.
If the searched results are not enough, you will go through a loop of <query> -> <information> -> <important_info> -> <search_complete> -> <query> (if not complete) ..., to help the generation model to generate a better answer with more relevant information searched.
You should show the search query between <query> and </query> in JSON format.
Based on the search query, we will return the top searched results between <information> and </information>. You need to put the doc ids of the important documents (up to 3 documents, within the current information window) between <important_info> and </important_info> (e.g., <important_info>[1, 4]</important_info>).
A search query MUST be followed by a <search_complete> tag if the search is not complete.
After reviewing the information, you must decide whether to continue searching with a new query or indicate that the search is complete. If you need more information, use <search_complete>False</search_complete> to indicate you want to continue searching with a better query. Otherwise, use <search_complete>True</search_complete> to terminate the search.
During the process, you can add reasoning process within <think></think> tag whenever you want. Note: Only the important information would be used for the generation model to produce an answer.
"""

    if retriever == "bm25":
        input_str += """Note: The search query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately."""

    input_str += """
For a question and initial searched results:
<question>
[user's question]
</question>
<information>
[initial searched results]
</information>

If the initial searched results are enough to produce an answer, you should output:
<search_complete>
True
</search_complete>

If the initial searched results are not enough to produce an answer, you should output:
<query>
{
    "query": "[search query]"
} 
</query>
<information>
[top searched results based on the above search query]
</information>
<important_info>
[doc ids]
</important_info>
<search_complete>
False
</search_complete>
<query>
{
    "query": "[search query]"
}
</query>
...... (can be several turns until <search_complete> is True)

<search_complete>
True
</search_complete>

Now, start the loop with the following question and initial searched results:
"""

    input_str += f"""
<question>
{dp['question']}
</question>
<information>
{dp['initial_searched_results'].strip()}
</information>
"""
    return input_str



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--retriever', default="bm25")
    parser.add_argument('--initial_searched_results_dir', default="data/RAG_Retrieval/test")
    
    
    import pandas as pd
    df = pd.read_parquet("data/nq_hotpotqa_train/test_e5_ug.parquet")
    
    question_set = set()

    from collections import defaultdict
    source_questions = defaultdict(list)
    
    for data_source in df['data_source'].unique():
        source_df = df[df['data_source'] == data_source]
        if len(source_df) > 3000:
            sampled_df = source_df.sample(n=3000, random_state=42)
        else:
            sampled_df = source_df
        for i in range(len(sampled_df)):
            question = sampled_df.iloc[i]['reward_model']['ground_truth']['question']
            if question not in question_set:
                question_set.add(question)
                source_questions[data_source].append(question)
    
    # print statistics
    print(f"Total number of questions: {len(question_set)}")
    for data_source in source_questions:
        print(f"{data_source}: {len(source_questions[data_source])}")
    
    
        
    
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
            
        initial_searched_results = json.load(open(os.path.join(args.initial_searched_results_dir, f'{data_source}_output_sequences.json')))

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
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                example['initial_searched_results'] = initial_searched_results[example['question']]['context_with_info'].split("\nDoc 4")[0] + "\n"
                
                question = make_prefix(example, args.retriever)
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
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        def filter_fn(example):
            # Clean question for question set check
            question = example['question'].strip()
            if question[-1] != '?':
                question += '?'
            
            # Only include questions that were sampled from test_e5_ug.parquet
            return question in source_questions[data_source]

        # First filter, then map
        test_dataset = test_dataset.filter(filter_fn)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.to_parquet(os.path.join(local_dir, f'test_{args.retriever}_u1_sampled.parquet'))
    
    # print statistics of test_u1_sampled.parquet
    df = pd.read_parquet(os.path.join(local_dir, f'test_{args.retriever}_u1_sampled.parquet'))
    print(f"Total number of questions: {len(df)}")
    for data_source in df['data_source'].unique():
        print(f"{data_source}: {len(df[df['data_source'] == data_source])}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
