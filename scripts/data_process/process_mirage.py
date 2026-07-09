import json
import os
import pandas as pd
import datasets
from typing import Dict, List


def make_prefix(full_question, initial_searched_results_str, retriever='e5'):

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
{full_question}
</question>
<information>
{initial_searched_results_str}
</information>
"""
    return input_str


def process_benchmark_data(file_path: str, initial_searched_results_dir: str) -> List[Dict]:
    """Process benchmark.json into the required format."""
    with open(file_path, 'r') as f:
        benchmark_data = json.load(f)
        
    
    processed_data = []
    
    for data_source, questions in benchmark_data.items():
        initial_searched_results = json.load(open(os.path.join(initial_searched_results_dir, f'{data_source}_output_sequences.json')))
        for q_id, q_data in questions.items():
            # Get the question and options
            question = q_data['question']
            options = q_data['options']
            
            # Create the full question with options
            full_question = f"{question}\nOptions:\n"
            for opt_key, opt_text in options.items():
                full_question += f"{opt_key}: {opt_text}\n"
            
            initial_searched_results_str = initial_searched_results[full_question]['context_with_info'].split("\nDoc 4")[0] + "\n"
            prompted_question = make_prefix(full_question, initial_searched_results_str)
            
            # Get the correct answer text based on the answer key
            answer_key = q_data['answer']
            golden_answer = [f"{answer_key}: {options[answer_key]}"]
            
            # Create the data point in the required format
            data_point = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompted_question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "question": full_question,
                        "target": golden_answer,
                        "gt_docs": []  # Empty list as we don't have supporting facts
                    }
                },
                "extra_info": {
                    'split': 'test',
                    'index': q_id,
                }
            }
            
            processed_data.append(data_point)
    
    return processed_data

def main():
    # Input and output paths
    input_file = "data/mirage/benchmark.json"
    output_file = "data/mirage/mirage_test.parquet"
    initial_searched_results_dir = "data/mirage/rag_e5"
    
    # Process the data
    processed_data = process_benchmark_data(input_file, initial_searched_results_dir)
    
    # Convert to dataset and save as parquet
    dataset = datasets.Dataset.from_list(processed_data)
    dataset.to_parquet(output_file)
    
    # Print statistics
    print(f"Total number of questions: {len(dataset)}")
    for data_source in dataset.unique('data_source'):
        count = len(dataset.filter(lambda x: x['data_source'] == data_source))
        print(f"{data_source}: {count}")

if __name__ == "__main__":
    main() 