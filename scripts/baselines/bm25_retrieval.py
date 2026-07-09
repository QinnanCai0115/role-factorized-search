#!/usr/bin/env python3

import pandas as pd
import requests
import json
import argparse
import os
import sys
sys.path.append('./')
from generator_llms.query_rewrite import rewrite_query
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def search(query: str, endpoint: str, input_parquet: str):
    payload = {
        "queries": [query],
        "topk": 12,
        "return_scores": True
    }
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        results = response.json()['result']
    except Exception as e:
        print(f"[ERROR] Retrieval failed for query: {query}\n{e}")
        return ""

    def _passages2string(retrieval_result, input_parquet: str):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            # if "mirage" in input_parquet:
            #     if "." in content:
            #         title = content.split(".")[0]
            #         text = content.split(".")[1]
            #     else:
            #         title = content.split("\n")[0]
            #         text = "\n".join(content.split("\n")[1:])
            # else:
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0], input_parquet)

def process_question(row, rewriter, endpoint, dataset):
    # q = row['question']
    # golden_answers = list(row['golden_answers'])
    q = row['reward_model']['ground_truth']['question']
    golden_answers = row['reward_model']['ground_truth']['target'].tolist()  # Convert numpy array to Python list
    
    if rewriter == "none":
        rewritten_query = q
    elif rewriter == "triviaqa":
        rewritten_query = rewrite_query(q, "triviaqa")
    elif rewriter == "nq":
        rewritten_query = rewrite_query(q, "nq")
    elif rewriter == "squad":
        rewritten_query = rewrite_query(q, "squad")
        
    if 'mirage' in dataset:
        rewritten_query = rewritten_query.split('\nOptions:')[0]
        # print(rewritten_query)
    
    retrieval_result = search(rewritten_query, endpoint, dataset)
    return q, {
        'golden_answers': golden_answers,
        'context_with_info': retrieval_result
    }

def main():
    parser = argparse.ArgumentParser(description="Run retrieval and save JSON outputs.")
    parser.add_argument("--input_parquet", required=True, help="Input .parquet file with QA data.")
    parser.add_argument("--rewriter", required=True, help="Path to the rewriter model.")
    parser.add_argument("--output_dir", required=True, default="data/BM25", help="Directory to store output JSON files.")
    parser.add_argument("--endpoint", required=True, help="Retrieval API endpoint URL (e.g., http://127.0.0.1:3000/retrieve)")
    parser.add_argument("--num_workers", type=int, default=20, help="Number of worker threads for parallel processing")
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.rewriter+"_deepretrieval")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    # data_sources = ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']
    # data_sources = ['nq', 'hotpotqa']
    data_sources = ['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu']
    
    for data_source in data_sources:
        print(f"[INFO] Processing: {data_source}")
        retrieval_info = {}
        qa_data = df[df['data_source'] == data_source]

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for _, row in qa_data.iterrows():
                future = executor.submit(process_question, row, args.rewriter, args.endpoint, args.input_parquet)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {data_source}"):
                try:
                    q, question_info = future.result()
                    retrieval_info[q] = question_info
                except Exception as e:
                    print(f"[ERROR] Failed to process question: {e}")

        out_path = os.path.join(args.output_dir, f"{data_source}_output_sequences.json")
        with open(out_path, 'w') as f:
            json.dump(retrieval_info, f, indent=4)
        print(f"[INFO] Saved: {out_path}")

if __name__ == "__main__":
    main()
