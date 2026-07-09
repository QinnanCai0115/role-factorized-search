#!/usr/bin/env python3

import pandas as pd
import requests
import json
import argparse
import os

def search(query: str, endpoint: str):
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

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])

def main():
    parser = argparse.ArgumentParser(description="Run retrieval and save JSON outputs.")
    parser.add_argument("--input_parquet", required=True, help="Input .parquet file with QA data.")
    parser.add_argument("--output_dir", required=True, help="Directory to store output JSON files.")
    parser.add_argument("--endpoint", required=True, help="Retrieval API endpoint URL (e.g., http://127.0.0.1:8000/retrieve)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)

    data_sources = ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']

    for data_source in data_sources:
        print(f"[INFO] Processing: {data_source}")
        retrieval_info = {}
        qa_data = df[df['data_source'] == data_source]

        for index, row in qa_data.iterrows():
            q = row['question']
            golden_answers = list(row['golden_answers'])
            retrieval_result = search(q, args.endpoint)
            question_info = {
                'golden_answers': golden_answers,
                'context_with_info': retrieval_result
            }
            retrieval_info[q] = question_info

        out_path = os.path.join(args.output_dir, f"{data_source}_output_sequences.json")
        with open(out_path, 'w') as f:
            json.dump(retrieval_info, f, indent=4)
        print(f"[INFO] Saved: {out_path}")

if __name__ == "__main__":
    main()
