import os
import argparse
import pandas as pd
from tqdm import tqdm

def track_progress(current_question, parquet_path):
    """Track progress based on current question in the parquet file"""
    # Load the parquet file
    df = pd.read_parquet(parquet_path)
    total_questions = len(df)
    
    # Find the position of the current question
    current_position = None
    for idx, row in df.iterrows():
        # The question is in the prompt field of the first message
        question = row['prompt'][0]['content']
        if current_question.strip() in question:
            current_position = idx
            break
    
    if current_position is None:
        print("Warning: Current question not found in the parquet file")
        return
    
    progress = ((current_position + 1) / total_questions) * 100
    print(f"\nProgress: {progress:.2f}% ({current_position + 1}/{total_questions} questions)")
    print(f"Remaining questions: {total_questions - (current_position + 1)}")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--current_question', required=True, help='how long is a prime minister term in uk?')
    # parser.add_argument('--parquet_path', required=True, help='Path to the test_e5_ug.parquet file')
    
    # args = parser.parse_args()
    
    current_question = 'what percentage of sunlight is captured by plants to convert it into food energy?'
    parquet_path = '/home/pj20/server-04/search-c1/data/nq_hotpotqa_train/test_e5_ug.parquet'
    
    track_progress(current_question, parquet_path) 