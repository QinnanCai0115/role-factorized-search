import os
import random
import pandas as pd
import json
from verl.utils.reward_score.rag_2 import generate_answer_zero_shot, check_answer_correct
from verl.utils.hdfs_io import copy, makedirs
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from datetime import datetime

def load_previous_results(zeroshot_cache_file):
    """Load previous results if they exist"""
    print(f"Checking for previous results at: {zeroshot_cache_file}")
    if os.path.exists(zeroshot_cache_file):
        print(f"Loading previous results from {zeroshot_cache_file}")
        with open(zeroshot_cache_file, 'r') as f:
            return json.load(f)
    print("No previous results found")
    return {}

def save_results(zeroshot_answers, zeroshot_cache_file, stats_file, total_questions, correct_zeroshot):
    """Save results and statistics"""
    try:
        # Ensure directories exist
        cache_dir = os.path.dirname(zeroshot_cache_file)
        stats_dir = os.path.dirname(stats_file)
        
        print(f"Creating directories if needed:")
        print(f"Cache directory: {cache_dir}")
        print(f"Stats directory: {stats_dir}")
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        
        # Save zeroshot answers
        print(f"Saving zeroshot answers to: {zeroshot_cache_file}")
        with open(zeroshot_cache_file, 'w') as f:
            json.dump(zeroshot_answers, f)
        
        # Save statistics
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_questions': total_questions,
            'processed_questions': len(zeroshot_answers),
            'correct_zeroshot': correct_zeroshot,
            'accuracy': correct_zeroshot / len(zeroshot_answers) if zeroshot_answers else 0,
            'remaining_questions': total_questions - len(zeroshot_answers)
        }
        
        print(f"Saving statistics to: {stats_file}")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print("Save completed successfully")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        raise

def process_question(row, lock):
    try:
        # Extract question and golden answers from ground_truth
        question = row['reward_model']['ground_truth']['question']
        golden_answers = row['reward_model']['ground_truth']['target']
        
        # Generate zeroshot answer
        zeroshot_answer = generate_answer_zero_shot(prompt=question)
        
        # Check if answer is correct
        is_correct = check_answer_correct(answer=zeroshot_answer, golden_answers=golden_answers)
        
        return question, zeroshot_answer, is_correct
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return None, None, None

def process_dataset(input_file, output_file, zeroshot_cache_file, hdfs_dir=None, num_workers=16):
    # Load the dataset
    print(f"Loading dataset from {input_file}")
    df = pd.read_parquet(input_file)
    
    # Initialize counters and shared data structures
    total_questions = len(df)
    correct_zeroshot = 0
    processed_questions = 0
    lock = threading.Lock()
    
    # Load previous results
    zeroshot_answers = load_previous_results(zeroshot_cache_file)
    correct_zeroshot = sum(1 for info in zeroshot_answers.values() if info['score'] == 1)
    processed_questions = len(zeroshot_answers)
    
    # Filter out already processed questions
    processed_questions_set = set(zeroshot_answers.keys())
    remaining_df = df[~df['reward_model'].apply(lambda x: x['ground_truth']['question']).isin(processed_questions_set)]
    
    print(f"Found {processed_questions} previously processed questions")
    print(f"Remaining questions to process: {len(remaining_df)}")
    
    # Create stats file path
    stats_file = os.path.join(os.path.dirname(zeroshot_cache_file), 'zeroshot_stats.json')
    print(f"Stats file will be saved to: {stats_file}")
    
    # Process remaining questions in parallel
    print(f"Processing remaining questions with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_question, row, lock): idx for idx, row in remaining_df.iterrows()}
        
        # Process results as they complete
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                question, zeroshot_answer, is_correct = future.result()
                
                if question is not None:  # Skip failed results
                    # Update counters and store results
                    with lock:
                        if is_correct:
                            correct_zeroshot += 1
                        
                        # Store results in the same format as used by RewardManager
                        zeroshot_answers[question] = {
                            'answer': zeroshot_answer,
                            'score': 1 if is_correct else 0
                        }
                        
                        processed_questions += 1
                        
                        # Save intermediate results periodically
                        if processed_questions % 5000 == 0:
                            print(f"\nSaving intermediate results after processing {processed_questions} questions")
                            save_results(zeroshot_answers, zeroshot_cache_file, stats_file, total_questions, correct_zeroshot)
                
                pbar.update(1)
    
    # Save final results
    print("\nSaving final results")
    save_results(zeroshot_answers, zeroshot_cache_file, stats_file, total_questions, correct_zeroshot)
    
    # Calculate and print final statistics
    accuracy = correct_zeroshot / total_questions
    print(f"\nFinal Statistics:")
    print(f"Total questions: {total_questions}")
    print(f"Correct zeroshot answers: {correct_zeroshot}")
    print(f"Zeroshot accuracy: {accuracy:.2%}")
    print(f"Results saved to: {zeroshot_cache_file}")
    print(f"Statistics saved to: {stats_file}")
    
    # Filter dataset
    correct_questions = [q for q, info in zeroshot_answers.items() if info['score'] == 1]
    num_to_remove = len(correct_questions) // 2  # Remove 50% of correct zeroshot questions
    questions_to_remove = random.sample(correct_questions, num_to_remove)
    
    # Create filtered dataset
    filtered_df = df[~df['reward_model'].apply(lambda x: x['ground_truth']['question']).isin(questions_to_remove)]
    
    # Save filtered dataset
    print(f"\nSaving filtered dataset to: {output_file}")
    filtered_df.to_parquet(output_file)
    print(f"Removed {num_to_remove} questions that were correctly answered in zeroshot")
    
    # Copy to HDFS if specified
    if hdfs_dir:
        print(f"\nCopying to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=os.path.dirname(output_file), dst=hdfs_dir)
        print(f"Copied results to HDFS: {hdfs_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="data/nq_hotpotqa_train/train_e5_ug.parquet", help='Path to input parquet file')
    parser.add_argument('--output_file', default="data/nq_hotpotqa_train/train_e5_ug_filtered.parquet", help='Path to save filtered parquet file')
    parser.add_argument('--zeroshot_cache_file', default="data/Qwen_Qwen2.5-14B-Instruct-GPTQ-Int4-3b/zeroshot_answers_.json", help='Path to save zeroshot answers JSON file')
    parser.add_argument('--hdfs_dir', default=None, help='Optional HDFS directory to copy results to')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of worker threads to use')
    
    args = parser.parse_args()
    process_dataset(args.input_file, args.output_file, args.zeroshot_cache_file, args.hdfs_dir, args.num_workers) 