import json
import os
import random
from verl.utils.reward_score.rag_2 import check_answer_correct, em_check
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

def load_context_cache(context_dir: str, data_sources: list) -> dict:
    """Load all context files into memory"""
    cache = {}
    for source in data_sources:
        context_file = os.path.join(context_dir, f"{source}_output_sequences.json")
        if os.path.exists(context_file):
            print(f"Loading context file: {context_file}")
            with open(context_file, 'r') as f:
                cache[source] = json.load(f)
        else:
            print(f"Warning: Context file not found: {context_file}")
    return cache

def extract_answer_from_response(response_str: str) -> str:
    """Extract answer from response string within <answer> tags"""
    if not response_str:
        return ""
    
    # Find content between <answer> tags
    start_tag = "<answer>"
    end_tag = "</answer>"
    
    start_idx = response_str.find(start_tag)
    if start_idx == -1:
        return response_str.strip()  # Return full response if no tags found
    
    start_idx += len(start_tag)
    end_idx = response_str.find(end_tag, start_idx)
    
    if end_idx == -1:
        return response_str[start_idx:].strip()  # Return everything after start tag if no end tag
    
    return response_str[start_idx:end_idx].strip()

def process_questions_batch(questions_batch: list) -> dict:
    """Process a batch of questions"""
    results = {
        'correct': 0,
        'em_correct': 0,
        'total': len(questions_batch)
    }
    
    for question, data in questions_batch:
        # Extract answer from response_str
        model_output = extract_answer_from_response(data.get('response_str', ''))
        golden_answers = data.get('golden_answers', [])
        
        if not isinstance(golden_answers, list):
            golden_answers = [golden_answers]
        
        # Check correctness and EM
        if check_answer_correct(answer=model_output, golden_answers=golden_answers):
            results['correct'] += 1
        if em_check(prediction=model_output, golden_answers=golden_answers):
            results['em_correct'] += 1
    
    return results

def evaluate_results(context_dir: str, num_workers: int = 16, random_seed: int = 42, sampling_enabled: bool = False):
    # Set random seed
    random.seed(random_seed)
    
    # Get list of data sources from directory
    data_sources = [f.split('_output_sequences.json')[0] for f in os.listdir(context_dir) 
                   if f.endswith('_output_sequences.json')]
    
    # Load context cache
    context_cache = load_context_cache(context_dir, data_sources)
    
    # Initialize statistics
    stats = {}
    total_correct = 0
    total_em = 0
    total_questions = 0
    
    # Process each data source
    for data_source, questions in context_cache.items():
        print(f"\nEvaluating {data_source}")
        
        # Apply sampling if enabled
        if sampling_enabled and len(questions) > 3000:
            print(f"Sampling 3000 questions from {data_source} (total: {len(questions)})")
            questions = dict(random.sample(list(questions.items()), 3000))
        else:
            print(f"Using all {len(questions)} questions from {data_source}")
        
        # Convert questions to list of tuples for processing
        questions_list = list(questions.items())
        
        # Process in batches
        batch_size = 32
        batches = [questions_list[i:i + batch_size] for i in range(0, len(questions_list), batch_size)]
        
        correct = 0
        em_correct = 0
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_questions_batch, batch): i for i, batch in enumerate(batches)}
            
            for future in tqdm(as_completed(futures), total=len(batches), desc=f"Processing {data_source}"):
                batch_results = future.result()
                correct += batch_results['correct']
                em_correct += batch_results['em_correct']
        
        # Calculate accuracy and EM for this data source
        accuracy = correct / len(questions) if questions else 0
        em_accuracy = em_correct / len(questions) if questions else 0
        
        stats[data_source] = {
            'total': len(questions),
            'correct': correct,
            'accuracy': accuracy,
            'em_correct': em_correct,
            'em_accuracy': em_accuracy
        }
        
        total_correct += correct
        total_em += em_correct
        total_questions += len(questions)
        print(f"Total correct: {correct}, Total em correct: {em_correct}, Total questions: {len(questions)}")
    
    # Calculate overall accuracy and EM
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    overall_em = total_em / total_questions if total_questions > 0 else 0
    stats['overall'] = {
        'total': total_questions,
        'correct': total_correct,
        'accuracy': overall_accuracy,
        'em_correct': total_em,
        'em_accuracy': overall_em
    }
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for source, source_stats in stats.items():
        print(f"\n{source}:")
        print(f"Total questions: {source_stats['total']}")
        print(f"Correct answers: {source_stats['correct']} ({source_stats['accuracy']:.2%})")
        print(f"EM correct answers: {source_stats['em_correct']} ({source_stats['em_accuracy']:.2%})")
    
    # Save results
    output_file = os.path.join(os.path.dirname(context_dir), "results", "r1_no_search_eval_stats.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--context_dir', default="/home/pj20/server-04/search-c1/data/output_sequences_r1_3b",
    #                   help='Directory containing context files')
    parser.add_argument('--context_dir', default="data/output_sequences_r1_7b_mirage")
    parser.add_argument('--num_workers', type=int, default=16,
                      help='Number of worker processes to use')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducible sampling')
    parser.add_argument('--sampling_enabled', action='store_true',
                      help='Enable sampling of questions')
    
    args = parser.parse_args()
    evaluate_results(args.context_dir, args.num_workers, args.random_seed, False)



