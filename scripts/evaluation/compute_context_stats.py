import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import logging
from typing import Dict, List
import numpy as np

def setup_logger(log_file):
    logger = logging.getLogger('context_stats')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def compute_stats_for_field(data: Dict, field_name: str, tokenizer) -> Dict:
    """Compute statistics for a specific field"""
    stats = {
        'total_samples': 0,
        'total_tokens': 0,
        'token_lengths': []
    }
    
    for question, sample_data in data.items():
        if field_name in sample_data:
            content = sample_data[field_name]
            # Tokenize the content
            tokens = tokenizer.encode(content)
            token_count = len(tokens)
            
            # Update statistics
            stats['total_samples'] += 1
            stats['total_tokens'] += token_count
            stats['token_lengths'].append(token_count)
    
    # Compute averages and percentiles if we have samples
    if stats['total_samples'] > 0:
        stats['avg_tokens_per_sample'] = stats['total_tokens'] / stats['total_samples']
        stats['token_length_percentiles'] = {
            'p50': np.percentile(stats['token_lengths'], 50),
            'p90': np.percentile(stats['token_lengths'], 90),
            'p95': np.percentile(stats['token_lengths'], 95),
            'p99': np.percentile(stats['token_lengths'], 99)
        }
    
    return stats

def compute_context_stats(context_dir: str, logger) -> Dict:
    """Compute token statistics for all context files in the directory"""
    # Initialize tokenizer
    TOKENIZER_MODEL = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    
    # Initialize statistics
    stats = {
        'context_with_info': {
            'total_samples': 0,
            'total_tokens': 0,
            'token_lengths': [],
            'per_source_stats': {}
        },
        'response_str': {
            'total_samples': 0,
            'total_tokens': 0,
            'token_lengths': [],
            'per_source_stats': {}
        }
    }
    
    # Get all context files
    context_files = [f for f in os.listdir(context_dir) if f.endswith('_output_sequences.json')]
    
    for context_file in tqdm(context_files, desc="Processing context files"):
        source_name = context_file.replace('_output_sequences.json', '')
        file_path = os.path.join(context_dir, context_file)
        
        # Initialize per-source statistics for both fields
        for field in ['context_with_info', 'response_str']:
            stats[field]['per_source_stats'][source_name] = {
                'total_samples': 0,
                'total_tokens': 0,
                'token_lengths': []
            }
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Compute statistics for each field
            for field in ['context_with_info', 'response_str']:
                field_stats = compute_stats_for_field(data, field, tokenizer)
                
                # Update global statistics
                stats[field]['total_samples'] += field_stats['total_samples']
                stats[field]['total_tokens'] += field_stats['total_tokens']
                stats[field]['token_lengths'].extend(field_stats['token_lengths'])
                
                # Update per-source statistics
                stats[field]['per_source_stats'][source_name] = field_stats
                
        except Exception as e:
            logger.error(f"Error processing {context_file}: {str(e)}")
    
    # Compute final averages and percentiles for global statistics
    for field in ['context_with_info', 'response_str']:
        if stats[field]['total_samples'] > 0:
            stats[field]['avg_tokens_per_sample'] = stats[field]['total_tokens'] / stats[field]['total_samples']
            stats[field]['token_length_percentiles'] = {
                'p50': np.percentile(stats[field]['token_lengths'], 50),
                'p90': np.percentile(stats[field]['token_lengths'], 90),
                'p95': np.percentile(stats[field]['token_lengths'], 95),
                'p99': np.percentile(stats[field]['token_lengths'], 99)
            }
    
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context_dir', required=True, help='Directory containing context files')
    parser.add_argument('--output_file', required=True, help='Path to save statistics JSON file')
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.output_file.replace('.json', '.log')
    logger = setup_logger(log_file)
    
    logger.info(f"Starting context statistics computation for directory: {args.context_dir}")
    
    # Compute statistics
    stats = compute_context_stats(args.context_dir, logger)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Log summary for each field
    for field in ['context_with_info', 'response_str']:
        logger.info(f"\n=== Statistics for {field} ===")
        logger.info(f"Total samples processed: {stats[field]['total_samples']}")
        logger.info(f"Average tokens per sample: {stats[field]['avg_tokens_per_sample']:.2f}")
        logger.info("\nToken length percentiles:")
        for percentile, value in stats[field]['token_length_percentiles'].items():
            logger.info(f"{percentile}: {value:.2f}")
        
        logger.info("\nPer-source statistics:")
        for source, source_stats in stats[field]['per_source_stats'].items():
            logger.info(f"\n{source}:")
            logger.info(f"  Samples: {source_stats['total_samples']}")
            logger.info(f"  Average tokens: {source_stats['avg_tokens_per_sample']:.2f}")
            logger.info("  Percentiles:")
            for percentile, value in source_stats['token_length_percentiles'].items():
                logger.info(f"    {percentile}: {value:.2f}")
    
    logger.info(f"\nDetailed statistics saved to: {args.output_file}")
    logger.info(f"Log file saved to: {log_file}")

if __name__ == '__main__':
    main() 