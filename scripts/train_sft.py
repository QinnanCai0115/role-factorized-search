import os
import torch
import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
import logging
from datetime import datetime
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,7'

# Model configuration
model_name = 'Qwen/Qwen2.5-3B-Instruct'
output_dir = "sft_model"

# Custom JSON encoder to handle NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Load and process data
def load_and_process_data(train_file, test_file):
    logger.info(f"Loading data from {train_file} and {test_file}")
    # Load train data
    train_df = pd.read_parquet(train_file)
    logger.info(f"Loaded {len(train_df)} training examples")
    
    # Process train data
    train_data = []
    for _, row in train_df.iterrows():
        question = row['reward_model']['ground_truth']['question']
        answer = row['reward_model']['ground_truth']['target'][0]  # Get first answer from golden_answers
        
        # Create chat format
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        train_data.append(messages)
    
    # Create train dataset
    train_dataset = Dataset.from_dict({
        "messages": train_data
    })
    
    # Load test data
    test_df = pd.read_parquet(test_file)
    logger.info(f"Loaded {len(test_df)} test examples")
    
    # Process test data
    test_data = []
    test_metadata = []  # Store additional metadata for test data
    for _, row in test_df.iterrows():
        question = row['reward_model']['ground_truth']['question']
        answer = row['reward_model']['ground_truth']['target'][0]  # Get first answer from golden_answers
        
        # Create chat format
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        test_data.append(messages)
        
        # Store metadata
        test_metadata.append({
            'data_source': row['data_source'],
            'question': question,
            'golden_answers': row['reward_model']['ground_truth']['target']
        })
    
    # Create test dataset
    test_dataset = Dataset.from_dict({
        "messages": test_data
    })
    
    return train_dataset, test_dataset, test_metadata

# def get_latest_checkpoint(output_dir):
#     checkpoint_dirs = [d for d in os.listdir(output_dir) if re.match(r"checkpoint-\\d+", d)]
#     if not checkpoint_dirs:
#         return None
#     latest = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
#     return os.path.join(output_dir, latest)

def generate_predictions(model, tokenizer, test_dataset, test_metadata, save_every=10, save_path='results/test_results_partial.json'):
    logger.info("Generating predictions for test set...")
    model.eval()
    results = {}
    
    # Group by data source
    for idx, metadata in enumerate(tqdm(test_metadata, desc="Generating predictions")):
        data_source = metadata['data_source']
        question = metadata['question']
        golden_answers = metadata['golden_answers']
        
        # Convert golden_answers to list if it's a numpy array
        if isinstance(golden_answers, np.ndarray):
            golden_answers = golden_answers.tolist()
        
        # Prepare input
        messages = [{"role": "user", "content": question}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Store results
        if data_source not in results:
            results[data_source] = {}
        
        results[data_source][question] = {
            'model_output': prediction,
            'golden_answers': golden_answers
        }
        # Save every N steps
        if (idx + 1) % save_every == 0:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved intermediate results after {idx+1} predictions to {save_path}")
    
    return results

def main():
    logger.info("Starting training process...")
    
    # Load and process data
    train_dataset, test_dataset, test_metadata = load_and_process_data(
        train_file='data/nq_hotpotqa_train/train_e5_u1.parquet',
        test_file='data/nq_hotpotqa_train/test_e5_u1.parquet'
    )
    
    # Load model and tokenizer
    logger.info(f"Loading model {model_name}")
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map='auto',
    )
    
    # Load from latest checkpoint
    # checkpoint_path = get_latest_checkpoint(output_dir)
    # if checkpoint_path:
    #     logger.info(f"Loading from checkpoint: {checkpoint_path}")
    #     model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    # else:
    logger.info(f"No checkpoint found, loading from base model")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Training configuration
    logger.info("Setting up training configuration")
    training_args = SFTConfig(
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        save_strategy='epoch',
        output_dir=output_dir,
        logging_steps=10,  # Log every 10 steps
        logging_dir='logs',  # Directory for storing logs
        report_to=None,  # Disable tensorboard logging
    )
    
    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    
    # Generate predictions for test set
    logger.info("Generating predictions for test set...")
    os.makedirs('results', exist_ok=True)
    test_results = generate_predictions(model, tokenizer, test_dataset, test_metadata, save_every=10, save_path='results/test_results_partial.json')
    
    # Save test results
    results_file = 'results/test_results.json'
    logger.info(f"Saving test results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main() 