import json
import pandas as pd
import random

def main():
    # Read the JSON file
    with open('rag_e5_top12_haiku.json', 'r') as f:
        rag_results = json.load(f)
    
    # Read the parquet file
    df = pd.read_parquet('data/nq_hotpotqa_train/test_r1_v6.parquet')
    
    # Create a mapping of questions to golden answers
    question_to_golden = {}
    for _, row in df.iterrows():
        question = row['reward_model']['ground_truth']['question']
        golden_answer = row['reward_model']['ground_truth']['target']
        question_to_golden[question] = golden_answer
    
    # Flatten the RAG results
    all_samples = []
    for dataset, questions in rag_results.items():
        for question, result in questions.items():
            all_samples.append({
                'dataset': dataset,
                'question': question,
                'rag_answer': result['answer'],
                'rag_score': result['score'],
                'rag_em_score': result['em_score'],
                'golden_answer': question_to_golden.get(question, 'Not found')
            })
    
    # Randomly select 1000 samples
    selected_samples = random.sample(all_samples, min(1000, len(all_samples)), random_state=528)
    
    # Save the results
    output_df = pd.DataFrame(selected_samples)
    output_df.to_csv('sampled_rag_results.csv', index=False)
    
    # Print some statistics
    print(f"Total samples processed: {len(all_samples)}")
    print(f"Selected samples: {len(selected_samples)}")
    print("\nDataset distribution:")
    print(output_df['dataset'].value_counts())
    
    # Print some example samples
    print("\nExample samples:")
    for i, row in output_df.head(3).iterrows():
        print(f"\nSample {i+1}:")
        print(f"Question: {row['question']}")
        print(f"RAG Answer: {row['rag_answer']}")
        print(f"Golden Answer: {row['golden_answer']}")
        print(f"RAG Score: {row['rag_score']}")
        print(f"EM Score: {row['rag_em_score']}")

if __name__ == '__main__':
    main() 