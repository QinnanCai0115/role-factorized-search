import json
from verl.utils.reward_score.rag_2 import check_answer_correct, em_check

# Load test results
with open("/home/pj20/server-04/search-c1/results/test_results_sft.json", "r") as f:
    results = json.load(f)

# Initialize statistics
stats = {}
total_correct = 0
total_em = 0
total_questions = 0

# Process each data source
for data_source, questions in results.items():
    correct = 0
    em_correct = 0
    for question, data in questions.items():
        model_output = data['model_output']
        golden_answers = data['golden_answers']
        
        # Remove "assistant\n" prefix if present
        if model_output.startswith("assistant\n"):
            model_output = model_output[len("assistant\n"):]
        
        # Check correctness and EM
        if check_answer_correct(answer=model_output, golden_answers=golden_answers):
            correct += 1
            total_correct += 1
        if em_check(prediction=model_output, golden_answers=golden_answers):
            em_correct += 1
            total_em += 1
        total_questions += 1
    
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
output_file = "/home/pj20/server-04/search-c1/results/test_results_sft_stats.json"
with open(output_file, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"\nResults saved to: {output_file}")



