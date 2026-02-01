import json

# Load the generated answers file
# with open('knn_lm_answers_100k_A5000_question_paraphrased_finetuned_embedding_weird_word.json', 'r') as f:
#     generated_data = json.load(f)
with open('knn_lm_answers_100k_A5000.json', 'r') as f:
    generated_data = json.load(f)

# # Load the ground-truth answers file
# with open('../../../dataset/private/syn_traj/scalability_dataset/test_100k_paraphrased.json', 'r') as f:
#     ground_truth_data = json.load(f)

# Load the ground-truth answers file
with open('../../../dataset/private/syn_traj/scalability_dataset/test_100k.json', 'r') as f:
    ground_truth_data = json.load(f)

# Create a dictionary for quick lookup of ground-truth answers by question
ground_truth_dict = {item['question']: item['answer'] for item in ground_truth_data}

# Create the combined result
result = []
for item in generated_data:
    question = item['question']
    generated_answer = item['answer']
    
    # Get the ground-truth answer for this question
    ground_truth_answer = ground_truth_dict.get(question, "NOT FOUND")
    
    result.append({
        'question': question,
        'generated_answer': generated_answer,
        'ground_truth_answer': ground_truth_answer
    })

# Save the result to a new JSON file in the current directory
# output_file = 'combined_answers_finetuned_embedding_weird_word.json'
output_file = 'combined_answers_normal_question_pretrained_embedding.json'

with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"Successfully created {output_file} with {len(result)} entries")
print(f"Sample entry:")
if result:
    print(json.dumps(result[0], indent=2))