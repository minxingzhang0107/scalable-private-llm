import json

def process_qa_files(original_file, ground_truth_file, reference_file, output_file):
    """
    Process QA pairs from multiple JSON files and create a consolidated output.
    
    Args:
        original_file (str): Path to the original JSON file
        ground_truth_file (str): Path to the ground truth JSON file
        reference_file (str): Path to the reference JSON file
        output_file (str): Path for the output JSON file
    """
    
    # Load the original JSON file
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Load the ground truth JSON file
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    
    # Load the reference JSON file
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
    
    # Create dictionaries for quick lookup by original_index
    ground_truth_dict = {item['original_index']: item['answer'] for item in ground_truth_data}
    reference_dict = {item['original_index']: item['answer'] for item in reference_data}
    
    # Process each QA pair from the original file
    processed_data = []
    
    for qa_pair in original_data:
        original_index = qa_pair['original_index']
        
        # Create the new QA pair structure
        new_qa_pair = {
            'question': qa_pair['question'],
            'generated_answer_after_removal': qa_pair['answer'],
            'ground_truth_answer': ground_truth_dict.get(original_index, ''),
            'generated_answer_pretrained_LM_only': reference_dict.get(original_index, '')
        }
        
        processed_data.append(new_qa_pair)
    
    # Save the processed data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed! Output saved to {output_file}")
    print(f"Processed {len(processed_data)} QA pairs")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    original_file = "weird_word_replacement_embedding/author_removal_20pct_removed_authors_results.json"
    ground_truth_file = "../dataset/private/tofu/tofu_test_question_paraphrased.json"
    reference_file = "../results/private/tofu/lm_only/pretrained_lm_only_generated_answers.json"
    output_file = "weird_word_replacement_embedding/author_removal_20pct_removed_authors_privacy_quality_comparison_results.json"
    
    process_qa_files(original_file, ground_truth_file, reference_file, output_file)