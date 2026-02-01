import json

def combine_qa_pairs():
    # Load the original test file
    with open('tofu_test.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Load the paraphrased file
    with open('tofu_test_paraphrased.json', 'r', encoding='utf-8') as f:
        paraphrased_data = json.load(f)
    
    # Create a dictionary for quick lookup by original_index from original data
    original_lookup = {item['original_index']: item for item in original_data}
    
    # Combine the data: paraphrased questions with original answers
    combined_data = []
    
    for para_item in paraphrased_data:
        original_index = para_item['original_index']
        
        # Find matching item in original data
        if original_index in original_lookup:
            original_item = original_lookup[original_index]
            
            # Create new combined item
            combined_item = {
                'question': para_item['question'],  # Use paraphrased question
                'answer': original_item['answer'],   # Use original answer
                'combined_text': f"{para_item['question']} {original_item['answer']}",
                'author': original_item['author'],
                'author_index': original_item['author_index'],
                'original_index': original_index,
                'split': original_item['split']
            }
            
            combined_data.append(combined_item)
        else:
            print(f"Warning: No matching original_index {original_index} found in original data")
    
    # Save the combined data to a new file
    output_filename = 'tofu_test_question_paraphrased.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"Combined {len(combined_data)} QA pairs")
    print(f"Output saved to: {output_filename}")

if __name__ == "__main__":
    combine_qa_pairs()