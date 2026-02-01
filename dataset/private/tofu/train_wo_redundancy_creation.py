import json
import random

def replace_questions_in_train_file(train_file_path, extended_file_path, output_file_path):
    """
    Creates a new JSON file by replacing questions from the extended file
    while keeping everything else the same from the original train file.
    
    Args:
        train_file_path: Path to the original tofu_train.json file
        extended_file_path: Path to the tofu_train_extended.json file  
        output_file_path: Path where the new JSON file will be saved
    """
    
    # Read the original train file
    with open(train_file_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Read the extended file
    with open(extended_file_path, 'r', encoding='utf-8') as f:
        extended_data = json.load(f)
    
    # Create a mapping from original_index to questions in the extended file
    # Since there can be multiple entries with the same original_index, 
    # we'll store them in a list and pick the first one (or random)
    extended_index_to_questions = {}
    
    for entry in extended_data:
        original_index = entry['original_index']
        if original_index not in extended_index_to_questions:
            extended_index_to_questions[original_index] = []
        extended_index_to_questions[original_index].append(entry['question'])
    
    print(f"Loaded {len(train_data)} entries from train file")
    print(f"Loaded {len(extended_data)} entries from extended file")
    print(f"Found {len(extended_index_to_questions)} unique original_indexes in extended file")
    
    # Create new data by modifying the train data
    new_data = []
    matched_count = 0
    
    for entry in train_data:
        # Create a copy of the original entry
        new_entry = entry.copy()
        
        original_index = entry['original_index']
        
        # Check if we have a replacement question for this original_index
        if original_index in extended_index_to_questions:
            # Get the new question (pick the first one from the list)
            # You can change this to random.choice() if you want random selection
            new_question = extended_index_to_questions[original_index][0]
            
            # Update the question
            new_entry['question'] = new_question
            
            # Update combined_text to be NEW question + OLD answer
            old_answer = entry['answer']
            new_entry['combined_text'] = f"{new_question} {old_answer}"
            
            matched_count += 1
        else:
            print(f"Warning: No matching original_index {original_index} found in extended file")
        
        new_data.append(new_entry)
    
    print(f"Successfully matched and replaced {matched_count} out of {len(train_data)} entries")
    
    # Save the new data to output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"New JSON file saved to: {output_file_path}")
    
    return new_data

def replace_questions_random_selection(train_file_path, extended_file_path, output_file_path):
    """
    Same as above but randomly selects from multiple questions with the same original_index
    """
    
    # Read the files
    with open(train_file_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(extended_file_path, 'r', encoding='utf-8') as f:
        extended_data = json.load(f)
    
    # Create mapping
    extended_index_to_questions = {}
    for entry in extended_data:
        original_index = entry['original_index']
        if original_index not in extended_index_to_questions:
            extended_index_to_questions[original_index] = []
        extended_index_to_questions[original_index].append(entry['question'])
    
    # Process entries
    new_data = []
    matched_count = 0
    
    for entry in train_data:
        new_entry = entry.copy()
        original_index = entry['original_index']
        
        if original_index in extended_index_to_questions:
            # Randomly select a question from available options
            new_question = random.choice(extended_index_to_questions[original_index])
            
            new_entry['question'] = new_question
            new_entry['combined_text'] = f"{new_question} {entry['answer']}"
            matched_count += 1
        
        new_data.append(new_entry)
    
    # Save results
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"Randomly selected and replaced {matched_count} questions")
    return new_data

# Example usage
if __name__ == "__main__":
    # Use the first function (picks first question when multiple exist)
    replace_questions_in_train_file(
        train_file_path='tofu_original.json',
        extended_file_path='tofu_train_w_redundancy.json', 
        output_file_path='tofu_train_wo_redundancy.json'
    )
    
    # Or use the second function (randomly picks when multiple exist)
    # replace_questions_random_selection(
    #     train_file_path='tofu_train.json',
    #     extended_file_path='tofu_train_extended.json',
    #     output_file_path='tofu_train_new_random.json'
    # )