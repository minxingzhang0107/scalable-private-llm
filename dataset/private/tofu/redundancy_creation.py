import json
import random
from typing import List, Dict, Any
from collections import defaultdict


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Load JSON data from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: List[Dict[str, Any]], filepath: str):
    """Save JSON data to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def group_by_original_index(data: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group QA pairs by their original_index."""
    grouped = defaultdict(list)
    for item in data:
        grouped[item['original_index']].append(item)
    return grouped


def get_unique_questions_and_answers(items: List[Dict[str, Any]]) -> tuple:
    """Get unique questions and answers from a list of items."""
    unique_questions = list(set(item['question'] for item in items))
    unique_answers = list(set(item['answer'] for item in items))
    return unique_questions, unique_answers


def process_original_index(original_index: int, 
                           items_file1: List[Dict[str, Any]], 
                           item_file2: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a single original_index to create QA pairs.
    
    Args:
        original_index: The original_index to process
        items_file1: List of items from file1 with this original_index
        item_file2: The single item from file2 with this original_index
    
    Returns:
        List of QA pairs (can be 4, 6, or 9 depending on unique questions/answers)
    """
    # Get unique questions and answers from file1
    unique_questions, unique_answers = get_unique_questions_and_answers(items_file1)
    
    # Get question and answer from file2
    question_from_file2 = item_file2['question']
    answer_from_file2 = item_file2['answer']
    
    # Process questions
    if len(unique_questions) == 3:
        # If file2's question is already in the pool, remove a different one
        if question_from_file2 in unique_questions:
            # Remove one that's not from file2
            other_questions = [q for q in unique_questions if q != question_from_file2]
            if other_questions:
                unique_questions.remove(random.choice(other_questions))
        else:
            # Remove one randomly and add file2's question
            unique_questions.remove(random.choice(unique_questions))
    
    # Add question from file2 if not already there
    if question_from_file2 not in unique_questions:
        unique_questions.append(question_from_file2)
    
    # Process answers (same logic)
    if len(unique_answers) == 3:
        # If file2's answer is already in the pool, remove a different one
        if answer_from_file2 in unique_answers:
            # Remove one that's not from file2
            other_answers = [a for a in unique_answers if a != answer_from_file2]
            if other_answers:
                unique_answers.remove(random.choice(other_answers))
        else:
            # Remove one randomly and add file2's answer
            unique_answers.remove(random.choice(unique_answers))
    
    # Add answer from file2 if not already there
    if answer_from_file2 not in unique_answers:
        unique_answers.append(answer_from_file2)
    
    # Verify file2's question and answer are in the pools
    assert question_from_file2 in unique_questions, f"File2 question not in pool for index {original_index}"
    assert answer_from_file2 in unique_answers, f"File2 answer not in pool for index {original_index}"
    
    # Create all combinations (can be 4, 6, or 9 QA pairs)
    qa_pairs = []
    for question in unique_questions:
        for answer in unique_answers:
            qa_pair = {
                'question': question,
                'answer': answer,
                'combined_text': f"{question} {answer}",
                'author': item_file2['author'],
                'author_index': item_file2['author_index'],
                'original_index': original_index,
                'split': 'train'
            }
            qa_pairs.append(qa_pair)
    
    return qa_pairs, len(unique_questions), len(unique_answers)


def create_final_dataset(file1_path: str, file2_path: str, output_path: str):
    """
    Create the final dataset with QA pairs per original_index.
    
    Args:
        file1_path: Path to tofu_train_w_redundancy.json
        file2_path: Path to tofu_train.json
        output_path: Path to save tofu_train_w_redundancy_final.json
    """
    print(f"Loading {file1_path}...")
    data1 = load_json_file(file1_path)
    
    print(f"Loading {file2_path}...")
    data2 = load_json_file(file2_path)
    
    # Group data by original_index
    grouped_file1 = group_by_original_index(data1)
    grouped_file2 = group_by_original_index(data2)
    
    print(f"File1 has {len(grouped_file1)} unique original_index values")
    print(f"File2 has {len(grouped_file2)} unique original_index values")
    print()
    
    # Process each original_index
    final_data = []
    processed_count = 0
    skipped_count = 0
    combination_stats = defaultdict(int)  # Track how many indices have 4, 6, or 9 pairs
    
    for original_index in sorted(grouped_file1.keys()):
        if original_index not in grouped_file2:
            print(f"Warning: original_index {original_index} not found in file2, skipping...")
            skipped_count += 1
            continue
        
        items_file1 = grouped_file1[original_index]
        item_file2 = grouped_file2[original_index][0]  # Only one item per original_index in file2
        
        try:
            qa_pairs, num_questions, num_answers = process_original_index(original_index, items_file1, item_file2)
            final_data.extend(qa_pairs)
            processed_count += 1
            combination_count = len(qa_pairs)
            combination_stats[combination_count] += 1
        except AssertionError as e:
            print(f"Error processing original_index {original_index}: {e}")
            skipped_count += 1
    
    print(f"\nProcessed {processed_count} original_index values")
    print(f"Skipped {skipped_count} original_index values")
    print(f"Generated {len(final_data)} total QA pairs")
    print()
    print("Combination statistics:")
    for combo_count in sorted(combination_stats.keys()):
        num_indices = combination_stats[combo_count]
        print(f"  {combo_count} QA pairs: {num_indices} original_index values")
    
    print(f"\nSaving to {output_path}...")
    save_json_file(final_data, output_path)
    
    print(f"Done! Saved {len(final_data)} QA pairs to {output_path}")


# Example usage
if __name__ == '__main__':
    # File paths
    file1 = 'tofu_train_w_redundancy.json'
    file2 = 'tofu_train.json'
    output = 'tofu_train_w_redundancy_final.json'
    
    print("=" * 70)
    print("Creating final dataset with QA pairs per original_index")
    print("=" * 70)
    print(f"Input file 1 (paraphrased): {file1}")
    print(f"Input file 2 (original): {file2}")
    print(f"Output file: {output}")
    print()
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    create_final_dataset(file1, file2, output)
    
    print("\n" + "=" * 70)
    print("Process complete!")
    print("=" * 70)