import json
import sys
from typing import Dict, List, Any

def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Load JSON file and return the data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{filepath}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file '{filepath}': {e}")
        sys.exit(1)

def check_train_test_match(train_file: str = "tofu_train.json", test_file: str = "tofu_test_question_paraphrased.json"):
    """
    Check if each test entry has matching original_index in train with same question & answer.
    """
    print(f"Checking train-test match: '{train_file}' vs '{test_file}'")
    print("=" * 60)
    
    train_data = load_json_file(train_file)
    test_data = load_json_file(test_file)
    
    print(f"Train entries: {len(train_data)}")
    print(f"Test entries: {len(test_data)}")
    
    # Create dictionary mapping original_index to train entries
    train_by_original_index = {}
    for entry in train_data:
        original_idx = entry.get('original_index')
        if original_idx is not None:
            train_by_original_index[original_idx] = entry
    
    matches = 0
    no_matches = 0
    
    for test_entry in test_data:
        test_original_idx = test_entry.get('original_index')
        
        if test_original_idx is not None and test_original_idx in train_by_original_index:
            train_entry = train_by_original_index[test_original_idx]
            
            if (test_entry.get('question') == train_entry.get('question') and 
                test_entry.get('answer') == train_entry.get('answer')):
                matches += 1
            else:
                no_matches += 1
        else:
            no_matches += 1
    
    print(f"Matching entries (same original_index + same Q&A): {matches}")
    print(f"Non-matching entries: {no_matches}")
    
    all_matched = no_matches == 0
    if all_matched:
        print("✅ RESULT: ALL test entries match in train (same original_index + same Q&A)")
    else:
        print("❌ RESULT: NOT ALL test entries match in train")
    
    print()
    return all_matched

def check_test_paraphrased_match(test_file: str = "tofu_test_wo_paraphrase.json", paraphrased_file: str = "tofu_test_question_paraphrased.json"):
    """
    Check if test and paraphrased files have matching original_indices and same number of entries,
    and whether questions & answers are exactly the same.
    """
    print(f"Checking test-paraphrased match: '{test_file}' vs '{paraphrased_file}'")
    print("=" * 60)
    
    test_data = load_json_file(test_file)
    paraphrased_data = load_json_file(paraphrased_file)
    
    print(f"Test entries: {len(test_data)}")
    print(f"Paraphrased entries: {len(paraphrased_data)}")
    
    # Check if same number of entries
    same_count = len(test_data) == len(paraphrased_data)
    print(f"Same number of entries: {'✅ YES' if same_count else '❌ NO'}")
    
    # Create dictionary mapping original_index to paraphrased entries
    paraphrased_by_original_index = {}
    for entry in paraphrased_data:
        original_idx = entry.get('original_index')
        if original_idx is not None:
            paraphrased_by_original_index[original_idx] = entry
    
    # Check original_index matching
    matching_indices = 0
    missing_indices = 0
    same_questions = 0
    same_answers = 0
    both_qa_same = 0
    
    for test_entry in test_data:
        test_original_idx = test_entry.get('original_index')
        
        if test_original_idx is not None and test_original_idx in paraphrased_by_original_index:
            matching_indices += 1
            paraphrased_entry = paraphrased_by_original_index[test_original_idx]
            
            # Check if questions are exactly the same
            if test_entry.get('question') == paraphrased_entry.get('question'):
                same_questions += 1
            
            # Check if answers are exactly the same
            if test_entry.get('answer') == paraphrased_entry.get('answer'):
                same_answers += 1
            
            # Check if both Q&A are the same
            if (test_entry.get('question') == paraphrased_entry.get('question') and 
                test_entry.get('answer') == paraphrased_entry.get('answer')):
                both_qa_same += 1
        else:
            missing_indices += 1
    
    print(f"Matching original_indices: {matching_indices}")
    print(f"Missing original_indices: {missing_indices}")
    
    all_indices_match = missing_indices == 0
    print(f"All original_indices match: {'✅ YES' if all_indices_match else '❌ NO'}")
    
    print(f"Entries with exactly same questions: {same_questions}")
    print(f"Entries with exactly same answers: {same_answers}")
    print(f"Entries with both Q&A exactly same: {both_qa_same}")
    
    all_qa_same = both_qa_same == len(test_data)
    print(f"All Q&A exactly the same: {'✅ YES' if all_qa_same else '❌ NO'}")
    
    print()
    return all_indices_match, all_qa_same

def check_train_question_paraphrased_match(train_file: str = "tofu_train.json", question_paraphrased_file: str = "tofu_test_question_paraphrased.json"):
    """
    Check if all indices in question_paraphrased appear in train, 
    and separately check question and answer matching.
    """
    print(f"Checking train vs question-paraphrased: '{train_file}' vs '{question_paraphrased_file}'")
    print("=" * 60)
    
    train_data = load_json_file(train_file)
    question_paraphrased_data = load_json_file(question_paraphrased_file)
    
    print(f"Train entries: {len(train_data)}")
    print(f"Question-paraphrased entries: {len(question_paraphrased_data)}")
    
    # Create dictionary mapping original_index to train entries
    train_by_original_index = {}
    for entry in train_data:
        original_idx = entry.get('original_index')
        if original_idx is not None:
            train_by_original_index[original_idx] = entry
    
    # Check indices, questions, and answers separately
    indices_in_train = 0
    indices_missing = 0
    same_questions = 0
    same_answers = 0
    
    for para_entry in question_paraphrased_data:
        para_original_idx = para_entry.get('original_index')
        
        if para_original_idx is not None and para_original_idx in train_by_original_index:
            indices_in_train += 1
            train_entry = train_by_original_index[para_original_idx]
            
            # Check if questions are exactly the same
            if para_entry.get('question') == train_entry.get('question'):
                print(para_entry.get('original_index'))
                same_questions += 1
            
            # Check if answers are exactly the same
            if para_entry.get('answer') == train_entry.get('answer'):
                same_answers += 1
        else:
            indices_missing += 1
    
    print(f"Indices found in train: {indices_in_train}")
    print(f"Indices missing from train: {indices_missing}")
    
    all_indices_in_train = indices_missing == 0
    print(f"All indices appear in train: {'✅ YES' if all_indices_in_train else '❌ NO'}")
    
    print(f"Entries with exactly same questions: {same_questions}")
    print(f"Entries with exactly same answers: {same_answers}")
    
    all_questions_same = same_questions == len(question_paraphrased_data)
    all_answers_same = same_answers == len(question_paraphrased_data)
    
    print(f"All questions exactly the same: {'✅ YES' if all_questions_same else '❌ NO'}")
    print(f"All answers exactly the same: {'✅ YES' if all_answers_same else '❌ NO'}")
    
    print()
    return all_indices_in_train, all_questions_same, all_answers_same

def main():
    """Main function to run all three checks."""
    print("DATASET CONSISTENCY CHECKER")
    print("=" * 70)
    print()
    
    # Check 1: Train vs Test
    train_test_match = check_train_test_match()
    
    # Check 2: Test vs Paraphrased
    indices_match, qa_same = check_test_paraphrased_match()
    
    # Check 3: Train vs Question-Paraphrased
    train_para_indices, train_para_questions, train_para_answers = check_train_question_paraphrased_match()
    
    print("FINAL SUMMARY:")
    print("=" * 40)
    print(f"1. Train-Test match (same original_index + Q&A): {'✅ YES' if train_test_match else '❌ NO'}")
    print(f"2. Test-Paraphrased original_indices match: {'✅ YES' if indices_match else '❌ NO'}")
    print(f"3. Test-Paraphrased Q&A exactly same: {'✅ YES' if qa_same else '❌ NO'}")
    print(f"4. Train-QuestionPara all indices in train: {'✅ YES' if train_para_indices else '❌ NO'}")
    print(f"5. Train-QuestionPara questions exactly same: {'✅ YES' if train_para_questions else '❌ NO'}")
    print(f"6. Train-QuestionPara answers exactly same: {'✅ YES' if train_para_answers else '❌ NO'}")

if __name__ == "__main__":
    main()