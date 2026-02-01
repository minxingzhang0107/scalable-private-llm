import json

def compare_qa_files(file1_path, file2_path):
    """
    Compare two JSON files containing QA pairs to check:
    1. Same number of QA pairs
    2. question, author, author_index, original_index, split are exactly the same
    3. answer and combined_text are different
    """
    
    # Load both JSON files
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        with open(file2_path, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return False
    
    # Check if both are lists
    if not isinstance(data1, list) or not isinstance(data2, list):
        print("Error: Both files should contain lists of QA pairs")
        return False
    
    # Check number of QA pairs
    if len(data1) != len(data2):
        print(f"❌ Different number of QA pairs: File1 has {len(data1)}, File2 has {len(data2)}")
        return False
    else:
        print(f"✅ Same number of QA pairs: {len(data1)}")
    
    # Fields that should be the same
    same_fields = ['question', 'author', 'author_index', 'original_index', 'split']
    
    # Fields that should be different
    different_fields = ['answer', 'combined_text']
    
    all_checks_passed = True
    
    # Compare each QA pair
    for i, (qa1, qa2) in enumerate(zip(data1, data2)):
        print(f"\n--- Checking QA pair {i+1} ---")
        
        # Check fields that should be the same
        for field in same_fields:
            if field not in qa1 or field not in qa2:
                print(f"❌ Missing field '{field}' in QA pair {i+1}")
                all_checks_passed = False
                continue
                
            if qa1[field] != qa2[field]:
                print(f"❌ Field '{field}' differs in QA pair {i+1}:")
                print(f"    File1: {qa1[field]}")
                print(f"    File2: {qa2[field]}")
                all_checks_passed = False
            else:
                print(f"✅ Field '{field}' is the same")
        
        # Check fields that should be different
        for field in different_fields:
            if field not in qa1 or field not in qa2:
                print(f"❌ Missing field '{field}' in QA pair {i+1}")
                all_checks_passed = False
                continue
                
            if qa1[field] == qa2[field]:
                print(f"❌ Field '{field}' is the same in QA pair {i+1} (should be different):")
                print(f"    Both files: {qa1[field]}")
                all_checks_passed = False
            else:
                print(f"✅ Field '{field}' is different (as expected)")
                print(f"    File1: {qa1[field][:100]}{'...' if len(qa1[field]) > 100 else ''}")
                print(f"    File2: {qa2[field][:100]}{'...' if len(qa2[field]) > 100 else ''}")
    
    # Final summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY:")
    print(f"{'='*50}")
    
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Same number of QA pairs")
        print("✅ All metadata fields (question, author, author_index, original_index, split) are identical")
        print("✅ All content fields (answer, combined_text) are different")
        return True
    else:
        print("❌ SOME CHECKS FAILED!")
        print("Please review the issues listed above.")
        return False

def compare_qa_files_summary_only(file1_path, file2_path):
    """
    Same comparison but only shows summary without detailed per-item output
    """
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        with open(file2_path, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
    except Exception as e:
        print(f"Error loading files: {e}")
        return False
    
    if len(data1) != len(data2):
        print(f"❌ Different number of QA pairs: {len(data1)} vs {len(data2)}")
        return False
    
    same_fields = ['question', 'author', 'author_index', 'original_index', 'split']
    different_fields = ['answer', 'combined_text']
    
    issues = []
    
    for i, (qa1, qa2) in enumerate(zip(data1, data2)):
        for field in same_fields:
            if qa1.get(field) != qa2.get(field):
                issues.append(f"QA {i+1}: {field} differs")
        
        for field in different_fields:
            if qa1.get(field) == qa2.get(field):
                issues.append(f"QA {i+1}: {field} is same (should differ)")
    
    print(f"Number of QA pairs (raw): {len(data1)}")
    print(f"Number of QA pairs (new): {len(data2)}")
    
    if not issues:
        print("🎉 ALL CHECKS PASSED!")
        return True
    else:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        return False

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    file1 = "tofu_toy_weird_name_raw.json"
    file2 = "tofu_toy_weird_name.json"
    
    print("=== DETAILED COMPARISON ===")
    compare_qa_files(file1, file2)
    
    print("\n\n=== SUMMARY ONLY ===")
    compare_qa_files_summary_only(file1, file2)