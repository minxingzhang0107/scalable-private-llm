#!/usr/bin/env python3
"""
tofu_train_test_split.py - Modified TOFU Dataset Processing 
- Randomly select 50 authors
- Use all 20 QA pairs from selected authors as training (1000 total)
- Randomly select 2 QA pairs from each selected author for testing (100 total)
- NO PARAPHRASING - use original test data directly
"""

import os
import json
import shutil
from datasets import load_dataset
from collections import defaultdict
import random



def check_existing_data(output_dir="dataset/private/tofu"):
    """Check if processed data already exists"""
    required_files = [
        "tofu_train.json",
        "tofu_test.json", 
        "metadata.json"
    ]
    
    if not os.path.exists(output_dir):
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            return False
    
    return True

def remove_existing_data():
    """Remove existing data files and directories"""
    directories_to_remove = [
        "dataset/private/tofu"
    ]
    
    for directory in directories_to_remove:
        if os.path.exists(directory):
            print(f"🗑️  Removing existing directory: {directory}")
            shutil.rmtree(directory)
            print(f"✅ Removed: {directory}")

def process_tofu_dataset():
    """Process TOFU dataset with sequential author IDs"""
    print("🚀 Modified TOFU Dataset Processing")
    print("=" * 50)
    
    # Load dataset
    print("📥 Loading TOFU dataset...")
    try:
        dataset = load_dataset("locuslab/TOFU", "full")['train']
        print(f"✅ Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Dataset structure
    examples_per_author = 20
    total_examples = len(dataset)
    expected_authors = total_examples // examples_per_author
    
    print(f"\n📊 Dataset structure:")
    print(f"  Total examples: {total_examples}")
    print(f"  Examples per author: {examples_per_author}")
    print(f"  Expected authors: {expected_authors}")
    
    # Process in groups of 20 with sequential author IDs
    print(f"\n👥 Organizing into {expected_authors} authors...")
    
    authors_data = []
    
    for author_idx in range(expected_authors):
        start_idx = author_idx * examples_per_author
        end_idx = start_idx + examples_per_author
        
        # Create simple author ID
        author_id = f"author_{author_idx + 1}"
        
        # Get all examples for this author
        group_examples = []
        for i in range(start_idx, min(end_idx, total_examples)):
            example = dataset[i]
            group_examples.append({
                'question': example['question'],
                'answer': example['answer'],
                'original_index': i
            })
        
        authors_data.append({
            'author_id': author_id,
            'author_index': author_idx,
            'examples': group_examples,
            'start_idx': start_idx,
            'end_idx': end_idx - 1
        })
        
        if (author_idx + 1) % 50 == 0:
            print(f"  Processed {author_idx + 1}/{expected_authors} authors...")
    
    print(f"\n✅ Organization complete:")
    print(f"  Total authors: {len(authors_data)}")
    print(f"  Each author: {examples_per_author} examples")
    
    return authors_data, dataset

def create_modified_train_test_split(authors_data, random_seed=42):
    """
    Modified split:
    - Randomly select 50 authors
    - Use all 20 QA pairs from selected authors as training (1000 total)
    - Randomly select 2 QA pairs from each selected author for testing (100 total)
    """
    print(f"\n✂️  Creating modified train/test split")
    
    random.seed(random_seed)
    
    # Randomly select 50 authors
    selected_authors = random.sample(authors_data, 50)
    selected_author_ids = [author['author_id'] for author in selected_authors]
    
    print(f"🎯 Randomly selected 50 authors for training:")
    print(f"   {selected_author_ids[:10]}... (showing first 10)")
    
    train_data = []
    test_candidates = []
    
    # Process selected authors
    for author_data in selected_authors:
        author_id = author_data['author_id']
        examples = author_data['examples']
        
        # Shuffle this author's examples
        shuffled_examples = examples.copy()
        random.shuffle(shuffled_examples)
        
        # ALL 20 examples go to training
        for example in shuffled_examples:
            train_data.append({
                'question': example['question'],
                'answer': example['answer'],
                'combined_text': example['question'] + " " + example['answer'],
                'author': author_id,
                'author_index': author_data['author_index'],
                'original_index': example['original_index'],
                'split': 'train'
            })
        
        # Randomly select 2 examples for test candidates (will be paraphrased)
        test_examples = random.sample(shuffled_examples, 2)
        for example in test_examples:
            test_candidates.append({
                'question': example['question'],
                'answer': example['answer'],
                'combined_text': example['question'] + " " + example['answer'],
                'author': author_id,
                'author_index': author_data['author_index'],
                'original_index': example['original_index'],
                'split': 'test'
            })
    
    print(f"\n✅ Split complete:")
    print(f"  Train examples: {len(train_data)} (20 × 50 authors)")
    print(f"  Test candidates: {len(test_candidates)} (2 × 50 authors)")
    print(f"  Selected authors: {len(selected_authors)}")
    
    return train_data, test_candidates, selected_author_ids

def prepare_test_data(test_candidates):
    """Use original test data directly - NO PARAPHRASING"""
    print(f"\n📋 Preparing test data (using originals, no paraphrasing)...")
    
    # Just return the original test candidates as-is
    print(f"✅ Test data prepared: {len(test_candidates)} original test items")
    return test_candidates

def save_organized_dataset(train_data, test_data, selected_author_ids, output_dir="dataset/private/tofu"):
    """Save the organized dataset"""
    print(f"\n💾 Saving organized dataset to {output_dir}")
    
    # Create fresh directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main train/test files
    train_file = os.path.join(output_dir, "tofu_train.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {train_file}")
    
    test_file = os.path.join(output_dir, "tofu_test.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {test_file}")
    
    # Save metadata
    metadata = {
        'total_authors_selected': len(selected_author_ids),
        'selected_authors': selected_author_ids,
        'examples_per_author_total': 20,
        'train_per_author': 20,  # All examples used for training
        'test_per_author': 2,    # 2 examples paraphrased for testing
        'train_examples': len(train_data),
        'test_examples': len(test_data),
        'split_method': 'random_50_authors_all_train_original_test',
        'random_seed': 42,
        'paraphrasing_applied': False,
        'notes': [
            'Randomly selected 50 authors out of all available authors',
            'Used all 20 QA pairs from selected authors for training (1000 total)',
            'Randomly selected 2 QA pairs from each author for testing (100 total)',
            'Test data uses original questions and answers (no paraphrasing)',
            'Both train and test data from the same 50 selected authors'
        ]
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {metadata_file}")
    
    return {
        'train_file': train_file,
        'test_file': test_file,
        'metadata_file': metadata_file
    }

def main():
    """Main processing pipeline"""
    print("🎯 Processing TOFU Dataset with Modified Split (NO PARAPHRASING)")
    
    # Check if processed data already exists
    if check_existing_data():
        print("✅ Processed data already exists! Skipping processing...")
        print("📁 Found existing files:")
        print("   - dataset/private/tofu/tofu_train.json")
        print("   - dataset/private/tofu/tofu_test.json")
        print("   - dataset/private/tofu/metadata.json")
        print("🔄 Delete these files if you want to regenerate the data")
        return True
    
    # Process the dataset
    authors_data, dataset = process_tofu_dataset()
    if not authors_data:
        return False
    
    # Create modified train/test split
    train_data, test_candidates, selected_author_ids = create_modified_train_test_split(authors_data, random_seed=42)
    
    # Prepare test data (no paraphrasing)
    original_test_data = prepare_test_data(test_candidates)
    
    # Save organized dataset
    files_created = save_organized_dataset(train_data, original_test_data, selected_author_ids)
    
    print(f"\n🎉 TOFU dataset processing complete!")
    print(f"📊 Final statistics:")
    print(f"  Selected authors: {len(selected_author_ids)}")
    print(f"  Train examples: {len(train_data)} (all 20 from each selected author)")
    print(f"  Test examples: {len(original_test_data)} (2 original from each selected author)")
    print(f"  Test data: 100% original (no paraphrasing applied)")
    print(f"\n📋 Selected authors sample:")
    for i, author_id in enumerate(selected_author_ids[:5]):
        print(f"  {author_id}")
    print(f"  ... and {len(selected_author_ids) - 5} more")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Processing failed!")
        exit(1)