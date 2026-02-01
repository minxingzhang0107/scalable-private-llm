#!/usr/bin/env python3
"""
author_name_extraction_mapping.py - Extract COMPLETE author names from TOFU JSON data
Gets the full names that actually appear in the text (e.g., "Maria Garcia Alvarez", not just "Maria Garcia")
"""

import json
import re
from collections import Counter

def extract_person_names(text):
    """
    Extract potential person names - handles various formats including 3+ word names
    """
    # Pattern 1: Standard two capitalized words
    pattern1 = r'\b[A-Z]\w+\s+[A-Z]\w+\b'
    
    # Pattern 2: Three capitalized words (like "Maria Garcia Alvarez")
    pattern2 = r'\b[A-Z]\w+\s+[A-Z]\w+\s+[A-Z]\w+\b'
    
    # Pattern 3: Four capitalized words (like "Maria Garcia Alvarez Rodriguez")
    pattern3 = r'\b[A-Z]\w+\s+[A-Z]\w+\s+[A-Z]\w+\s+[A-Z]\w+\b'
    
    # Pattern 4: Names with "van", "de", "al" (like Isabella van Pletzen)
    pattern4 = r'\b[A-Z]\w+\s+(?:van|de|da|von|al|el)\s+[A-Z]\w+\b'
    
    # Pattern 5: Hyphenated names (like Tae-ho Park)
    pattern5 = r'\b[A-Z][\w-]+\s+[A-Z]\w+\b'
    
    names = []
    # Check longer patterns first to get complete names
    names.extend(re.findall(pattern3, text))  # 4 words first
    names.extend(re.findall(pattern2, text))  # 3 words second
    names.extend(re.findall(pattern4, text))  # van/de names
    names.extend(re.findall(pattern5, text))  # hyphenated
    names.extend(re.findall(pattern1, text))  # 2 words last
    
    return names

def is_obviously_not_a_person(name):
    """
    Simple check - is this obviously NOT a person name?
    """
    name_lower = name.lower()
    
    # Common non-person patterns
    bad_patterns = [
        # Starts with obvious non-name words
        'the ', 'has ', 'does ', 'did ', 'was ', 'is ', 'are ', 'can ', 'will ',
        
        # Contains obvious non-name words
        'fiction', 'award', 'prize', 'literature', 'book', 'novel',
        'city', 'town', 'street', 'road', 'avenue',
        'biologist', 'engineer', 'scientist', 'attendant', 'officer',
        'university', 'college', 'school',
        'monsoon', 'pursuit', 'stress', 'vessel', 'chances', 'tale',
        'melody', 'schizophrenia', 'revisited', 'evolutionary',
        'zodiac', 'empowerment', 'literary', 'african', 'korean art',
        'charles dickens', 'south african', 'garcia alvarez', 'award of',
        'excellence in', 'theological literature', 'madrid spain'
    ]
    
    return any(pattern in name_lower for pattern in bad_patterns)

def get_longest_name_containing(candidate_names, short_name):
    """
    Given a list of candidate names, find the longest one that contains the short name
    """
    matching_names = []
    for name in candidate_names:
        if short_name.lower() in name.lower():
            matching_names.append(name)
    
    if matching_names:
        # Return the longest name (most complete)
        return max(matching_names, key=len)
    return short_name

def manual_corrections(author_names):
    """
    Apply manual corrections for known problematic cases
    """
    corrections = {
        'author_61': 'Chris Delaney',
        'author_66': 'Isabella van Pletzen', 
        'author_89': 'The Author',
        'author_122': 'Sara van Dyke',
        'author_190': 'Tae-ho Park'
    }
    
    print("\n🔧 Applying manual corrections for edge cases...")
    for author_id, correct_name in corrections.items():
        if author_id in author_names:
            old_name = author_names[author_id]
            author_names[author_id] = correct_name
            print(f"  ✅ {author_id}: {old_name} → {correct_name}")
    
    return author_names

def analyze_training_data():
    """
    Load training data and find COMPLETE author names
    """
    print("🔍 Loading training data...")
    
    try:
        with open("dataset/private/tofu/tofu_train.json", 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"✅ Loaded {len(train_data)} training examples")
    except FileNotFoundError:
        print("❌ Could not find dataset/private/tofu/tofu_train.json")
        return False
    
    # Group by author_id
    train_by_author = {}
    for example in train_data:
        author_id = example['author']
        if author_id not in train_by_author:
            train_by_author[author_id] = []
        train_by_author[author_id].append(example)
    
    print(f"📊 Found {len(train_by_author)} authors in training data")
    
    # Find author names
    author_names = {}
    
    for author_id in sorted(train_by_author.keys(), key=lambda x: int(x.split('_')[1])):
        examples = train_by_author[author_id]
        
        print(f"\n🔍 Processing {author_id}...")
        
        # Collect all names from this author's examples
        all_names = []
        for example in examples:
            question_names = extract_person_names(example['question'])
            answer_names = extract_person_names(example['answer'])
            combined_names = extract_person_names(example.get('combined_text', ''))
            all_names.extend(question_names + answer_names + combined_names)
        
        if all_names:
            # Count frequencies
            name_counts = Counter(all_names)
            
            print(f"  🔎 Found names: {dict(name_counts.most_common(5))}")
            
            # Filter out obvious non-person names
            person_names = []
            for name, count in name_counts.most_common():
                if not is_obviously_not_a_person(name):
                    person_names.append((name, count))
            
            if person_names:
                # Get all valid person names as candidates
                candidate_names = [name for name, count in person_names]
                
                # Pick the most frequent person name
                best_name, best_count = person_names[0]
                
                # Try to find a longer version of this name
                complete_name = get_longest_name_containing(candidate_names, best_name)
                
                author_names[author_id] = complete_name
                print(f"  ✅ Selected: {complete_name} (frequency: {best_count})")
            else:
                author_names[author_id] = "Unknown"
                print(f"  ❌ No valid person names found")
        else:
            author_names[author_id] = "Unknown"
            print(f"  ❌ No names extracted")
    
    # Apply manual corrections for known edge cases
    author_names = manual_corrections(author_names)
    
    return author_names

def create_detailed_analysis(train_data, author_names):
    """
    Create detailed analysis of the training data
    """
    print("\n📊 Creating detailed analysis...")
    
    # Group by author
    train_by_author = {}
    for example in train_data:
        author_id = example['author']
        if author_id not in train_by_author:
            train_by_author[author_id] = []
        train_by_author[author_id].append(example)
    
    # Calculate statistics
    qa_pairs_per_author = len(train_data) // len(train_by_author) if train_by_author else 0
    
    analysis = {
        'summary_statistics': {
            'total_qa_pairs': len(train_data),
            'total_authors': len(train_by_author),
            'examples_per_author': qa_pairs_per_author,
            'authors_analyzed': len([name for name in author_names.values() if name != "Unknown"])
        },
        'author_details': {},
        'name_extraction_stats': {
            'successful_extractions': len([name for name in author_names.values() if name != "Unknown"]),
            'failed_extractions': len([name for name in author_names.values() if name == "Unknown"]),
            'manual_corrections_applied': 5
        }
    }
    
    # Add details for each author
    for author_id, examples in train_by_author.items():
        author_name = author_names.get(author_id, "Unknown")
        
        # Sample questions and answers
        sample_qa = examples[:2] if len(examples) >= 2 else examples
        
        analysis['author_details'][author_id] = {
            'extracted_name': author_name,
            'qa_pair_count': len(examples),
            'sample_questions': [qa['question'][:100] + "..." for qa in sample_qa],
            'sample_answers': [qa['answer'][:100] + "..." for qa in sample_qa]
        }
    
    return analysis

def save_results(author_names, train_data):
    """
    Save the extracted author names and analysis
    """
    print(f"\n💾 Saving results...")
    
    # Calculate QA pairs per author
    author_count = len(set(example['author'] for example in train_data))
    qa_pairs_per_author = len(train_data) // author_count if author_count > 0 else 0
    
    # Save author mapping
    result = {
        'author_mapping': author_names,
        'total_authors': len(author_names),
        'examples_per_author': qa_pairs_per_author,
        'extraction_method': 'complete_name_extraction_with_manual_corrections',
        'manual_corrections': [
            'author_61: Chris Delaney',
            'author_66: Isabella van Pletzen', 
            'author_89: The Author',
            'author_122: Sara van Dyke',
            'author_190: Tae-ho Park'
        ],
        'notes': [
            'Extraction based on COMPLETE names that appear in text',
            'Prioritizes longer names over shorter partial names',
            'Uses 3-4 word name patterns to capture full names like "Maria Garcia Alvarez"',
            'Manual corrections for 5 edge cases',
            'author_89 uses "The Author" since no actual name is provided'
        ]
    }
    
    with open("dataset/private/tofu/author_names_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved author mapping with complete names")
    
    # Save detailed analysis
    analysis = create_detailed_analysis(train_data, author_names)
    
    with open("dataset/private/tofu/author_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved detailed analysis")
    
    return True

def main():
    """
    Main function - extract COMPLETE author names
    """
    print("🎯 TOFU COMPLETE Author Name Extraction")
    print("=" * 60)
    
    # Load training data first (needed for analysis)
    try:
        with open("dataset/private/tofu/tofu_train.json", 'r', encoding='utf-8') as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print("❌ Could not find dataset/private/tofu/tofu_train.json")
        return False
    
    # Extract author names
    author_names = analyze_training_data()
    if not author_names:
        return False
    
    # Save results
    save_results(author_names, train_data)
    
    # Summary
    valid_names = sum(1 for name in author_names.values() if name not in ["Unknown"])
    print(f"\n🎉 Extraction complete!")
    print(f"📊 Results: {valid_names}/{len(author_names)} authors identified")
    print(f"🔧 Manual corrections: 5 edge cases handled")
    
    print(f"\n📋 Sample results (showing COMPLETE names):")
    for author_id in sorted(list(author_names.keys()))[:10]:
        print(f"  {author_id} = {author_names[author_id]}")
    
    print(f"\n🔧 Manual corrections applied:")
    manual_cases = ['author_61', 'author_66', 'author_89', 'author_122', 'author_190']
    for author_id in manual_cases:
        if author_id in author_names:
            print(f"  {author_id} = {author_names[author_id]}")
    
    print(f"\n✅ Complete names extracted (e.g., 'Maria Garcia Alvarez' not just 'Maria Garcia')")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Extraction failed!")
        exit(1)