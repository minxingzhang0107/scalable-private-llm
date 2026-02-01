import json
import os

def filter_qa_pairs_by_index(index_file_path, reference_file_path, output_file_path):
    """
    Filter QA pairs from reference file based on original_index values in index file.
    
    Args:
        index_file_path (str): Path to the index JSON file
        reference_file_path (str): Path to the reference JSON file
        output_file_path (str): Path for the output filtered JSON file
    """
    
    # Check if input files exist
    if not os.path.exists(index_file_path):
        print(f"Error: Index file '{index_file_path}' not found!")
        return
    
    if not os.path.exists(reference_file_path):
        print(f"Error: Reference file '{reference_file_path}' not found!")
        return
    
    try:
        # Load index file and extract original_index values
        print(f"Loading index file: {index_file_path}")
        with open(index_file_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        # Extract all original_index values from index file
        index_set = set()
        for qa_pair in index_data:
            if 'original_index' in qa_pair:
                index_set.add(qa_pair['original_index'])
        
        print(f"Found {len(index_set)} unique original_index values in index file")
        
        # Load reference file
        print(f"Loading reference file: {reference_file_path}")
        with open(reference_file_path, 'r', encoding='utf-8') as f:
            reference_data = json.load(f)
        
        print(f"Reference file contains {len(reference_data)} QA pairs")
        
        # Filter reference data based on original_index
        filtered_data = []
        for qa_pair in reference_data:
            if 'original_index' in qa_pair and qa_pair['original_index'] in index_set:
                filtered_data.append(qa_pair)
        
        # Save filtered data to new file
        print(f"Saving filtered data to: {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully filtered {len(filtered_data)} QA pairs from reference file")
        print(f"✅ Results saved to {output_file_path}")
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in one of the input files - {e}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """
    Main function with example usage.
    Modify the file paths below to match your actual files.
    """
    
    # File paths - modify these to match your actual file locations\
    # weird word replacement method
    # index_file = "weird_word_replacement_embedding/author_removal_20pct_retained_authors_results.json"           # Your index file
    # reference_file = "../results/private/tofu/1nn_lm_weird_word_replacement/combined_1nn_lm_finetuned_embedding_weird_word_perturbed_generated_answers_threshold_0.4.json"   # Your reference file  
    # output_file = "weird_word_replacement_embedding/author_removal_20pct_retained_authors_referencing_wo_removal_results.json"  # Output file name

    # name perturbation method
    index_file = "name_perturbation_embedding/author_removal_20pct_retained_authors_results.json"           # Your index file
    reference_file = "../results/private/tofu/1nn_lm_finetuned_name_perturbed_embedding/eps_10/combined_1nn_lm_finetuned_embedding_name_perturbation_generated_answers_threshold_0.4.json"   # Your reference file  
    output_file = "name_perturbation_embedding/author_removal_20pct_retained_authors_referencing_wo_removal_results.json"  # Output file name
    
    # Run the filtering
    filter_qa_pairs_by_index(index_file, reference_file, output_file)

if __name__ == "__main__":
    main()