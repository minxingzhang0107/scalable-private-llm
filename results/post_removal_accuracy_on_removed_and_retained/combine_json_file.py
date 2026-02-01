import json
import os

def combine_qa_data_by_index(ground_truth_file, generated_file, output_file):
    """
    Combines question, ground truth answer, and generated answer based on a
    shared 'original_index'.

    Args:
        ground_truth_file (str): Path to the JSON file with ground truth answers.
        generated_file (str): Path to the JSON file with generated answers.
        output_file (str): Path for the output JSON file.
    """
    try:
        # Load the ground truth data
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)

        # Load the generated answers data
        with open(generated_file, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)

        # Create a lookup dictionary from the ground truth data for efficient access.
        # This maps original_index -> ground_truth_answer.
        ground_truth_map = {
            item.get('original_index'): item.get('answer')
            for item in ground_truth_data if 'original_index' in item
        }

        # List to hold the combined data
        combined_data = []

        # Iterate through each item in the generated answers file
        for gen_item in generated_data:
            original_index = gen_item.get("original_index")

            # Skip if the generated item is missing essential keys
            if original_index is None or "question" not in gen_item or "answer" not in gen_item:
                print(f"Warning: Skipping malformed item in generated data: {gen_item}")
                continue

            # Find the corresponding ground truth answer using the map
            ground_truth_answer = ground_truth_map.get(original_index)

            if ground_truth_answer is not None:
                # Create a new dictionary with the desired structure
                new_item = {
                    "question": gen_item.get("question"),
                    "ground_truth_answer": ground_truth_answer,
                    "generated_answer": gen_item.get("answer") # Rename key
                }
                combined_data.append(new_item)
            else:
                # Warn if no matching ground truth answer is found for an index
                print(f"Warning: No ground truth answer found for original_index: {original_index}")

        total_questions = len(combined_data)
        print(f"Successfully matched and combined {total_questions} QA pairs.")

        # Save the combined data to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)

        print(f"Output saved to: {output_file}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except json.JSONDecodeError as e:
        print(f"Error: A file is not a valid JSON. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def generate_output_filename(generated_file):
    """
    Generate output filename based on the generated answers file name.
    
    Args:
        generated_file (str): Path to the generated answers file
        
    Returns:
        str: Generated output filename
    """
    base_name = os.path.splitext(os.path.basename(generated_file))[0]
    # This path can be adjusted as needed
    output_dir = "weird_word_replacement_embedding"
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    return os.path.join(output_dir, f"evaluation_{base_name}.json")


if __name__ == "__main__":
    # --- Configuration ---
    # File containing the ground truth answers with 'original_index'
    GROUND_TRUTH_FILENAME = '../dataset/private/tofu/tofu_test_question_paraphrased.json'
    
    # File containing the model's generated answers with 'original_index'
    # GENERATED_ANSWERS_FILENAME = 'weird_word_replacement_embedding/author_removal_20pct_retained_authors_results.json'
    GENERATED_ANSWERS_FILENAME = 'weird_word_replacement_embedding/author_removal_20pct_retained_authors_referencing_wo_removal_results.json'
    
    # Generate output filename based on the generated answers file
    OUTPUT_FILENAME = generate_output_filename(GENERATED_ANSWERS_FILENAME)
    
    print("--- Starting Data Combination ---")
    print(f"  Ground truth file: {GROUND_TRUTH_FILENAME}")
    print(f"  Generated answers file: {GENERATED_ANSWERS_FILENAME}")
    print(f"  Output file: {OUTPUT_FILENAME}")
    print("-" * 30)

    # Run the new function
    combine_qa_data_by_index(GROUND_TRUTH_FILENAME, GENERATED_ANSWERS_FILENAME, OUTPUT_FILENAME)