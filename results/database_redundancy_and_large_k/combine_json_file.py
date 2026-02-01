import json
import os

def combine_qa_data(ground_truth_file, generated_file, output_file):
    """
    Combines question, ground truth answer, and generated answer from two
    JSON files into a single file. It saves all entries (no limit).

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

        # Ensure both files have the same number of entries
        if len(ground_truth_data) != len(generated_data):
            print("Error: The two JSON files have a different number of entries.")
            return

        # List to hold the combined data
        combined_data = []

        # Use zip to iterate through both lists simultaneously
        for gt_item, gen_item in zip(ground_truth_data, generated_data):
            # Verify that the questions match before combining
            if gt_item.get("question") != gen_item.get("question"):
                print(f"Warning: Mismatched questions found. Skipping.")
                print(f"  - Ground Truth: {gt_item.get('question')}")
                print(f"  - Generated: {gen_item.get('question')}")
                continue

            # Create a new dictionary with the desired keys
            new_item = {
                "question": gt_item.get("question"),
                "ground_truth_answer": gt_item.get("answer"),
                "generated_answer": gen_item.get("answer")
            }
            combined_data.append(new_item)

        # Count the total number of questions found
        total_questions = len(combined_data)
        print(f"Found a total of {total_questions} questions.")

        # Save ALL the combined data to the output file (no limit)
        data_to_save = combined_data

        # Save the combined data to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            # Use indent for pretty-printing the JSON output
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)

        print(f"Successfully saved all {len(data_to_save)} QA pairs.")
        print(f"Output saved to: {output_file}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except json.JSONDecodeError:
        print("Error: One of the files is not a valid JSON file.")
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
    # Get the base filename without path and extension
    base_name = os.path.splitext(os.path.basename(generated_file))[0]
    
    # Create evaluation filename
    # private
    # return f"name_perturbation_embedding/k_7/evaluation_{base_name}.json"
    return f"weird_word_replacement_embedding/k_7/evaluation_{base_name}.json"




if __name__ == "__main__":
    # --- Configuration ---
    # File containing the ground truth answers
    # private ground truth file
    GROUND_TRUTH_FILENAME = '../dataset/private/tofu/tofu_test_question_paraphrased.json'
    
    # private generated file
    # GENERATED_ANSWERS_FILENAME = 'name_perturbation_embedding/k_7/combined_knn_lm_finetuned_embedding_name_perturbation_generated_answers_threshold_0.4.json'
    GENERATED_ANSWERS_FILENAME = 'weird_word_replacement_embedding/k_7/combined_knn_lm_finetuned_embedding_weird_word_perturbed_generated_answers_threshold_0.4.json'


    # Generate output filename based on the generated answers file
    OUTPUT_FILENAME = generate_output_filename(GENERATED_ANSWERS_FILENAME)
    
    print(f"Input files:")
    print(f"  Ground truth: {GROUND_TRUTH_FILENAME}")
    print(f"  Generated answers: {GENERATED_ANSWERS_FILENAME}")
    print(f"  Output file: {OUTPUT_FILENAME}")
    print()

    # Run the function
    combine_qa_data(GROUND_TRUTH_FILENAME, GENERATED_ANSWERS_FILENAME, OUTPUT_FILENAME)