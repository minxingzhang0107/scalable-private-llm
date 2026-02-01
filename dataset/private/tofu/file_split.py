import json
import os

def split_json_file(input_file, output_dir="split_files", items_per_file=100):
    """
    Split a JSON file containing QA pairs into multiple smaller files.
    
    Args:
        input_file (str): Path to the input JSON file
        output_dir (str): Directory to save the split files
        items_per_file (int): Number of items per output file
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the JSON data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")
        return
    
    # Check if data is a list
    if not isinstance(data, list):
        print("Error: JSON file should contain a list of QA pairs.")
        return
    
    total_items = len(data)
    print(f"Total items in the file: {total_items}")
    
    # Calculate number of output files needed
    num_files = (total_items + items_per_file - 1) // items_per_file
    print(f"Creating {num_files} output files...")
    
    # Split the data and save to separate files
    for i in range(num_files):
        start_index = i * items_per_file
        end_index = min(start_index + items_per_file, total_items)
        
        # Extract the subset of data
        subset = data[start_index:end_index]
        
        # Create output filename
        output_file = os.path.join(output_dir, f"qa_pairs_part_{i+1:02d}.json")
        
        # Save the subset to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        
        print(f"Created {output_file} with {len(subset)} items (indices {start_index}-{end_index-1})")
    
    print(f"\nSplitting complete! All files saved in '{output_dir}' directory.")

# Example usage
if __name__ == "__main__":
    # Replace 'your_qa_file.json' with the actual path to your JSON file
    input_filename = "tofu_train.json"
    
    # Split the file into chunks of 100 items each
    split_json_file(input_filename, output_dir="split_qa_files", items_per_file=100)