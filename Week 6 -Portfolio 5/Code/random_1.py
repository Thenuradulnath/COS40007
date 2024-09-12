import os
import random
import shutil
import json

def get_random_test_samples_and_move(root_dir, output_test_dir, sample_size=10):
    # Ensure the output directory exists
    os.makedirs(output_test_dir, exist_ok=True)
    
    # Get all JSON files in the root directory
    all_files = [f for f in os.listdir(root_dir) if f.endswith('.json')]
    
    if not all_files:
        print(f"No JSON files found in the directory: {root_dir}")
        return
    
    # Ensure the sample size does not exceed the number of available files
    sample_size = min(sample_size, len(all_files))
    
    # Set the seed for reproducibility
    random.seed(42)
    
    # Randomly select the test files
    test_files = random.sample(all_files, sample_size)
    
    # Debugging: Print selected test files
    print(f"Selected test files: {test_files}")
    
    # Move selected test files and their corresponding image files to the test folder
    moved_files_count = 0
    for test_file in test_files:
        # Move JSON file
        json_src_path = os.path.join(root_dir, test_file)
        json_dst_path = os.path.join(output_test_dir, test_file)
        
        print(f"Moving JSON file: {json_src_path} -> {json_dst_path}")
        shutil.move(json_src_path, json_dst_path)
        moved_files_count += 1
        
        # Move corresponding image file (assuming the image path is stored in the JSON)
        with open(json_dst_path, 'r') as f:
            data = json.load(f)
            image_filename = data.get('imagePath')  # Using .get to avoid errors if key is missing
            if not image_filename:
                print(f"Error: 'imagePath' not found in {test_file}")
                continue
            
            image_src_path = os.path.join(root_dir, image_filename)
            image_dst_path = os.path.join(output_test_dir, image_filename)
            
            if os.path.exists(image_src_path):
                print(f"Moving image file: {image_src_path} -> {image_dst_path}")
                shutil.move(image_src_path, image_dst_path)
            else:
                print(f"Image file not found: {image_src_path}")

    print(f"Moved {moved_files_count} JSON files to {output_test_dir}")

# Example usage:
dataset_root_dir = 'log-labelled'
output_test_dir = 'test_data_log'
get_random_test_samples_and_move(dataset_root_dir, output_test_dir)
