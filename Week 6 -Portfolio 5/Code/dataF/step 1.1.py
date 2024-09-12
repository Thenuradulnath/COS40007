import os
import shutil
import random

# Directories
rust_dir = 'Corrosion/rust'
no_rust_dir = 'Corrosion/no rust'
test_dir = 'test_data'

# Create test directories if they don't exist
os.makedirs(f'{test_dir}/rust', exist_ok=True)
os.makedirs(f'{test_dir}/no rust', exist_ok=True)

# Function to move random files to the test set
def move_random_files(source_dir, target_dir, num_files):
    files = os.listdir(source_dir)
    selected_files = random.sample(files, num_files)
    
    for file in selected_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))

# Move 10 images from rust and no rust to test set
move_random_files(rust_dir, f'{test_dir}/rust', 10)
move_random_files(no_rust_dir, f'{test_dir}/no rust', 10)

print("Test set created with 10 'rust' and 10 'no rust' images.")
