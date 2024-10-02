import os
import random
import shutil

# Paths to the original dataset
train_images_dir = 'images/train'
train_labels_dir = 'bounding_boxes/gt_boxes_yolo'

# Paths for selected images and labels
selected_train_images_dir = 'dataset/train/images'
selected_train_labels_dir = 'dataset/train/labels'

# Create directories
os.makedirs(selected_train_images_dir, exist_ok=True)
os.makedirs(selected_train_labels_dir, exist_ok=True)

# Randomly select 400 images
image_filenames = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
selected_images = random.sample(image_filenames, 400)

# Copy selected images and labels
for img in selected_images:
    shutil.copy(os.path.join(train_images_dir, img), selected_train_images_dir)
    label_file = img.replace('.jpg', '.txt')
    shutil.copy(os.path.join(train_labels_dir, label_file), selected_train_labels_dir)

print("Training images and labels selected.")
