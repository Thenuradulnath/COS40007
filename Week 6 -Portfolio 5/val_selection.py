import os
import random
import shutil

def select_training_images(train_images_dir, train_labels_dir, selected_images_dir, selected_labels_dir):
    os.makedirs(selected_images_dir, exist_ok=True)
    os.makedirs(selected_labels_dir, exist_ok=True)

    image_filenames = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
    selected_images = random.sample(image_filenames, 400)

    for img in selected_images:
        shutil.copy(os.path.join(train_images_dir, img), selected_images_dir)
        label_file = img.replace('.jpg', '.txt')
        shutil.copy(os.path.join(train_labels_dir, label_file), selected_labels_dir)

    print("Training images and labels selected.")

# Select 400 random training images and their labels
select_training_images('images/train', 'bounding_boxes/gt_boxes_yolo_train', 'dataset/train/images', 'dataset/train/labels')
