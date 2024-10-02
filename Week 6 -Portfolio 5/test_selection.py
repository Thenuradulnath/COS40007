import os
import random
import shutil

def select_test_images(test_images_dir, selected_test_images_dir):
    os.makedirs(selected_test_images_dir, exist_ok=True)

    test_image_filenames = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
    selected_test_images = random.sample(test_image_filenames, 40)

    for img in selected_test_images:
        shutil.copy(os.path.join(test_images_dir, img), selected_test_images_dir)

    print("Test images selected.")

# Select 40 random test images
select_test_images('images/test', 'dataset/test/images')
