import os
import random
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as KE  # Adjusted import for TensorFlow 2.x
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import utils
from mrcnn import visualize

# Step 1: Dataset Preparation

# Define paths
dataset_dir = 'log-labelled'
test_dir = 'test_data_log'

# Ensure the test directory exists
os.makedirs(test_dir, exist_ok=True)

# Select 10 random images from the dataset for testing
def select_random_files(source_dir, target_dir, num_files=10):
    files = os.listdir(source_dir)
    selected_files = random.sample(files, num_files)
    
    for file in selected_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))

# Move 10 images to the test set
select_random_files(dataset_dir, test_dir, num_files=10)

# Step 2: Mask R-CNN Configuration and Dataset

class LogConfig(Config):
    NAME = "log"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + Log
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

config = LogConfig()

# Step 3: Define Dataset Class for Training
class LogDataset(utils.Dataset):
    def load_logs(self, dataset_dir):
        self.add_class("dataset", 1, "log")
        images = os.listdir(dataset_dir)
        for i, image_file in enumerate(images):
            image_id = os.path.splitext(image_file)[0]
            self.add_image("dataset", image_id=i, path=os.path.join(dataset_dir, image_file))

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = np.ones([info['height'], info['width']], dtype=np.bool)
        class_ids = np.array([1])
        return mask, class_ids

# Step 4: Load Dataset
dataset_train = LogDataset()
dataset_train.load_logs(dataset_dir)
dataset_train.prepare()

# Step 5: Initialize Mask R-CNN Model
model = MaskRCNN(mode="training", model_dir="./", config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                            "mrcnn_bbox", "mrcnn_mask"])

# Step 6: Train Model
model.train(dataset_train, dataset_train, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

# Step 7: Test Model on Test Set

# Reconfigure for inference mode
inference_config = LogConfig()
inference_config.GPU_COUNT = 1
inference_config.IMAGES_PER_GPU = 1

model_inference = MaskRCNN(mode="inference", model_dir="./", config=inference_config)
model_inference.load_weights('mask_rcnn_coco.h5', by_name=True)

# Evaluate the model on the test set
test_images = os.listdir(test_dir)

for image_file in test_images:
    image = cv2.imread(os.path.join(test_dir, image_file))
    results = model_inference.detect([image], verbose=0)
    r = results[0]
    
    # Display the result
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ['BG', 'log'], r['scores'])

    # Save the output image
    output_file = f"output_{image_file}"
    cv2.imwrite(output_file, image)

print("Task 2 Completed: Mask R-CNN model trained and tested on log images.")
