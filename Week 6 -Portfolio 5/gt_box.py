import os
import pandas as pd

# Function to convert bounding boxes into YOLO format
def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

# Convert CSV annotations to YOLO format and save to .txt files
def create_yolo_labels(csv_file, output_dir):
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        image_name = row['filename']
        img_width = row['width']
        img_height = row['height']
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

        # Convert to YOLO format
        x_center, y_center, width, height = convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height)
        
        # Write YOLO annotation
        yolo_label = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        txt_file = os.path.join(output_dir, image_name.replace('.jpg', '.txt'))

        with open(txt_file, 'a') as f:
            f.write(yolo_label)

    print(f"YOLO format conversion completed for {csv_file}")

# Convert annotations for training and test sets
create_yolo_labels('bounding_boxes\train_labels.csv', 'yolo_annotations/train')
create_yolo_labels('bounding_boxes\test_labels.csv', 'yolo_annotations/test')
