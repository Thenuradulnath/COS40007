import pandas as pd

# Path to the CSV file containing IoU results
iou_csv = 'results/iou_results.csv'

# Load the CSV into a DataFrame
df = pd.read_csv(iou_csv)

# Check how many images have IoU >= 0.9
iou_threshold = 0.9
high_iou_count = df[df['IoU'] >= iou_threshold].shape[0]
total_images = df.shape[0]

# Calculate the percentage of images with IoU >= threshold
percentage_high_iou = (high_iou_count / total_images) * 100
print(f"Percentage of images with IoU >= {iou_threshold}: {percentage_high_iou:.2f}%")
