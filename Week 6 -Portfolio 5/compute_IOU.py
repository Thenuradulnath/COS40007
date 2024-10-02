import os
import pandas as pd

def compute_iou(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)
    intersection = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area_box1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area_box2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union = area_box1 + area_box2 - intersection
    return intersection / union if union != 0 else 0

def compute_and_save_iou(ground_truth_dir, prediction_dir, output_csv):
    iou_results = []

    for file_name in os.listdir(ground_truth_dir):
        gt_file = os.path.join(ground_truth_dir, file_name)
        pred_file = os.path.join(prediction_dir, file_name)

        if os.path.exists(pred_file):
            with open(gt_file, 'r') as f_gt, open(pred_file, 'r') as f_pred:
                gt_box = list(map(float, f_gt.readline().split()[1:]))
                pred_box = list(map(float, f_pred.readline().split()[1:]))
                iou = compute_iou(gt_box, pred_box)
                iou_results.append({'filename': file_name, 'confidence': 1.0, 'IoU': iou})
        else:
            iou_results.append({'filename': file_name, 'confidence': 0.0, 'IoU': 0.0})

    df = pd.DataFrame(iou_results)
    df.to_csv(output_csv, index=False)
    print(f"IoU results saved to {output_csv}")

# Compute IoU and save results
compute_and_save_iou('bounding_boxes/gt_boxes_yolo_test', 'predictions', 'results/iou_results.csv')
