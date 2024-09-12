import os
import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import json

def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def load_model(model_path, num_classes=2):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def process_image(image_path, model, device, confidence_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    masks = prediction[0]['masks'].cpu().numpy()
    
    # Filter out low confidence predictions
    high_conf_indices = np.where(scores > confidence_threshold)[0]
    boxes = boxes[high_conf_indices]
    scores = scores[high_conf_indices]
    masks = masks[high_conf_indices]
    
    return len(boxes), boxes, scores, masks

def visualize_and_save(image_path, boxes, scores, masks, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for box, score, mask in zip(boxes, scores, masks):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Log: {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        mask = mask.squeeze()
        mask = (mask > 0.5).astype(np.uint8) * 255
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask == 255] = [0, 0, 255]
        image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
    
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def count_logs_in_images(model_path, image_dir, output_dir):
    model, device = load_model(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            count, boxes, scores, masks = process_image(image_path, model, device)
            
            output_path = os.path.join(output_dir, f"detected_{image_name}")
            visualize_and_save(image_path, boxes, scores, masks, output_path)
            
            results[image_name] = {
                "count": int(count),
                "output_image": output_path
            }
            
            print(f"Detected {count} logs in {image_name}")
    
    # Save results to JSON
    with open(os.path.join(output_dir, 'log_counts.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    model_path = "log_mask_rcnn.pth"  # Path to  trained model
    image_dir = "test_data_log"  # Directory containing test images
    output_dir = "log_count"  # Directory to save results
    
    count_logs_in_images(model_path, image_dir, output_dir)
    print(f"Results saved in {output_dir}")