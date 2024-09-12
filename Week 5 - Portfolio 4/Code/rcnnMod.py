import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image, ImageDraw
import numpy as np
import cv2

class LogDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        all_files = [f for f in os.listdir(root_dir) if f.endswith('.json')]
        random.seed(42)
        self.test_files = random.sample(all_files, 10)
        self.train_files = [f for f in all_files if f not in self.test_files]
        
        self.files = self.train_files if is_train else self.test_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        json_path = os.path.join(self.root_dir, self.files[idx])
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        img_path = os.path.join(self.root_dir, data['imagePath'])
        img = Image.open(img_path).convert("RGB")
        
        masks = []
        boxes = []
        for shape in data['shapes']:
            points = shape['points']
            mask = self.create_mask(img.size, points)
            masks.append(mask)
            boxes.append(self.get_bounding_box(points))

        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # all objects are logs
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_path"] = img_path

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, target

    def create_mask(self, img_size, points):
        mask = Image.new('L', img_size, 0)
        draw = ImageDraw.Draw(mask)
        try:
            draw.polygon(points, fill=1)
        except ValueError:
            flattened_points = [coord for point in points for coord in point]
            draw.polygon(flattened_points, fill=1)
        return np.array(mask)

    def get_bounding_box(self, points):
        points = np.array(points).reshape(-1, 2)
        x_coordinates, y_coordinates = points[:, 0], points[:, 1]
        return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]

def get_model_instance_segmentation(num_classes):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def train_model(model, data_loader, device, num_epochs=10):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items() if k != 'image_path'} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
        
        print(f"Epoch {epoch} loss: {total_loss / len(data_loader)}")
    
    return model

def visualize_predictions(image, boxes, scores, masks, threshold=0.5):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Red, Blue
    
    for i, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
        if score >= threshold:
            x1, y1, x2, y2 = box.astype(int)
            color = colors[i % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f'log {score:.6f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            mask = mask.squeeze()
            mask = (mask > 0.5).astype(np.uint8) * 255
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            colored_mask[mask == 255] = color
            img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)
    
    return img

def test_and_visualize(model, test_loader, device, output_dir):
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for j, image in enumerate(images):
                orig_image = Image.open(targets[j]['image_path']).convert("RGB")
                
                boxes = outputs[j]['boxes'].cpu().numpy()
                scores = outputs[j]['scores'].cpu().numpy()
                masks = outputs[j]['masks'].cpu().numpy()
                
                result_image = visualize_predictions(orig_image, boxes, scores, masks)
                
                output_path = os.path.join(output_dir, f'result_{i}_{j}.jpg')
                cv2.imwrite(output_path, result_image)
                print(f"Saved result for image {i}_{j} to {output_path}")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Dataset
    dataset = LogDataset('log-labelled', is_train=True)
    dataset_test = LogDataset('test_data_log', is_train=False)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Model
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Training
    model = train_model(model, data_loader, device)

    # Save the model
    torch.save(model.state_dict(), 'log_mask_rcnn.pth')

    # Create output directory for test results
    output_dir = 'test_res_log'
    os.makedirs(output_dir, exist_ok=True)

    # Test and visualize
    test_and_visualize(model, data_loader_test, device, output_dir)

if __name__ == "__main__":
    main()