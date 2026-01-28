import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

class FractureDetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.confidence_threshold = 0.5
        self.model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        
        if device == "cuda" and torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([transforms.ToTensor(),])

    def predict(self, image_path: str, save_path: str = None) -> dict:
        try:
            img_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image: {e}")
            return {"has_fracture": False, "confidence": 0.0, "box": None}

        img_tensor = self.transform(img_pil).to(self.device)
        with torch.no_grad():
            predictions = self.model([img_tensor])

        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        best_box = None
        best_score = 0.0
        has_fracture = False
        valid_indices = np.where(scores > self.confidence_threshold)[0]
        
        if len(valid_indices) > 0:
            max_score_idx = valid_indices[np.argmax(scores[valid_indices])]
            best_score = float(scores[max_score_idx])
            best_box = boxes[max_score_idx].astype(int).tolist() # [x1, y1, x2, y2]
            has_fracture = True
        result = {"has_fracture": has_fracture, "confidence": best_score, "box": best_box}

        if save_path:
            cv_img = cv2.imread(image_path)
            if cv_img is not None:
                if has_fracture:
                    x1, y1, x2, y2 = best_box
                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    label = f"Fracture: {best_score:.2f}"
                    cv2.putText(cv_img, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(save_path, cv_img)

        return result
