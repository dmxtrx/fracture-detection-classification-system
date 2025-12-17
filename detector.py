import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        self.conv_last = nn.Conv2d(64, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        flat = self.avgpool(conv4).view(conv4.size(0), -1)
        class_logit = self.classifier(flat)
        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        mask_logit = self.conv_last(x)
        return mask_logit, class_logit

class FractureDetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = UNet()
        if device == "cuda" and torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(device)
        self.model.eval()
        self.img_size = 640
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def non_max_suppression(self, boxes, scores, iou_threshold=0.3):
        if not boxes: return []
        boxes = np.array(boxes)
        scores = np.array(scores)
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        area = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(scores)
        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            union = area[i] + area[idxs[:last]] - intersection
            iou = intersection / (union + 1e-6)
            idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > iou_threshold)[0])))
        return boxes[pick].tolist()

    def predict(self, image_path: str, save_path: str = None) -> dict:
        img_pil = Image.open(image_path).convert("RGB")
        w_orig, h_orig = img_pil.size
        window_size = self.img_size
        stride = int(window_size * 0.75)
        full_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
        count_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
        
        if w_orig <= window_size * 1.2 and h_orig <= window_size * 1.2:
            img_resized = img_pil.resize((self.img_size, self.img_size))
            img_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mask_logit, _ = self.model(img_tensor)
                pred_mask = torch.sigmoid(mask_logit).cpu().numpy()[0, 0]
                full_mask = cv2.resize(pred_mask, (w_orig, h_orig))
        else:
            img_np = np.array(img_pil)
            for y in range(0, h_orig, stride):
                for x in range(0, w_orig, stride):
                    y_end = min(y + window_size, h_orig)
                    x_end = min(x + window_size, w_orig)
                    y_start = max(0, y_end - window_size)
                    x_start = max(0, x_end - window_size)
                    patch = img_np[y_start:y_end, x_start:x_end]
                    patch_pil = Image.fromarray(patch).resize((self.img_size, self.img_size))
                    patch_tensor = self.transform(patch_pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        mask_logit, _ = self.model(patch_tensor)
                        pred_patch = torch.sigmoid(mask_logit).cpu().numpy()[0, 0]
                        pred_patch_resized = cv2.resize(pred_patch, (x_end - x_start, y_end - y_start))
                        full_mask[y_start:y_end, x_start:x_end] += pred_patch_resized
                        count_mask[y_start:y_end, x_start:x_end] += 1.0
            full_mask /= np.maximum(count_mask, 1.0)

        mask_uint8 = (full_mask > 0.6).astype(np.uint8)
        kernel = np.ones((5,5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = {"has_fracture": False, "confidence": 0.0, "box": None}
        detected_boxes = []
        box_scores = []

        if contours:
            for cnt in contours:
                if cv2.contourArea(cnt) > (w_orig * h_orig) * 0.005:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if 0.15 < float(w)/h < 8.0:
                        detected_boxes.append([x, y, x+w, y+h])
                        box_scores.append(float(full_mask[y:y+h, x:x+w].mean()))

            final_boxes = self.non_max_suppression(detected_boxes, box_scores, iou_threshold=0.2)
            
            if final_boxes:
                result["has_fracture"] = True
                result["box"] = max(final_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                x, y, x2, y2 = map(int, result["box"])
                result["confidence"] = float(full_mask[y:y2, x:x2].mean())

                if save_path:
                    cv_img = cv2.imread(image_path)
                    cv2.rectangle(cv_img, (x, y), (x2, y2), (0, 0, 255), 4)
                    cv2.imwrite(save_path, cv_img)
                    
        return result
