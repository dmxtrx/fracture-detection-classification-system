import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        def res_block(channels):
            return nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                                 nn.GroupNorm(32, channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                                 nn.GroupNorm(32, channels))
        
        def downsample(in_c, out_c):
            return nn.Sequential(nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False),
                                 nn.GroupNorm(32, out_c),
                                 nn.ReLU(inplace=True))

        self.start = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        
        self.block1 = res_block(64)
        self.down1 = downsample(64, 128)
        
        self.block2 = res_block(128)
        self.down2 = downsample(128, 256)
        
        self.block3 = res_block(256)
        self.down3 = downsample(256, 512)
        
        self.block4 = res_block(512)

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 1))

    def forward(self, x):
        x = self.start(x)
        x = torch.relu(x + self.block1(x))
        x = self.down1(x)   
        x = torch.relu(x + self.block2(x))
        
        x = self.down2(x)
        x = torch.relu(x + self.block3(x))
        x = self.down3(x)
        x = torch.relu(x + self.block4(x))
        return self.classifier(x)

class FractureClassifier:
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        self.model = CNN().to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            self.model.load_state_dict(state_dict['model_state_dict'])
        elif isinstance(state_dict, dict):
            self.model.load_state_dict(state_dict)
        else:
            self.model = state_dict

        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(img_tensor)
            prob_normal = torch.sigmoid(logit).item()
            
            if prob_normal > 0.5:
                pred_class = "Normal"
                prob = prob_normal
            else:
                pred_class = "Fracture"
                prob = 1.0 - prob_normal

        return pred_class, prob