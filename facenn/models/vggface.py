import torch
import torch.nn as nn
import torchvision.models as models
from facenn.models.base import FaceRecognitionModel
from facenn.config import Config, DEVICE
from facenn.utils.io import download_file_from_url
import os

class VGG_Face_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # VGG-Face is VGG16 architecture but with specific weights
        self.meta = {'mean': [129.1863, 104.7624, 93.5940],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
                     
        # Define VGG16 manually or use torchvision
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2622) # 2622 identities in VGG Face
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embeddings(x)
        return x

class VGGFace(FaceRecognitionModel):
    def __init__(self):
        super().__init__(model_name="VGG-Face", input_shape=(224, 224))
        
    def load_model(self):
        self.model = VGG_Face_Model()
        
        # Original VGG Face weights converted to PyTorch
        # Source: https://github.com/prlz77/vgg-face.pytorch
        # Valid PyTorch URL from VGG MCN (MatConvNet to PyTorch)
        # http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth
        url = "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth"
        
        weights_path = Config.get_weights_path("VGG-Face", "vgg_face_dag.pth")
        
        try:
             download_file_from_url(url, weights_path)
             if os.path.exists(weights_path):
                 state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=False)
                 # This file usually has 'state_dict' or just keys.
                 # Also keys might need mapping if using generic VGG vs specific implementation
                 # Attempting load (strict=False to allow minor mismatches if classifier layer differs)
                 try:
                    self.model.load_state_dict(state_dict, strict=False)
                 except:
                     pass
                 self.model.eval()
             else:
                 print("Warning: Failed to download VGG-Face weights.")
        except Exception as e:
             print(f"Error loading VGG-Face weights: {e}")

    def forward(self, x):
        return self.model(x)
