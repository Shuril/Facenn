import torch
import torch.nn as nn
import torchvision.models as models
from facenn.models.analyzers.base import BaseAnalyzer
from facenn.config import Config, DEVICE
from facenn.utils.io import download_file_from_url
import os

class FairFaceModel(BaseAnalyzer):
    def __init__(self):
        super().__init__(model_name="FairFace", input_shape=(224, 224))
        self.race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        self.gender_labels = ['Male', 'Female']
        self.age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        
    def load_model(self):
        # fairface usually uses resnet34
        self.model = models.resnet34(weights=None)
        
        # Override the last layer to output Race/Gender/Age
        # FairFace multi-task output: 7 (race) + 2 (gender) + 9 (age) = 18 outputs
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 18)
        self.model.to(DEVICE)
        
        # Download weights from official FairFace Google Drive
        # ID: 11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB
        url = "https://drive.google.com/uc?id=11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB"
        weights_path = Config.get_weights_path("FairFace", "res34_fair_align_multi_7.pt")
        
        try:
             download_file_from_url(url, weights_path)
             if os.path.exists(weights_path):
                 # FairFace weights often have a different structure
                 checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
                 # Handle if state_dict is inside a key
                 if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                     self.model.load_state_dict(checkpoint['state_dict'])
                 else:
                     self.model.load_state_dict(checkpoint)
                 self.model.eval()
             else:
                 print("Warning: Failed to download FairFace weights.")
        except Exception as e:
             print(f"Error loading FairFace weights: {e}")

    def analyze(self, face_img: torch.Tensor):
        output = self.predict(face_img)
        # output shape [B, 18]
        # split into race [0:7], gender [7:9], age [9:18]
        race_scores = output[:, 0:7]
        gender_scores = output[:, 7:9]
        age_scores = output[:, 9:18]
        
        race_idx = torch.argmax(race_scores, dim=1).item()
        gender_idx = torch.argmax(gender_scores, dim=1).item()
        age_idx = torch.argmax(age_scores, dim=1).item()
        
        return {
            "race": self.race_labels[race_idx],
            "gender": self.gender_labels[gender_idx],
            "age": self.age_labels[age_idx]
        }
