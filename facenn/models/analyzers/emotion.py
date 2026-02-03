import torch
import torch.nn as nn
import torchvision.models as models
from facenn.models.analyzers.base import BaseAnalyzer
from facenn.config import Config, DEVICE
from facenn.utils.io import download_file_from_url
import os

class HSEmotionModel(BaseAnalyzer):
    def __init__(self):
        super().__init__(model_name="HSEmotion", input_shape=(224, 224))
        self.emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        
    def load_model(self):
        # HSEmotion specifically uses EfficientNet-B0. Using timm for compatibility with official weights.
        import timm
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=8)
        self.model.to(DEVICE)
        
        # Source: https://github.com/sb-ai-lab/EmotiEffLib
        url = "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/enet_b0_8_best_vgaf.pt"
        
        weights_path = Config.get_weights_path("HSEmotion", "enet_b0_8_best_vgaf.pt")
        
        try:
             download_file_from_url(url, weights_path)
             if os.path.exists(weights_path):
                 checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
                 # The checkpoint usually contains just the state_dict or a dict with it
                 if isinstance(checkpoint, dict):
                     if 'state_dict' in checkpoint:
                         self.model.load_state_dict(checkpoint['state_dict'])
                     elif 'model' in checkpoint:
                         self.model.load_state_dict(checkpoint['model'])
                     else:
                         self.model.load_state_dict(checkpoint)
                 elif hasattr(checkpoint, 'state_dict'):
                     # If it's a full model object (e.g. from timm)
                     self.model.load_state_dict(checkpoint.state_dict())
                 else:
                     self.model.load_state_dict(checkpoint)
                 self.model.eval()
             else:
                 print("Warning: Failed to download HSEmotion weights.")
        except Exception as e:
             # Try a mirror if the first one fails
             print(f"Error loading HSEmotion weights: {e}")

    def analyze(self, face_img: torch.Tensor):
        # HSEmotion expects normalized 224x224 RGB
        output = self.predict(face_img)
        idx = torch.argmax(output, dim=1).item()
        return self.emotion_labels[idx]
