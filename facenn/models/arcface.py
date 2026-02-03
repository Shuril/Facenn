import os
import torch
import torch.nn as nn
from facenn.models.base import FaceRecognitionModel
from facenn.config import Config, DEVICE

from facenn.models.backbones.irse import IR_SE_50
from facenn.utils.io import download_file_from_url

class ArcFace(FaceRecognitionModel):
    def __init__(self):
        super().__init__(model_name="ArcFace", input_shape=(112, 112))
        
    def load_model(self):
        self.model = IR_SE_50(self.input_shape)
        
        # Weights from a public repo compatible with this architecture
        # Source: HuggingFace (AIRI-Institute)
        url = "https://huggingface.co/AIRI-Institute/StyleFeatureEditor/resolve/main/pretrained_models/model_ir_se50.pth"
        
        weights_path = Config.get_weights_path("ArcFace", "model_ir_se50.pth")
        
        try:
             download_file_from_url(url, weights_path)
             if os.path.exists(weights_path):
                 # Set weights_only=False to allow loading older pickles (std behavior in <2.6 but safer to be explicit)
                 # Note: This is required for many research weights.
                 try:
                    state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=False)
                 except TypeError:
                     # Fallback for older torch versions
                     state_dict = torch.load(weights_path, map_location=DEVICE)
                     
                 self.model.load_state_dict(state_dict)
             else:
                 print(f"Warning: Failed to download weights to {weights_path}")
        except Exception as e:
             print(f"Error loading ArcFace weights: {e}")
             
        self.model.to(DEVICE)
        self.model.eval()
             
    def forward(self, x):
        return self.model(x)


