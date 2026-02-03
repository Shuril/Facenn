import torch
from facenn.models.base import FaceRecognitionModel
from facenn.config import Config, DEVICE
import os

class EdgeFace(FaceRecognitionModel):
    def __init__(self, model_name="edgeface_s_gamma_05"):
        # EdgeFace usually outputs 512-dim embeddings
        super().__init__(model_name=model_name, input_shape=(112, 112))
        
    def load_model(self):
        print(f"Loading {self.model_name} via torch.hub...")
        # Redirect torch hub cache to project weights
        hub_dir = os.path.join(Config.FACENN_HOME, "hub")
        torch.hub.set_dir(hub_dir)
        
        try:
            # Load from official repo
            self.model = torch.hub.load('otroshi/edgeface', self.model_name, source='github')
            self.model.to(DEVICE)
            self.model.eval()
        except Exception as e:
            print(f"Error loading EdgeFace: {e}")
            print("Trying fallback or check internet connection.")
            # Fallback or re-raise
            raise e

    def forward(self, x):
        return self.model(x)
