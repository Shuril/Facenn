import torch
from facenn.models.base import FaceRecognitionModel
from facenn.config import Config, DEVICE
import os

class EdgeFace(FaceRecognitionModel):
    def __init__(self, model_name="edgeface_s_gamma_05"):
        # EdgeFace usually outputs 512-dim embeddings
        super().__init__(model_name=model_name, input_shape=(112, 112))
        
    def load_model(self):
        hub_dir = os.path.join(Config.ensure_facenn_home(), "hub")
        torch.hub.set_dir(hub_dir)
        try:
            self.model = torch.hub.load("otroshi/edgeface", self.model_name, source="github")
            self.model.to(DEVICE)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to load EdgeFace model '{self.model_name}': {exc}") from exc

    def forward(self, x):
        return self.model(x)
