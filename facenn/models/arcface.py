import os
import torch
import torch.nn as nn
from facenn.models.base import FaceRecognitionModel
from facenn.config import Config, DEVICE

from facenn.models.backbones.irse import IR_SE_50
from facenn.utils.io import download_file_from_url

ARCFACE_URLS = [
    Config.get_release_weights_url("arcface_ir50.pth"),
    "https://huggingface.co/AIRI-Institute/StyleFeatureEditor/resolve/main/pretrained_models/model_ir_se50.pth",
]


class ArcFace(FaceRecognitionModel):
    def __init__(self):
        super().__init__(model_name="ArcFace", input_shape=(112, 112))

    def load_model(self):
        self.model = IR_SE_50(self.input_shape)
        weights_path = Config.get_weights_path("ArcFace", "arcface_ir50.pth")
        legacy_weights_path = Config.get_weights_path("ArcFace", "model_ir_se50.pth")

        target_path = legacy_weights_path if os.path.exists(legacy_weights_path) else weights_path
        if not os.path.exists(target_path):
            download_file_from_url(ARCFACE_URLS, target_path)

        try:
            state_dict = torch.load(target_path, map_location=DEVICE, weights_only=False)
        except TypeError:
            state_dict = torch.load(target_path, map_location=DEVICE)

        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()
             
    def forward(self, x):
        return self.model(x)


