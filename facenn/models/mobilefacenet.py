from typing import Optional
import os
import numpy as np
import torch

from facenn.config import Config
from facenn.models.base import FaceRecognitionModel
from facenn.utils.io import download_file_from_url

MBF_FILENAME = "w600k_mbf.onnx"
MBF_URLS = [
    Config.get_release_weights_url(MBF_FILENAME),
]


class MobileFaceNet(FaceRecognitionModel):
    """
    MobileFaceNet: Lightweight face recognition model (512-dim embeddings).
    Trained on WebFace600K dataset for fast and accurate face recognition on CPU and mobile.
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_name="MobileFaceNet", input_shape=(112, 112))
        self.model_path = model_path
        self.session = None
        self.input_name = None

    def load_model(self):
        import onnxruntime as ort

        weights_path = self.model_path or Config.get_weights_path("MobileFaceNet", MBF_FILENAME)
        if not os.path.exists(weights_path):
            download_file_from_url(MBF_URLS, weights_path)

        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(weights_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("MobileFaceNet runs on ONNX runtime; use predict() directly")

    def predict(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if self.session is None:
            self.load_model()

        np_input = img_tensor.detach().cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {self.input_name: np_input})
        emb_np = outputs[0]

        norm = np.linalg.norm(emb_np, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        emb_np = emb_np / norm

        return torch.from_numpy(emb_np).to(img_tensor.device)


class MobileFaceNetV2(MobileFaceNet):
    """Backwards compatibility alias for MobileFaceNet."""
    pass
