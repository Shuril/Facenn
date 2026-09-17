from typing import Optional
import os
import cv2
import numpy as np
import torch

from facenn.config import Config
from facenn.models.base import FaceRecognitionModel
from facenn.utils.io import download_file_from_url

SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
SFACE_URLS = [
    Config.get_release_weights_url(SFACE_FILENAME),
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
]


class SFace(FaceRecognitionModel):
    """
    SFace: Lightweight face recognition model from OpenCV Zoo (128-dim embeddings).
    Runs directly in OpenCV DNN without extra deep learning runtime overhead.
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_name="SFace", input_shape=(112, 112))
        self.model_path = model_path
        self.recognizer: Optional[cv2.FaceRecognizerSF] = None

    def load_model(self):
        weights_path = self.model_path or Config.get_weights_path("SFace", SFACE_FILENAME)
        if not os.path.exists(weights_path):
            download_file_from_url(SFACE_URLS, weights_path)

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=weights_path,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SFace uses OpenCV DNN engine; call predict() instead")

    def predict(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if self.recognizer is None:
            self.load_model()

        batch_embeddings = []
        cpu_tensors = img_tensor.detach().cpu()
        for i in range(cpu_tensors.shape[0]):
            arr = cpu_tensors[i].numpy().transpose(1, 2, 0)
            if arr.min() < 0:
                arr = (arr * 128.0 + 127.5).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

            feat = self.recognizer.feature(arr)
            norm = np.linalg.norm(feat, axis=-1, keepdims=True)
            norm = np.maximum(norm, 1e-12)
            feat = feat / norm
            batch_embeddings.append(feat)

        feats_np = np.vstack(batch_embeddings).astype(np.float32)
        return torch.from_numpy(feats_np).to(img_tensor.device)
