import onnxruntime as ort
import torch
from facenn.models.analyzers.onnx_base import ONNXAnalyzer
from facenn.config import Config
from facenn.utils.io import download_file_from_url
import os

EMOTION_FILENAME = "enet_b0_8_best_vgaf.onnx"
EMOTION_URLS = [
    Config.get_release_weights_url(EMOTION_FILENAME),
    "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx",
]


class HSEmotionONNX(ONNXAnalyzer):
    def __init__(self):
        super().__init__(model_name="HSEmotion-ONNX", input_shape=(224, 224))
        self.emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    def load_model(self):
        weights_path = Config.get_weights_path("HSEmotion", EMOTION_FILENAME)

        if not os.path.exists(weights_path):
            download_file_from_url(EMOTION_URLS, weights_path)

        self.session = ort.InferenceSession(weights_path, providers=self.providers)

    def analyze(self, face_img: torch.Tensor):
        output = self.predict(face_img)
        idx = torch.argmax(output, dim=1).item()
        return self.emotion_labels[idx]
