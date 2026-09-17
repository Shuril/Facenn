import onnxruntime as ort
import torch
from facenn.models.analyzers.onnx_base import ONNXAnalyzer
from facenn.config import Config
from facenn.utils.io import download_file_from_url
import os

class HSEmotionONNX(ONNXAnalyzer):
    def __init__(self):
        super().__init__(model_name="HSEmotion-ONNX", input_shape=(224, 224))
        self.emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        
    def load_model(self):
        url = "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx?raw=true"
        weights_path = Config.get_weights_path("HSEmotion", "enet_b0_8_best_vgaf.onnx")

        if not os.path.exists(weights_path):
            download_file_from_url(url, weights_path)

        self.session = ort.InferenceSession(weights_path, providers=self.providers)

    def analyze(self, face_img: torch.Tensor):
        output = self.predict(face_img)
        idx = torch.argmax(output, dim=1).item()
        return self.emotion_labels[idx]
