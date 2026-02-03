import onnxruntime as ort
import torch
from facenn.models.analyzers.onnx_base import ONNXAnalyzer
from facenn.config import Config
from facenn.utils.io import download_file_from_url
import os

class FairFaceONNX(ONNXAnalyzer):
    def __init__(self):
        super().__init__(model_name="FairFace-ONNX", input_shape=(224, 224))
        self.race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        self.gender_labels = ['Male', 'Female']
        self.age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        
    def load_model(self):
        # Source: https://github.com/yakhyo/fairface-onnx
        url = "https://github.com/yakhyo/fairface-onnx/releases/download/weights/fairface.onnx"
        weights_path = Config.get_weights_path("FairFace", "fairface.onnx")
        
        try:
             download_file_from_url(url, weights_path)
             if os.path.exists(weights_path):
                 # Prefer CoreML on Mac if available? 
                 # For now, stick to CPU/Default which is usually fast enough in ONNX
                 self.session = ort.InferenceSession(weights_path, providers=self.providers)
             else:
                 print("Warning: Failed to download FairFace ONNX weights.")
        except Exception as e:
             print(f"Error loading FairFace ONNX: {e}")

    def analyze(self, face_img: torch.Tensor):
        outputs = self.predict(face_img)
        # Assuming order from yakhyo/fairface-onnx based on debug:
        # outputs[0]: Race (1, 7)
        # outputs[1]: Gender (1, 2)
        # outputs[2]: Age (1, 9)
        
        race_scores = outputs[0]
        gender_scores = outputs[1]
        age_scores = outputs[2]
        
        race_idx = torch.argmax(race_scores, dim=1).item()
        gender_idx = torch.argmax(gender_scores, dim=1).item()
        age_idx = torch.argmax(age_scores, dim=1).item()
        
        return {
            "race": self.race_labels[race_idx],
            "gender": self.gender_labels[gender_idx],
            "age": self.age_labels[age_idx]
        }
