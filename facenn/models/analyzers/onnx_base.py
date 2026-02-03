import onnxruntime as ort
import numpy as np
import torch
from abc import ABC, abstractmethod

class ONNXAnalyzer(ABC):
    def __init__(self, model_name: str, input_shape: tuple):
        self.model_name = model_name
        self.input_shape = input_shape
        self.session = None
        self.providers = self._get_best_providers()

    def _get_best_providers(self):
        available = ort.get_available_providers()
        providers = []
        if 'CoreMLExecutionProvider' in available:
            providers.append('CoreMLExecutionProvider')
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers

    @abstractmethod
    def load_model(self):
        pass

    def predict(self, face_img: torch.Tensor):
        if self.session is None:
            self.load_model()
        
        # Convert torch tensor to numpy
        if isinstance(face_img, torch.Tensor):
            face_img = face_img.detach().cpu().numpy()
        
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: face_img})
        
        res_tensors = []
        for o in outputs:
            t = torch.tensor(o)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            res_tensors.append(t)
            
        return res_tensors if len(res_tensors) > 1 else res_tensors[0]
