from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseAnalyzer(ABC):
    def __init__(self, model_name: str, input_shape: tuple):
        self.model_name = model_name
        self.input_shape = input_shape
        self.model = None

    @abstractmethod
    def load_model(self):
        pass

    def predict(self, face_img: torch.Tensor):
        if self.model is None:
            self.load_model()
        
        with torch.no_grad():
            output = self.model(face_img)
        return output
