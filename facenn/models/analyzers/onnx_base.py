from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort
import torch


class ONNXAnalyzer(ABC):
    def __init__(self, model_name: str, input_shape: Tuple[int, int]):
        self.model_name = model_name
        self.input_shape = input_shape
        self.session: Optional[ort.InferenceSession] = None
        self.providers = self._get_best_providers()

    def _get_best_providers(self) -> List[str]:
        available = ort.get_available_providers()
        providers = []
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    @abstractmethod
    def load_model(self):
        pass

    def predict(self, face_img: Union[torch.Tensor, np.ndarray]):
        if self.session is None:
            self.load_model()

        if isinstance(face_img, torch.Tensor):
            img_np = face_img.detach().cpu().numpy()
        else:
            img_np = np.asarray(face_img, dtype=np.float32)

        if img_np.ndim == 3:
            img_np = np.expand_dims(img_np, axis=0)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_np})

        res_tensors = []
        for o in outputs:
            t = torch.tensor(o)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            res_tensors.append(t)

        return res_tensors if len(res_tensors) > 1 else res_tensors[0]

