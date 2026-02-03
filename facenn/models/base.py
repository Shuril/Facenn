from abc import ABC, abstractmethod
import torch.nn as nn
from facenn.config import DEVICE

class FaceRecognitionModel(ABC):
    def __init__(self, model_name: str, input_shape: tuple):
        self.model_name = model_name
        self.input_shape = input_shape
        self.model = None
        self.backend = 'torch' # 'torch' or 'onnx'
        self.session = None

    @abstractmethod
    def load_model(self):
        """Loads partial or full model weights."""
        pass

    @abstractmethod
    def forward(self, x):
        """Forward pass to extract embeddings."""
        pass
        
    def predict(self, img_tensor):
        """Standard prediction wrapper."""
        if self.backend == 'onnx':
            return self._predict_onnx(img_tensor)

        if self.model is None:
            self.load_model()
            self.model.to(DEVICE)
            self.model.eval()
            
            # Optional: Automatic OpenVINO optimization if requested
            # This is a placeholder for where one would trigger it
            if DEVICE.type == 'cpu' and Config.USE_OPENVINO: 
                 from facenn.utils.openvino_utils import optimize_for_openvino
                 # We need an example input
                 example = torch.randn(1, 3, self.input_shape[0], self.input_shape[1])
                 self.model = optimize_for_openvino(self.model, example)
            
        return self.model(img_tensor.to(DEVICE))
        
    def _predict_onnx(self, img_tensor):
        import onnxruntime as ort
        import numpy as np
        
        if self.session is None:
            onnx_path = self.ensure_onnx()
            
            providers = ['CPUExecutionProvider']
            if 'CoreMLExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CoreMLExecutionProvider')
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            
        # Prepare input
        # img_tensor is usually (1, 3, H, W) torch tensor
        input_name = self.session.get_inputs()[0].name
        
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.detach().cpu().numpy()
        else:
            img_np = img_tensor
            
        outputs = self.session.run(None, {input_name: img_np})
        
        # Return as tensor to match torch behavior
        return torch.tensor(outputs[0]).to(DEVICE)

    def ensure_onnx(self):
        """
        Ensures an ONNX version of this model exists.
        If not, exports the PyTorch model.
        Returns the path to the ONNX model.
        """
        from facenn.config import Config
        import os
        from facenn.utils.onnx_export import export_to_onnx
        
        onnx_name = f"{self.model_name}.onnx"
        weights_path = Config.get_weights_path(self.model_name, onnx_name)
        
        if os.path.exists(weights_path):
            return weights_path
            
        print(f"ONNX model for {self.model_name} not found. Exporting...")
        
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
            
        # Export
        success = export_to_onnx(
            self.model, 
            weights_path, 
            input_shape=(1, 3, self.input_shape[0], self.input_shape[1])
        )
        
        if success:
            return weights_path
        else:
            raise RuntimeError(f"Failed to export {self.model_name} to ONNX.")
