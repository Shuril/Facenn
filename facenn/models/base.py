from abc import ABC, abstractmethod
from typing import Optional, Tuple
import logging
import os
import torch
import torch.nn as nn

from facenn.config import Config, DEVICE

logger = logging.getLogger("facenn")


class FaceRecognitionModel(ABC):
    def __init__(self, model_name: str, input_shape: Tuple[int, int] = (112, 112)):
        self.model_name = model_name
        self.input_shape = input_shape
        self.model: Optional[nn.Module] = None
        self.backend = "torch"  # 'torch' or 'onnx'
        self.session = None

    @abstractmethod
    def load_model(self):
        """Loads model weights."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract face embeddings."""
        pass

    def predict(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Runs model inference and returns L2-normalized embedding tensor.
        """
        if self.backend == "onnx":
            return self._predict_onnx(img_tensor)

        if self.model is None:
            self.load_model()
            self.model.to(DEVICE)
            self.model.eval()

            if DEVICE.type == "cpu" and Config.USE_OPENVINO:
                try:
                    from facenn.utils.openvino_utils import optimize_for_openvino

                    example = torch.randn(1, 3, self.input_shape[0], self.input_shape[1])
                    self.model = optimize_for_openvino(self.model, example)
                except Exception as exc:
                    logger.warning("Failed to optimize model with OpenVINO: %s", exc)

        with torch.no_grad():
            emb = self.model(img_tensor.to(DEVICE))
            # Normalize embedding vector
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb

    def _predict_onnx(self, img_tensor: torch.Tensor) -> torch.Tensor:
        import onnxruntime as ort

        if self.session is None:
            onnx_path = self.ensure_onnx()
            providers = ["CPUExecutionProvider"]
            available = ort.get_available_providers()
            if "CoreMLExecutionProvider" in available:
                providers.insert(0, "CoreMLExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")

            self.session = ort.InferenceSession(onnx_path, providers=providers)

        input_name = self.session.get_inputs()[0].name
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.detach().cpu().numpy()
        else:
            img_np = img_tensor

        outputs = self.session.run(None, {input_name: img_np})
        out_tensor = torch.tensor(outputs[0]).to(DEVICE)
        return torch.nn.functional.normalize(out_tensor, p=2, dim=-1)

    def ensure_onnx(self) -> str:
        """
        Ensures an ONNX file exists for this model.
        Exports PyTorch model if not present.
        """
        from facenn.utils.onnx_export import export_to_onnx

        onnx_name = f"{self.model_name}.onnx"
        weights_path = Config.get_weights_path(self.model_name, onnx_name)

        if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
            return weights_path

        logger.info("Exporting %s to ONNX format...", self.model_name)
        if self.model is None:
            self.load_model()

        success = export_to_onnx(
            self.model,
            weights_path,
            input_shape=(1, 3, self.input_shape[0], self.input_shape[1]),
        )

        if success and os.path.exists(weights_path):
            return weights_path

        raise RuntimeError(f"Failed to export model {self.model_name} to ONNX.")

