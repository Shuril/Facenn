import os
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("facenn")

class Config:
    # Use project-local weights folder instead of home directory cache
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FACENN_HOME = os.getenv("FACENN_HOME", os.path.join(BASE_DIR, "weights"))
    logger.info(f"Using weights directory: {FACENN_HOME}")
    USE_OPENVINO = os.getenv("FACENN_USE_OPENVINO", "0") == "1"
    
    @staticmethod
    def get_device(backend_preference=None):
        """
        Auto-detects the best available device or uses preference.
        Options: 'cuda', 'mps', 'vulkan', 'xpu', 'openvino', 'cpu'.
        """
        if backend_preference:
            return torch.device(backend_preference)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        elif hasattr(torch, 'xpu') and torch.xpu.is_available(): # Intel XPU
             return torch.device("xpu")
        elif torch.is_vulkan_available():
             return torch.device("vulkan")
        else:
            try:
                 import torch_directml
                 return torch_directml.device()
            except ImportError:
                pass
            return torch.device("cpu")

    @staticmethod
    def ensure_facenn_home():
        if not os.path.exists(Config.FACENN_HOME):
            os.makedirs(Config.FACENN_HOME)
            
    @staticmethod
    def get_weights_path(model_name: str, file_name: str) -> str:
        Config.ensure_facenn_home()
        path = os.path.join(Config.FACENN_HOME, model_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return os.path.join(path, file_name)

DEVICE = Config.get_device()
logger.info(f"Running on device: {DEVICE}")
