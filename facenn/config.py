import os
import torch
import logging

logger = logging.getLogger("facenn")
logger.addHandler(logging.NullHandler())


class Config:
    FACENN_HOME = os.getenv("FACENN_HOME", os.path.expanduser("~/.cache/facenn"))
    FACENN_RELEASE_URL = os.getenv(
        "FACENN_RELEASE_URL",
        "https://github.com/Shuril/Facenn/releases/download/v0.2.0/",
    )
    USE_OPENVINO = os.getenv("FACENN_USE_OPENVINO", "0") == "1"

    @staticmethod
    def get_release_weights_url(filename: str) -> str:
        base = Config.FACENN_RELEASE_URL.rstrip("/")
        return f"{base}/{filename}"

    @staticmethod
    def get_device(preference=None):
        """
        Auto-detects the best available device or returns the specified preference.
        """
        if preference:
            if isinstance(preference, torch.device):
                return preference
            return torch.device(preference)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
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
    def ensure_facenn_home() -> str:
        os.makedirs(Config.FACENN_HOME, exist_ok=True)
        return Config.FACENN_HOME

    @staticmethod
    def get_weights_path(model_name: str, file_name: str) -> str:
        Config.ensure_facenn_home()
        path = os.path.join(Config.FACENN_HOME, model_name)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, file_name)


DEVICE = Config.get_device()

