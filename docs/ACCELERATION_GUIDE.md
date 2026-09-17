# Hardware Acceleration Guide

Facenn provides multiple execution backends to optimize inference speed across different platforms.

## Support Matrix

| Platform | Hardware Target | Supported Backend | Notes |
| :--- | :--- | :--- | :--- |
| **NVIDIA GPU** | CUDA | `torch` or `onnx` | CUDAExecutionProvider in ONNX Runtime. |
| **Apple Silicon (M-series)** | Metal Performance Shaders (MPS) / Apple Neural Engine (ANE) | `torch` (`mps`) or `onnx` (`CoreMLExecutionProvider`) | CoreML provider delivers fast CPU/ANE attribute analysis. |
| **Intel CPU / Arc GPU** | OpenVINO | `torch` (via `FACENN_USE_OPENVINO=1`) | Compiles PyTorch models for Intel architectures. |
| **Universal CPU** | x86_64 / ARM64 | `onnx` or `torch` | CPUExecutionProvider with multi-threading. |

---

## ONNX Runtime

ONNX Runtime is used for demographic and emotion analysis by default and can be optionally enabled for face recognition:

```python
from facenn import Facenn

# Use ONNX Runtime for recognition models
app = Facenn(recognition_model_name="ArcFace", recognition_backend="onnx")
```

On macOS, Facenn automatically registers `CoreMLExecutionProvider` if available. On systems with NVIDIA drivers and `onnxruntime-gpu`, `CUDAExecutionProvider` is preferred.

---

## Intel OpenVINO

For Intel processors, OpenVINO compilation can be enabled via an environment variable:

```bash
export FACENN_USE_OPENVINO=1
```

Requirements:
```bash
pip install openvino
```

---

## Environment Variables

- `FACENN_HOME`: Sets the root directory for downloaded model weights (Default: `~/.cache/facenn`).
- `FACENN_USE_OPENVINO`: Set to `1` to enable OpenVINO PyTorch compilation on Intel hardware.

---

## Checking Active Device

To verify the hardware device selected by Facenn:

```python
from facenn.config import DEVICE

print(f"Active compute device: {DEVICE}")
```

