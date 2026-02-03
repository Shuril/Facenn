# Acceleration Guide 🚀

Facenn-next is designed to squeeze every bit of performance out of your hardware.

## Support Matrix

| Hardware | Backend | Recommended Backend Setting | Notes |
| :--- | :--- | :--- | :--- |
| **NVIDIA GPU** | CUDA | `torch` or `onnx` | Fastest for all tasks. |
| **Apple Silicon (M1/M2/M3)** | MPS / CoreML | `onnx` | Use ONNX for 6x faster attribute analysis. |
| **Intel CPU/GPU** | OpenVINO | `torch` (with ENV) | Best for Intel Core and Arc GPUs. |
| **Generic CPU** | ONNX Runtime | `onnx` | Significantly faster than baseline PyTorch. |
| **AMD/Others** | Vulkan / DirectML | `vulkan` | Experimental cross-platform support. |

---

## 🏎️ Leveraging ONNX

For attribute analysis (Age, Gender, etc.), **ONNX is enabled by default**. It uses the `CoreMLExecutionProvider` on Mac and `CUDAExecutionProvider` on Linux.

### Recognition Auto-Export
Recognition models (ArcFace, FaceNet) are typically PyTorch models. Facenn can automatically convert them to ONNX:
```python
app = Facenn(recognition_backend='onnx')
```
*Tip: Requires `pip install onnxscript` for first-time conversion.*

---

## ❄️ Intel OpenVINO Optimization

If you are running on Intel hardware, you can trigger deep optimization for the PyTorch models (Recognition) using OpenVINO's Torch compiler:

1.  **Install dependencies**:
    ```bash
    pip install openvino openvino-telemetry
    ```
2.  **Enable flag**:
    ```bash
    export FACENN_USE_OPENVINO=1
    ```

---

## 🛠️ Environment Variables

- `FACENN_HOME`: Changes where weights are stored (Default: `weights/` in project root).
- `FACENN_USE_OPENVINO`: Set to `1` to enable OpenVINO compilation.

---

## 🧪 Device Troubleshooting

To check which device Facenn is currently using:
```python
from facenn.config import DEVICE
print(f"Device: {DEVICE}")
```
If you expect `cuda` but see `cpu`, ensure your `torch` version is CUDA-enabled (`torch.cuda.is_available()`).
