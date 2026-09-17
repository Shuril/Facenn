# Model Zoo

Facenn provides modular backbones for detection, recognition, and demographic/emotional analysis.
All models are automatically downloaded on first use from official GitHub Releases (`https://github.com/Shuril/Facenn/releases`) with automatic fallback to upstream mirrors.

## Face Recognition Models

Recognition models extract an L2-normalized embedding vector from an aligned face crop.

| Model Name | Embedding Dimension | Input Resolution | Runtime / Backend | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **ArcFace** | 512 | 112x112 | PyTorch | Default. High accuracy across diverse poses and lighting (IR-SE-50). |
| **SFace** | 128 | 112x112 | OpenCV DNN | Ultra-lightweight (~38 MB), zero extra DL framework overhead. |
| **MobileFaceNet** | 512 | 112x112 | ONNX Runtime | Lightweight (~13 MB), trained on WebFace600K. |
| **EdgeFace** | 512 | 112x112 | PyTorch Hub | MobileViT architecture optimized for edge devices. |
| **FaceNet** | 512 | 160x160 | PyTorch | Inception-ResNet-V1 trained on VGGFace2. |
| **Buffalo_L** | 512 | 112x112 | PyTorch | InsightFace IR-50 compatible weights. |
| **VGG-Face** | 2622 / 512 | 224x224 | PyTorch | Classic deep benchmark architecture. |

---

## Face Attribute Analyzers

Analyzers estimate demographics and emotions from cropped face images (224x224).

### FairFace (ResNet-34)
- **Tasks**: Age (9 intervals), Gender (Male / Female), Race (7 categories).
- **Backend**: Native ONNX Runtime with CoreML / CUDA / CPU providers.
- **Model File**: `fairface.onnx` (~81 MB)

### HSEmotion (EfficientNet-B0)
- **Tasks**: 8 primary emotions (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise).
- **Backend**: Native ONNX Runtime with CoreML / CUDA / CPU providers.
- **Model File**: `enet_b0_8_best_vgaf.onnx` (~15 MB)

---

## Face Detectors

| Detector | Landmarks | Backend | Use Case |
| :--- | :--- | :--- | :--- |
| **YuNet** | 5 Points | OpenCV DNN | Default. Built into OpenCV, ultra-fast on CPU (< 5 ms) with 5-point alignment landmarks. |
| **RetinaFace** | 5 Points | PyTorch | High accuracy on challenging, small, or occluded faces. |
| **Haar Cascade** | None | OpenCV Cascade | Built-in fallback without external downloads. |

---

## Adding Custom Recognition Models

Inherit from `FaceRecognitionModel` in `facenn.models.base` and implement `load_model()` and `forward()`:

```python
from facenn.models.base import FaceRecognitionModel

class MyModel(FaceRecognitionModel):
    def __init__(self):
        super().__init__(model_name="MyModel", input_shape=(112, 112))

    def load_model(self):
        # Load your PyTorch nn.Module into self.model
        pass

    def forward(self, x):
        return self.model(x)
```

