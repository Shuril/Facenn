# Model Zoo

Facenn provides modular backbones for detection, recognition, and demographic/emotional analysis.

## Face Recognition Models

Recognition models extract a normalized 512-dimensional embedding vector from an aligned face.

| Model Name | Embedding Dimension | Input Resolution | Architecture | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **ArcFace** | 512 | 112x112 | IR-SE-50 | Default. High accuracy across diverse poses and lighting. |
| **EdgeFace** | 512 | 112x112 | MobileViT | Optimized for edge and mobile inference. |
| **MobileFaceNet** | 512 | 112x112 | MobileNetV2 | Lightweight, low compute requirements. |
| **FaceNet** | 512 | 160x160 | Inception-ResNet | Standard classic benchmark architecture. |
| **Buffalo_L** | 512 | 112x112 | IR-SE-50 | High-capacity model pack compatible weights. |

---

## Face Attribute Analyzers

Analyzers estimate demographics and emotions from cropped face images (224x224).

### FairFace (ResNet-34)
- **Tasks**: Age (9 intervals), Gender (Male / Female), Race (7 categories).
- **Backend**: Native ONNX Runtime with CoreML / CUDA / CPU providers.

### HSEmotion (EfficientNet-B0)
- **Tasks**: 8 primary emotions (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise).
- **Backend**: Native ONNX Runtime and PyTorch options.

---

## Face Detectors

| Detector | Landmarks | Backbone | Use Case |
| :--- | :--- | :--- | :--- |
| **YuNet** | 5 Points | CNN | Default. Built into OpenCV DNN, ultra-fast on CPU with landmark output. |
| **RetinaFace** | 5 Points | ResNet-50 | High accuracy on challenging small or occluded faces. |
| **Haar Cascade** | None | Cascade | Classic legacy fallback for low-resource environments. |

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

