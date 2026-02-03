# Models Zoo 🦁

Facenn-next supports various models for recognition and analysis. Choosing the right model depends on your accuracy requirements and hardware constraints.

## Face Recognition Models

Recognition models convert a face image into a numerical embedding (vector).

| Model Name | Dimension | Input Shape | Complexity | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **ArcFace** | 512 | 112x112 | Moderate | High-accuracy general purpose. |
| **EdgeFace (S)** | 512 | 112x112 | Low | Mobile and Edge devices. |
| **MobileFaceNet**| 512 | 112x112 | Low | High speed on older hardware. |
| **FaceNet** | 512 | 160x160 | High | Legacy systems, compatibility. |
| **VGG-Face** | 2622 | 224x224 | Very High | Maximum feature detail (slow). |
| **Buffalo_L** | 512 | 112x112 | Moderate | Standard server-side high accuracy. |

---

## Face Attribute Analyzers

Analyzers estimate demographics and emotions.

### FairFace (ResNet34)
- **Primary Task**: Age, Gender, Race estimation.
- **Input**: 224x224 (RGB).
- **Backend Optimization**: Native ONNX (CoreML/MPS) support for ~10ms inference.

### HSEmotion (EfficientNet-B0)
- **Primary Task**: Emotion recognition (8 categories).
- **Architecture**: EfficientNet is optimized for mobile/edge.
- **Accuracy**: State-of-the-art for lightweight emotion models.

---

## Face Detectors

Detectors locate faces and provide landmarks.

- **OpenCV**: Very fast, reliable on CPU, but less accurate for small or occluded faces.
- **YuNet**: High-performance CPU detector (extremely small).
- **RetinaFace**: High-accuracy detector with landmarks.
- **CenterFace**: Anchor-free detector, good balance.
- **YOLOv12**: Experimental high-speed object-detection based face locator.

---

## Custom Model Training
Facenn utilizes standard PyTorch weights (`.pt`, `.pth`). To integrate your own trained model, you can inherit from `FaceRecognitionModel` in `facenn.models.base`.
