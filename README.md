# Facenn

Facenn is a lightweight, modular Python library for face detection, 5-point alignment, face verification, and facial attribute analysis. It supports both PyTorch and ONNX Runtime backends across CPU, Apple Silicon (MPS / CoreML), and NVIDIA CUDA.

## Features

- **Face Detection with Landmark Alignment**: Default YuNet detector extracts 5 facial landmarks and performs affine similarity transformation (112x112 canonical template) to maximize recognition accuracy.
- **Recognition Backends**: PyTorch and ONNX Runtime support for models including ArcFace (IR-SE50), EdgeFace, and MobileFaceNet.
- **Embedded Vector Database**: Built-in SQLite vector store with normalized NumPy matrix indexing for sub-millisecond similarity search.
- **Facial Analysis**: Demographics (age, gender, race via FairFace) and emotion classification (HSEmotion).
- **Flexible Hardware Support**: Auto-detects Apple Silicon (MPS/CoreML), NVIDIA CUDA, or CPU.

## Installation

### From Source
```bash
git clone https://github.com/Shuril/Facenn.git
cd Facenn
pip install -e .
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Face Verification
Compare two images to check whether they belong to the same identity:

```python
from facenn import Facenn

app = Facenn(recognition_model_name="ArcFace", detector_backend="yunet")
result = app.verify("person1.jpg", "person2.jpg", threshold=0.4)

print(result)
# {'verified': True, 'distance': 0.1842, 'similarity': 0.8158, 'threshold': 0.4, 'model': 'ArcFace'}
```

### 2. Facial Attribute Analysis
Estimate age, gender, race, and emotion:

```python
from facenn import Facenn

app = Facenn(analyzer_backend="onnx")
analysis = app.analyze("photo.jpg", actions=("age", "gender", "emotion"))

for face in analysis:
    print(face["region"], face.get("age"), face.get("gender"), face.get("emotion"))
```

### 3. Face Embeddings and Vector Search
Extract normalized embeddings and search within the built-in database:

```python
from facenn import Facenn

app = Facenn()

# Register identity
app.add_to_db("alice.jpg", identity="Alice")
app.add_to_db("bob.jpg", identity="Bob")

# Search nearest matches
matches = app.find("query.jpg", k=3, threshold=0.4)
print(matches)
```

## Model Reference

| Category | Model | Backbone | Input Resolution | Backend |
| :--- | :--- | :--- | :--- | :--- |
| **Detection** | YuNet (Default) | CNN | Dynamic | OpenCV DNN / ONNX |
| **Detection** | RetinaFace | ResNet-50 | Dynamic | PyTorch |
| **Detection** | Haar Cascade | Cascade | Dynamic | OpenCV |
| **Recognition** | ArcFace (Default) | IR-SE-50 | 112x112 | PyTorch / ONNX |
| **Recognition** | EdgeFace | MobileViT | 112x112 | PyTorch |
| **Recognition** | MobileFaceNet | MobileNetV2 | 112x112 | PyTorch / ONNX |
| **Analysis** | FairFace | ResNet-34 | 224x224 | ONNX / PyTorch |
| **Analysis** | HSEmotion | EfficientNet-B0 | 224x224 | ONNX / PyTorch |

## Configuration

Weights are stored by default under `~/.cache/facenn`. You can override the cache location with an environment variable:

```bash
export FACENN_HOME=/path/to/custom/weights
```

To run the unit test suite:
```bash
pytest tests/ -v
```

## License
MIT License

