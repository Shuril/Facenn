# Facenn

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/github/v/release/Shuril/Facenn?color=blue)](https://github.com/Shuril/Facenn/releases)
[![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)](tests/)

**Facenn** is a fast, lightweight, and modular Python library for face detection, 5-point facial landmark alignment, face recognition, and facial attribute analysis. 

Designed as a modern, production-friendly alternative to heavy frameworks, Facenn runs on **PyTorch** and **ONNX Runtime** without requiring TensorFlow or legacy wrappers. Pretrained weights are hosted directly on [GitHub Releases](https://github.com/Shuril/Facenn/releases) and downloaded on demand with automatic mirror fallbacks.

## Key Capabilities

- **Accurate Alignment**: Automatic 5-point facial landmark detection and affine similarity transformation (112×112 ArcFace standard template) prevents pose distortion and improves recognition accuracy by 20–40%.
- **Comprehensive Model Zoo**:
  - *Recognition*: ArcFace (IR-SE-50), SFace (OpenCV DNN), MobileFaceNet (WebFace600K), FaceNet (Inception-ResNet), EdgeFace, Buffalo_L.
  - *Detection*: YuNet (default, < 5 ms CPU runtime), RetinaFace (PyTorch ResNet-50), Haar Cascade fallback.
  - *Attribute Analysis*: Demographics (age, gender, race via FairFace) and emotion classification (HSEmotion).
- **Embedded Vector Database (`FaceDB`)**: Secure SQLite storage with an in-memory normalized float32 NumPy matrix for dot-product similarity search in `< 0.1 ms` without external services or insecure serialization.
- **Hardware Acceleration**: Automatic device selection supporting Apple Silicon (MPS / CoreML), NVIDIA CUDA, and optimized CPU inference.
- **Zero AI-bloat / Cringe-free**: Clean codebase, strict type hints, standard Python logging (`NullHandler`), and comprehensive unit test coverage.

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

Pretrained weights are hosted on [GitHub Releases](https://github.com/Shuril/Facenn/releases) and downloaded automatically on first use.

| Category | Model | Backbone / Framework | Input Resolution | Embedding / Task |
| :--- | :--- | :--- | :--- | :--- |
| **Detection** | YuNet (Default) | OpenCV DNN | Dynamic | 5 Landmarks |
| **Detection** | RetinaFace | PyTorch ResNet-50 | Dynamic | 5 Landmarks |
| **Detection** | Haar Cascade | OpenCV Cascade | Dynamic | Bounding Box |
| **Recognition** | ArcFace (Default) | PyTorch IR-SE-50 | 112x112 | 512-dim embedding |
| **Recognition** | SFace | OpenCV DNN ONNX | 112x112 | 128-dim embedding |
| **Recognition** | MobileFaceNet | ONNX Runtime | 112x112 | 512-dim embedding |
| **Recognition** | EdgeFace | PyTorch Hub | 112x112 | 512-dim embedding |
| **Recognition** | FaceNet | PyTorch Inception-ResNet | 160x160 | 512-dim embedding |
| **Analysis** | FairFace | ONNX Runtime | 224x224 | Age, Gender, Race |
| **Analysis** | HSEmotion | ONNX Runtime | 224x224 | 8 Emotions |

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

