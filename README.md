# Facenn-next 🚀

**Facenn-next** is a state-of-the-art, high-performance face recognition and analysis library for Python. Designed for speed and simplicity, it leverages **PyTorch** and **ONNX Runtime** to provide blazing fast inference on CPU, Apple Silicon (MPS/CoreML), and NVIDIA GPUs.

![Facenn Banner](https://img.shields.io/badge/Facenn-Next-brightgreen) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![License](https://img.shields.io/badge/License-MIT-orange)

## ✨ key Features

*   **⚡ Blazing Fast**: Optimized for local inference. **6x faster** analysis on Mac/CPU using ONNX.
*   **🧠 Logic & Auto-Convert**: Automatically converts PyTorch recognition models to ONNX for accelerated inference.
*   **🛠️ Hardware Aware**: Native support for **MPS** (Mac), **CUDA**, and **CoreML**.
*   **📚 Rich Model Zoo**: Includes ArcFace, FaceNet, VGG-Face, EdgeFace, MobileFaceNet, and more.
*   **📊 Deep Analysis**: Estimate Age, Gender, Race (FairFace) and Emotion (HSEmotion).
*   **💾 Vector Verification**: Built-in simple Vector DB for identity management.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

*(Optional) For automatic ONNX export of complex models:*
```bash
pip install onnxscript
```

---

## 🧠 Model Zoo

Facenn supports a wide range of models. You can easily switch between them.

| Model | Type | Architecture | Input Size | ONNX Support | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ArcFace** | Recognition | ResNet50 (IR-SE) | 112x112 | ✅ (Auto) | **Default**. Best balance of accuracy and speed. |
| **EdgeFace** | Recognition | MobileViT | 112x112 | ✅ (Native) | **SOTA for Edge**. Extremely lightweight and fast. |
| **MobileFaceNet** | Recognition | MobileNetV2 | 112x112 | ✅ (Auto) | Classic lightweight model for mobile. |
| **FaceNet** | Recognition | InceptionResNet | 160x160 | ✅ (Auto) | Google's classic model. |
| **VGG-Face** | Recognition | VGG16 | 224x224 | ✅ (Auto) | High accuracy, legacy architecture (Heavy). |
| **FairFace** | Analyzer | ResNet34 | 224x224 | ✅ (Native) | Age, Gender, and Race estimation. |
| **HSEmotion** | Analyzer | EfficientNet-B0 | 224x224 | ✅ (Native) | High-accuracy emotion recognition. |

---

## 🚀 Usage Guide

### 1. Face Verification
Compare two faces to check if they match.

```python
from facenn.core import Facenn

# Initialize with desired model (default: ArcFace)
app = Facenn(recognition_model_name='EdgeFace')

# Verify
result = app.verify("img1.jpg", "img2.jpg")
print(result)
# {'verified': True, 'distance': 0.24, 'threshold': 0.4, ...}
```

### 2. High-Performance Attribute Analysis
Analyze Age, Gender, Race, and Emotion. Use `analyzer_backend='onnx'` (Default) for maximum speed.

```python
# ONNX backend is roughly 6x faster on CPU/Mac than pure Torch
app = Facenn(analyzer_backend='onnx')

analysis = app.analyze("face.jpg", actions=['age', 'gender', 'emotion'])
print(analysis)
# [{'age': '20-29', 'gender': 'Female', 'emotion': 'Happy', ...}]
```

### 3. Accelerated Recognition (ONNX Backend)
You can run recognition models via ONNX Runtime for potential speedups on CPU. Facenn will **automatically export** your PyTorch model to ONNX on the first run.

```python
# Use 'onnx' backend for recognition
app = Facenn(recognition_model_name='MobileFaceNet', recognition_backend='onnx')

# First run will export model (~few seconds)
embeddings = app.represent("dataset/user.jpg")
```

### 4. Vector Database
Simple identity search.

```python
# Add identity
app.add_to_db("dataset/alice.jpg", "Alice")

# Search
matches = app.find("query.jpg", threshold=0.4)
```

### ⚙️ Configuration & Customization

Facenn automatically detects and uses the best available hardware. You can also manually configure backends and accelerators during initialization:

```python
app = Facenn(
    recognition_model_name='ArcFace',
    detector_backend='opencv',     # 'opencv', 'retinaface', 'yunet', 'yolov12'
    analyzer_backend='onnx',      # 'torch', 'onnx'
    recognition_backend='torch'    # 'torch', 'onnx', 'vulkan'
)
```

#### 🚀 Advanced Acceleration

*   **OpenVINO (Intel CPU/GPU)**: Enable Intel-specific optimization by setting `export FACENN_USE_OPENVINO=1`.
*   **Vulkan**: Supported via `recognition_backend='vulkan'` on compatible systems.
*   **XPU**: Native support for Intel Arc/Data Center GPUs.
*   **CoreML**: Used automatically by the `onnx` backend on macOS.

---

## 📂 Project Structure

All weights are stored in your project's `weights/` directory, keeping your system clean.
```
.
├── facenn/             # Core library
├── weights/            # Downloaded models (GitIgnored)
└── requirements.txt    # Dependencies
```

## 📄 License
MIT License
