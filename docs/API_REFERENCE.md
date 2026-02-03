# API Reference 📖

The primary interface for the library is the `Facenn` class located in `facenn.core`.

## Facenn Class

### `__init__`
Initializes the Facenn pipeline.

**Arguments:**
- `recognition_model_name` (str): Name of the model to use for face embeddings.
    - Default: `'ArcFace'`.
    - Options: `'ArcFace'`, `'FaceNet'`, `'MobileFaceNet'`, `'EdgeFace'`, `'VGG-Face'`, `'Buffalo_L'`.
- `detector_backend` (str): Backend for face detection.
    - Default: `'opencv'`.
    - Options: `'opencv'`, `'retinaface'`, `'yunet'`, `'centerface'`, `'yolov12'`.
- `analyzer_backend` (str): Backend for attribute analysis (Age, Gender, Race, Emotion).
    - Default: `'onnx'`.
    - Options: `'torch'`, `'onnx'`.
- `recognition_backend` (str): Backend for recognition models.
    - Default: `'torch'`.
    - Options: `'torch'`, `'onnx'`.

---

### `verify`
Compares two images to determine if they belong to the same person.

**Arguments:**
- `img1_path` (str|np.array): Path to the first image or a numpy array.
- `img2_path` (str|np.array): Path to the second image or a numpy array.
- `threshold` (float): Cosine distance threshold (lower is stricter). Default: `0.4`.

**Returns:**
- `dict`: Contains `verified` (bool), `distance` (float), `threshold` (float), and `model` name.

---

### `represent`
Extracts face embeddings from an image.

**Arguments:**
- `img_path` (str|np.array): Path to the image or a numpy array.

**Returns:**
- `list`: A list of 512-dim (or model-specific size) embeddings (tensors/arrays).

---

### `find`
Searches for a face in the internal Vector Database.

**Arguments:**
- `img_path` (str|np.array): Path to the query image.
- `threshold` (float): Maximum cosine distance to consider a match. Default: `0.4`.

**Returns:**
- `list`: A list of matches, each containing `identity` and `distance`.

---

### `add_to_db`
Registers a face into the Vector Database.

**Arguments:**
- `img_path` (str|np.array): Image containing the face.
- `identity` (str): Name/ID for this person.

**Returns:**
- `bool`: True if successful.

---

### `analyze`
Extracts demographic and emotional attributes from faces.

**Arguments:**
- `img_path` (str|np.array): Path to the image.
- `actions` (list): List of attributes to extract.
    - Options: `['age', 'gender', 'race', 'emotion']`.

**Returns:**
- `list`: List of analysis results per face, including the bounding box (`region`).

---

## Configuration (Config Class)

The `Config` class in `facenn.config` allows global settings:

- `FACENN_HOME`: Root directory for model weights. Controlled by `FACENN_HOME` environment variable.
- `USE_OPENVINO`: Boolean flag to enable OpenVINO optimization. Managed via `FACENN_USE_OPENVINO` env var.
- `DEVICE`: The auto-detected hardware device (`cuda`, `mps`, `vulkan`, `cpu`).
