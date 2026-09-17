# API Reference

The primary interface is the `Facenn` class located in `facenn`.

```python
from facenn import Facenn, Config, FaceDB
```

## `Facenn` Class

### `__init__`
Initializes the Facenn pipeline.

**Arguments:**
- `recognition_model_name` (str): Name of the face embedding model. Default: `'ArcFace'`. Options: `'ArcFace'`, `'FaceNet'`, `'MobileFaceNet'`, `'EdgeFace'`, `'Buffalo_L'`.
- `detector_backend` (str): Detector backend. Default: `'yunet'`. Options: `'yunet'`, `'retinaface'`, `'opencv'` (`'haarcascade'`).
- `analyzer_backend` (str): Backend for attribute analysis. Default: `'onnx'`. Options: `'onnx'`, `'torch'`.
- `recognition_backend` (str): Backend for recognition models. Default: `'torch'`. Options: `'torch'`, `'onnx'`.
- `db_path` (str, optional): Custom path for the SQLite database. Default: `~/.cache/facenn/identities.sqlite`.

---

### `detect_faces`
Detects bounding boxes and facial landmarks.

**Arguments:**
- `img` (str | np.ndarray | PIL.Image | bytes): Input image.

**Returns:**
- `List[dict]`: Dictionaries with `box` `[x, y, w, h]`, `confidence` (float), `landmarks` `(5, 2) np.ndarray`, and `keypoints`.

---

### `extract_faces`
Crops and aligns detected faces into standard chips.

**Arguments:**
- `img`: Input image.
- `target_size` (tuple, optional): Target chip size `(width, height)`. Default: model input resolution.
- `align` (bool): Whether to perform 5-point affine alignment. Default: `True`.

**Returns:**
- `List[np.ndarray]`: List of cropped/aligned BGR face chips.

---

### `represent`
Extracts L2-normalized face embedding vectors.

**Arguments:**
- `img`: Input image.
- `align` (bool): Whether to align faces before extraction. Default: `True`.

**Returns:**
- `List[np.ndarray]`: List of 1D float32 normalized embedding vectors.

---

### `verify`
Compares two images to determine if they represent the same identity.

**Arguments:**
- `img1`: Path or image of the first face.
- `img2`: Path or image of the second face.
- `threshold` (float): Cosine distance threshold (default: `0.4`).
- `enforce_detection` (bool): Whether to require face detection (default: `True`).
- `align` (bool): Whether to align faces prior to comparison (default: `True`).

**Returns:**
- `dict`: `{"verified": bool, "distance": float, "similarity": float, "threshold": float, "model": str}`.

---

### `find`
Searches the database for matching identities.

**Arguments:**
- `img`: Query image containing a face.
- `db_path` (str, optional): Alternative database path.
- `k` (int): Number of top matches to retrieve (default: `5`).
- `threshold` (float): Maximum cosine distance (default: `0.4`).

**Returns:**
- `pandas.DataFrame`: Matches with columns `['identity', 'distance', 'id']`.

---

### `add_to_db`
Registers a face into the database.

**Arguments:**
- `img`: Image containing the face.
- `identity` (str): Identity name or identifier.

**Returns:**
- `bool`: `True` if successfully registered.

---

### `analyze`
Extracts demographic and emotional attributes.

**Arguments:**
- `img`: Input image.
- `actions` (tuple): Attributes to extract. Options: `'age'`, `'gender'`, `'race'`, `'emotion'`.

**Returns:**
- `List[dict]`: Analysis results per face with region and requested attributes.

---

## `FaceDB` Class

SQLite-backed vector database with in-memory normalized matrix indexing.

```python
db = FaceDB(db_path="faces.sqlite")
db.add_face(embedding, identity="Alice")
results = db.search(query_embedding, k=5, threshold=0.4)
db.delete(identity="Alice")
db.clear()
```

