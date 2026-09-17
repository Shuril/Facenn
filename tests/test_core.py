import numpy as np
import pytest
import torch
from facenn.core import Facenn
from facenn.detectors.base import FaceDetector


class MockDetector(FaceDetector):
    def __init__(self, return_faces=True):
        super().__init__("MockDetector")
        self.return_faces = return_faces

    def detect_faces(self, img):
        if not self.return_faces:
            return []
        h, w = img.shape[:2]
        return [
            {
                "box": [10, 10, min(50, w - 20), min(50, h - 20)],
                "confidence": 0.99,
                "landmarks": np.array([
                    [20.0, 20.0],
                    [40.0, 20.0],
                    [30.0, 30.0],
                    [22.0, 40.0],
                    [38.0, 40.0],
                ], dtype=np.float32),
                "keypoints": {},
            }
        ]


def test_facenn_initialization():
    app = Facenn(detector_backend="opencv", db_path=":memory:")
    assert app.detector_backend == "opencv"
    assert app.recognition_model_name == "ArcFace"


def test_facenn_verify_no_faces():
    app = Facenn(detector_backend="opencv", db_path=":memory:")
    app.detector = MockDetector(return_faces=False)

    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)

    res = app.verify(img1, img2, enforce_detection=True)
    assert not res["verified"]
    assert "reason" in res


class MockRecognitionModel:
    def __init__(self):
        self.input_shape = (112, 112)
        self.backend = "torch"

    def predict(self, tensor):
        # Deterministic 512-dim normalized embedding
        emb = torch.zeros((1, 512), dtype=torch.float32)
        emb[0, 0] = 1.0
        return emb


def test_facenn_db_operations():
    app = Facenn(detector_backend="opencv", db_path=":memory:")
    app.detector = MockDetector(return_faces=True)
    app._recognition_model = MockRecognitionModel()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    embs = app.represent(img)
    assert len(embs) == 1
    assert embs[0].ndim == 1
    assert pytest.approx(np.linalg.norm(embs[0]), abs=1e-3) == 1.0

    # Add to DB
    ok = app.add_to_db(img, "PersonA")
    assert ok is True

    # Find in DB
    matches = app.find(img, threshold=0.1)
    assert len(matches) == 1
    assert matches.iloc[0]["identity"] == "PersonA"

