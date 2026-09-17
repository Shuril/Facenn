from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import cv2
import numpy as np
import pandas as pd
import torch

from facenn.config import DEVICE
from facenn.database import FaceDB
from facenn.detectors.base import FaceDetector
from facenn.detectors.retinaface import OpenCVFaceDetector, RetinaFaceWrapper
from facenn.detectors.yunet import YuNetWrapper
from facenn.models.arcface import ArcFace
from facenn.models.buffalo_l import Buffalo_L
from facenn.models.edgeface import EdgeFace
from facenn.models.facenet import FaceNet
from facenn.models.mobilefacenet import MobileFaceNet, MobileFaceNetV2
from facenn.models.sface import SFace
from facenn.models.vggface import VGGFace
from facenn.utils.distance import find_cosine_distance
from facenn.utils.image import crop_and_align_face, load_image

logger = logging.getLogger("facenn")


class Facenn:
    """
    Main face analysis and recognition interface.
    """

    RECOGNITION_MODELS = {
        "ArcFace": ArcFace,
        "FaceNet": FaceNet,
        "MobileFaceNet": MobileFaceNet,
        "MobileFaceNet v2": MobileFaceNetV2,
        "SFace": SFace,
        "EdgeFace": EdgeFace,
        "Buffalo_L": Buffalo_L,
        "VGG-Face": VGGFace,
    }

    DETECTORS = {
        "yunet": YuNetWrapper,
        "retinaface": RetinaFaceWrapper,
        "opencv": OpenCVFaceDetector,
        "haarcascade": OpenCVFaceDetector,
    }

    def __init__(
        self,
        recognition_model_name: str = "ArcFace",
        detector_backend: str = "yunet",
        analyzer_backend: str = "onnx",
        recognition_backend: str = "torch",
        db_path: Optional[str] = None,
    ):
        self.recognition_model_name = recognition_model_name
        self.detector_backend = detector_backend.lower()
        self.analyzer_backend = analyzer_backend.lower()
        self.recognition_backend = recognition_backend.lower()
        self.db_path = db_path

        self._recognition_model = None
        self._detector = None
        self._db: Optional[FaceDB] = None
        self._fairface_analyzer = None
        self._emotion_analyzer = None

    @property
    def recognition_model(self):
        if self._recognition_model is None:
            if self.recognition_model_name not in self.RECOGNITION_MODELS:
                available = list(self.RECOGNITION_MODELS.keys())
                raise ValueError(
                    f"Unknown model '{self.recognition_model_name}'. Available: {available}"
                )
            model_cls = self.RECOGNITION_MODELS[self.recognition_model_name]
            self._recognition_model = model_cls()
            self._recognition_model.backend = self.recognition_backend
        return self._recognition_model

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            if self.detector_backend not in self.DETECTORS:
                available = list(self.DETECTORS.keys())
                raise ValueError(
                    f"Unknown detector backend '{self.detector_backend}'. Available: {available}"
                )
            detector_cls = self.DETECTORS[self.detector_backend]
            self._detector = detector_cls()
        return self._detector

    @detector.setter
    def detector(self, value: FaceDetector):
        self._detector = value

    @property
    def db(self) -> FaceDB:
        if self._db is None:
            self._db = FaceDB(self.db_path)
        return self._db

    def detect_faces(self, img: Union[str, np.ndarray, Any]) -> List[Dict[str, Any]]:
        """Detects faces and facial landmarks in the given image."""
        img_bgr = load_image(img)
        return self.detector.detect_faces(img_bgr)

    def extract_faces(
        self,
        img: Union[str, np.ndarray, Any],
        target_size: Optional[Tuple[int, int]] = None,
        align: bool = True,
    ) -> List[np.ndarray]:
        """
        Extracts cropped and optionally aligned face chips from an image.
        """
        img_bgr = load_image(img)
        detections = self.detector.detect_faces(img_bgr)
        size = target_size or self.recognition_model.input_shape

        faces = []
        for det in detections:
            landmarks = det.get("landmarks") if align else None
            chip = crop_and_align_face(img_bgr, box=det["box"], landmarks=landmarks, target_size=size)
            faces.append(chip)
        return faces

    def represent(
        self,
        img: Union[str, np.ndarray, Any],
        align: bool = True,
    ) -> List[np.ndarray]:
        """
        Extracts L2-normalized embedding vectors for all detected faces.
        Returns a list of 1D numpy arrays.
        """
        img_bgr = load_image(img)
        detections = self.detector.detect_faces(img_bgr)
        if not detections:
            return []

        target_size = self.recognition_model.input_shape
        embeddings: List[np.ndarray] = []

        for det in detections:
            landmarks = det.get("landmarks") if align else None
            face_chip = crop_and_align_face(
                img_bgr,
                box=det["box"],
                landmarks=landmarks,
                target_size=target_size,
            )

            # Preprocessing: BGR [0, 255] -> normalized float tensor [-1, 1]
            tensor = face_chip.transpose(2, 0, 1).astype(np.float32)
            tensor = (tensor - 127.5) / 128.0
            tensor_torch = torch.from_numpy(tensor).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = self.recognition_model.predict(tensor_torch)

            if isinstance(emb, torch.Tensor):
                emb_np = emb.squeeze(0).detach().cpu().numpy()
            else:
                emb_np = np.asarray(emb).ravel()

            embeddings.append(emb_np.astype(np.float32))

        return embeddings

    def verify(
        self,
        img1: Union[str, np.ndarray, Any],
        img2: Union[str, np.ndarray, Any],
        threshold: float = 0.4,
        enforce_detection: bool = True,
        align: bool = True,
    ) -> Dict[str, Any]:
        """
        Verifies whether two face images belong to the same person.
        """
        emb1 = self.represent(img1, align=align)
        emb2 = self.represent(img2, align=align)

        if not emb1 or not emb2:
            if enforce_detection:
                return {
                    "verified": False,
                    "distance": 1.0,
                    "threshold": threshold,
                    "model": self.recognition_model_name,
                    "reason": "Face not detected in one or both images",
                }
            # Fallback: treat entire image as face crop
            emb1 = [self._represent_raw(img1)] if not emb1 else emb1
            emb2 = [self._represent_raw(img2)] if not emb2 else emb2

        dist = float(find_cosine_distance(emb1[0], emb2[0]))
        return {
            "verified": dist <= threshold,
            "distance": round(dist, 4),
            "similarity": round(1.0 - dist, 4),
            "threshold": threshold,
            "model": self.recognition_model_name,
        }

    def _represent_raw(self, img: Union[str, np.ndarray, Any]) -> np.ndarray:
        img_bgr = load_image(img)
        target_size = self.recognition_model.input_shape
        resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        tensor = (resized.transpose(2, 0, 1).astype(np.float32) - 127.5) / 128.0
        tensor_torch = torch.from_numpy(tensor).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.recognition_model.predict(tensor_torch)
        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def find(
        self,
        img: Union[str, np.ndarray, Any],
        db_path: Optional[str] = None,
        k: int = 5,
        threshold: float = 0.4,
    ) -> pd.DataFrame:
        """
        Searches the database for identities matching the face in the image.
        """
        embs = self.represent(img)
        if not embs:
            return pd.DataFrame(columns=["identity", "distance", "id"])

        database = self.db if db_path is None else FaceDB(db_path)
        return database.search(embs[0], k=k, threshold=threshold)

    def add_to_db(self, img: Union[str, np.ndarray, Any], identity: str) -> bool:
        """
        Registers a face identity into the vector database.
        """
        try:
            embs = self.represent(img)
            if embs:
                self.db.add_face(embs[0], identity)
                return True
        except Exception as exc:
            logger.error("Failed to add face for identity '%s': %s", identity, exc)
        return False

    def analyze(
        self,
        img: Union[str, np.ndarray, Any],
        actions: Tuple[str, ...] = ("age", "gender", "race", "emotion"),
    ) -> List[Dict[str, Any]]:
        """
        Analyzes faces for demographic and emotional attributes.
        """
        img_bgr = load_image(img)
        detections = self.detector.detect_faces(img_bgr)
        if not detections:
            return []

        results = []
        for det in detections:
            x, y, w, h = det["box"]
            h_img, w_img = img_bgr.shape[:2]
            x1 = max(0, min(x, w_img - 1))
            y1 = max(0, min(y, h_img - 1))
            x2 = max(x1 + 1, min(x + w, w_img))
            y2 = max(y1 + 1, min(y + h, h_img))
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            analysis: Dict[str, Any] = {
                "region": [x1, y1, x2 - x1, y2 - y1],
                "confidence": det.get("confidence", 1.0),
            }

            # FairFace attributes
            if any(a in actions for a in ("age", "gender", "race")):
                fairface = self._get_fairface_analyzer()
                resized = cv2.resize(crop, (224, 224)).astype(np.float32)
                norm = (resized.transpose(2, 0, 1) - 127.5) / 128.0
                norm_tensor = torch.from_numpy(norm).unsqueeze(0).to(DEVICE)
                ff_res = fairface.analyze(norm_tensor)
                if "age" in actions:
                    analysis["age"] = ff_res["age"]
                if "gender" in actions:
                    analysis["gender"] = ff_res["gender"]
                if "race" in actions:
                    analysis["race"] = ff_res["race"]

            # Emotion attribute
            if "emotion" in actions:
                emotion_model = self._get_emotion_analyzer()
                resized = cv2.resize(crop, (224, 224)).astype(np.float32)
                norm = (resized.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
                norm_tensor = torch.from_numpy(norm).unsqueeze(0).to(DEVICE)
                analysis["emotion"] = emotion_model.analyze(norm_tensor)

            results.append(analysis)

        return results

    def _get_fairface_analyzer(self):
        if self._fairface_analyzer is None:
            if self.analyzer_backend == "onnx":
                from facenn.models.analyzers.fairface_onnx import FairFaceONNX

                self._fairface_analyzer = FairFaceONNX()
            else:
                from facenn.models.analyzers.fairface import FairFaceModel

                self._fairface_analyzer = FairFaceModel()
        return self._fairface_analyzer

    def _get_emotion_analyzer(self):
        if self._emotion_analyzer is None:
            if self.analyzer_backend == "onnx":
                from facenn.models.analyzers.emotion_onnx import HSEmotionONNX

                self._emotion_analyzer = HSEmotionONNX()
            else:
                from facenn.models.analyzers.emotion import HSEmotionModel

                self._emotion_analyzer = HSEmotionModel()
        return self._emotion_analyzer


