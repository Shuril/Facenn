from abc import ABC, abstractmethod
import numpy as np
from facenn.config import DEVICE

class FaceDetector(ABC):
    def __init__(self, detector_name: str):
        self.detector_name = detector_name
        self.model = None

    @abstractmethod
    def detect_faces(self, img: np.ndarray):
        """
        Detect faces in an image.
        Returns: list of dicts with keys: 'box', 'confidence', 'keypoints'
        box: [x, y, w, h]
        """
        pass
