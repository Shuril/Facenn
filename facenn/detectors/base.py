from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class FaceDetector(ABC):
    def __init__(self, detector_name: str):
        self.detector_name = detector_name

    @abstractmethod
    def detect_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in a BGR image.

        Returns:
            List of dictionaries containing:
                - 'box': [x, y, w, h] (integers)
                - 'confidence': float (detection confidence)
                - 'landmarks': Optional np.ndarray of shape (5, 2)
                - 'keypoints': Optional dict with eye, nose, and mouth coordinates
        """
        pass

