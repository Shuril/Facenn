from typing import Any, Dict, List, Optional
import logging
import os
import cv2
import numpy as np

from facenn.config import Config
from facenn.detectors.base import FaceDetector
from facenn.utils.io import download_file_from_url

logger = logging.getLogger("facenn")

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URLS = [
    Config.get_release_weights_url(YUNET_FILENAME),
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]


class YuNetWrapper(FaceDetector):
    def __init__(
        self,
        conf_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        model_path: Optional[str] = None,
    ):
        super().__init__("YuNet")
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.model_path = model_path
        self.net = None

    def _ensure_model(self):
        if self.net is not None:
            return

        weights_path = self.model_path or Config.get_weights_path("YuNet", YUNET_FILENAME)
        if not os.path.exists(weights_path):
            download_file_from_url(YUNET_MODEL_URLS, weights_path)

        self.net = cv2.FaceDetectorYN.create(
            model=weights_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    def detect_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        self._ensure_model()
        if self.net is None or img is None or img.size == 0:
            return []

        h, w = img.shape[:2]
        self.net.setInputSize((int(w), int(h)))

        _, detections = self.net.detect(img)
        if detections is None or len(detections) == 0:
            return []

        results: List[Dict[str, Any]] = []
        for face in detections:
            box = [int(v) for v in face[0:4]]
            conf = float(face[-1])

            # OpenCV YuNet landmarks layout:
            # right eye (viewer left), left eye (viewer right), nose, right mouth, left mouth
            landmarks = np.array(
                [
                    [face[4], face[5]],
                    [face[6], face[7]],
                    [face[8], face[9]],
                    [face[10], face[11]],
                    [face[12], face[13]],
                ],
                dtype=np.float32,
            )

            results.append(
                {
                    "box": box,
                    "confidence": conf,
                    "landmarks": landmarks,
                    "keypoints": {
                        "left_eye": (float(face[4]), float(face[5])),
                        "right_eye": (float(face[6]), float(face[7])),
                        "nose": (float(face[8]), float(face[9])),
                        "mouth_left": (float(face[10]), float(face[11])),
                        "mouth_right": (float(face[12]), float(face[13])),
                    },
                }
            )

        return results

