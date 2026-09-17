from typing import Any, Dict, List, Optional
from itertools import product
import math
import os
import cv2
import numpy as np
import torch
import torchvision.ops

from facenn.config import Config, DEVICE
from facenn.detectors.base import FaceDetector
from facenn.models.retinaface import RetinaFace
from facenn.utils.io import download_file_from_url

RETINAFACE_URL = "https://huggingface.co/akhaliq/RetinaFace-R50/resolve/main/RetinaFace-R50.pth"


def _generate_priors(image_size: tuple, steps=(8, 16, 32), min_sizes=((16, 32), (64, 128), (256, 512))) -> torch.Tensor:
    h, w = image_size
    feature_maps = [[math.ceil(h / step), math.ceil(w / step)] for step in steps]
    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / w
                s_ky = min_size / h
                cx = (j + 0.5) * steps[k] / w
                cy = (i + 0.5) * steps[k] / h
                anchors.extend([cx, cy, s_kx, s_ky])
    return torch.tensor(anchors, dtype=torch.float32).view(-1, 4)


def _decode_boxes(loc: torch.Tensor, priors: torch.Tensor, variances=(0.1, 0.2)) -> torch.Tensor:
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        dim=-1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _decode_landmarks(pre: torch.Tensor, priors: torch.Tensor, variances=(0.1, 0.2)) -> torch.Tensor:
    priors_xy = priors[:, :2]
    priors_wh = priors[:, 2:]
    parts = []
    for idx in range(5):
        start = idx * 2
        parts.append(priors_xy + pre[:, start : start + 2] * variances[0] * priors_wh)
    return torch.cat(parts, dim=-1)


class RetinaFaceWrapper(FaceDetector):
    def __init__(self, conf_threshold: float = 0.8, nms_threshold: float = 0.4):
        super().__init__("RetinaFace")
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net: Optional[RetinaFace] = None

    def _ensure_model(self):
        if self.net is not None:
            return

        self.net = RetinaFace(phase="test").to(DEVICE)
        weights_path = Config.get_weights_path("RetinaFace", "Resnet50_Final.pth")

        if not os.path.exists(weights_path):
            download_file_from_url(RETINAFACE_URL, weights_path)

        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Strip potential 'module.' prefix from DataParallel training
        cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.net.load_state_dict(cleaned_state, strict=False)
        self.net.eval()

    def detect_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        self._ensure_model()
        if self.net is None or img is None or img.size == 0:
            return []

        h_orig, w_orig = img.shape[:2]
        img_float = np.float32(img)
        # Standard RetinaFace normalization (BGR minus pixel means)
        img_float -= (104, 117, 123)
        tensor_img = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            loc, conf, landmarks_raw = self.net(tensor_img)

        scale = torch.tensor([w_orig, h_orig, w_orig, h_orig], device=DEVICE)
        priors = _generate_priors((h_orig, w_orig)).to(DEVICE)

        boxes = _decode_boxes(loc.squeeze(0), priors) * scale
        scores = conf.squeeze(0)[:, 1]

        scale_landmarks = torch.tensor([w_orig, h_orig] * 5, device=DEVICE)
        landmarks = _decode_landmarks(landmarks_raw.squeeze(0), priors) * scale_landmarks

        # Filter by confidence
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]

        if boxes.shape[0] == 0:
            return []

        keep = torchvision.ops.nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        landmarks = landmarks[keep].cpu().numpy()

        results: List[Dict[str, Any]] = []
        for box, score, ldm in zip(boxes, scores, landmarks):
            x1, y1, x2, y2 = [int(v) for v in box]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            pts = ldm.reshape(5, 2).astype(np.float32)

            results.append(
                {
                    "box": [x1, y1, w, h],
                    "confidence": float(score),
                    "landmarks": pts,
                    "keypoints": {
                        "left_eye": (float(pts[0, 0]), float(pts[0, 1])),
                        "right_eye": (float(pts[1, 0]), float(pts[1, 1])),
                        "nose": (float(pts[2, 0]), float(pts[2, 1])),
                        "mouth_left": (float(pts[3, 0]), float(pts[3, 1])),
                        "mouth_right": (float(pts[4, 0]), float(pts[4, 1])),
                    },
                }
            )

        return results


class OpenCVFaceDetector(FaceDetector):
    """Haar Cascade face detector as a lightweight CPU fallback."""

    def __init__(self):
        super().__init__("OpenCV")
        xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(xml_path)

    def detect_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        if img is None or img.size == 0:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        results: List[Dict[str, Any]] = []
        for x, y, w, h in faces:
            results.append(
                {
                    "box": [int(x), int(y), int(w), int(h)],
                    "confidence": 1.0,
                    "landmarks": None,
                    "keypoints": None,
                }
            )
        return results

