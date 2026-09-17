from typing import Optional, Tuple, Union
from pathlib import Path
import base64
import cv2
import numpy as np
import requests
from PIL import Image

# Canonical 5-point landmark template for 112x112 face (InsightFace / ArcFace standard)
# Points: [image_left_eye, image_right_eye, nose, image_left_mouth, image_right_mouth]
ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def clip_box(box: Union[list, tuple, np.ndarray], img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Clips a [x, y, w, h] bounding box to image boundaries."""
    h_img, w_img = img_shape[:2]
    x, y, w, h = [int(v) for v in box]

    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(x1 + 1, min(x + w, w_img))
    y2 = max(y1 + 1, min(y + h, h_img))

    return x1, y1, x2 - x1, y2 - y1


def align_face_5point(
    img: np.ndarray,
    landmarks: np.ndarray,
    image_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Performs similarity transformation on a face using 5 facial landmarks.
    landmarks: np.ndarray of shape (5, 2)
    image_size: (width, height)
    """
    src_pts = np.asarray(landmarks, dtype=np.float32)
    if src_pts.shape != (5, 2):
        raise ValueError(f"Expected landmarks of shape (5, 2), got {src_pts.shape}")

    # Scale canonical template if target size is different from 112x112
    dst_pts = ARCFACE_DST_112.copy()
    if image_size != (112, 112):
        dst_pts[:, 0] *= image_size[0] / 112.0
        dst_pts[:, 1] *= image_size[1] / 112.0

    tfm, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if tfm is None:
        raise ValueError("Could not compute affine transformation from provided landmarks")

    return cv2.warpAffine(
        img,
        tfm,
        (image_size[0], image_size[1]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def crop_and_align_face(
    img: np.ndarray,
    box: Optional[Union[list, tuple, np.ndarray]] = None,
    landmarks: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Crops and aligns face. Uses 5-point alignment when landmarks are available,
    otherwise falls back to clipped bounding box crop and resize.
    """
    if landmarks is not None:
        landmarks_arr = np.asarray(landmarks, dtype=np.float32)
        if landmarks_arr.shape == (5, 2):
            try:
                return align_face_5point(img, landmarks_arr, target_size)
            except Exception:
                pass

    if box is not None:
        x, y, w, h = clip_box(box, img.shape)
        face_crop = img[y : y + h, x : x + w]
        if face_crop.size > 0:
            return cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LINEAR)

    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def load_image(img: Union[str, np.ndarray, Image.Image, Path, bytes]) -> np.ndarray:
    """
    Loads an image from various sources (path, numpy array, PIL Image, base64, URL).
    Returns BGR numpy array (standard OpenCV format).
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    if isinstance(img, Image.Image):
        rgb = np.array(img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(img, (str, Path)):
        img_str = str(img)

        if img_str.startswith("data:image/"):
            return _load_base64_image(img_str)

        path = Path(img_str)
        if path.is_file():
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Failed to decode image from path: {path}")
            return image

        if img_str.lower().startswith(("http://", "https://")):
            response = requests.get(img_str, timeout=15)
            response.raise_for_status()
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"Failed to decode image from URL: {img_str}")
            return decoded

    if isinstance(img, bytes):
        image_array = np.frombuffer(img, dtype=np.uint8)
        decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode image from bytes")
        return decoded

    raise TypeError(f"Unsupported image input type: {type(img)}")


def _load_base64_image(uri: str) -> np.ndarray:
    try:
        encoded_data = uri.split(",")[1] if "," in uri else uri
        decoded_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode base64 image data")
        return decoded
    except Exception as exc:
        raise ValueError(f"Invalid base64 image input: {exc}") from exc


def align_face(img: np.ndarray, left_eye: Tuple[float, float], right_eye: Tuple[float, float]) -> np.ndarray:
    """Rotates image based on two eye points (legacy helper)."""
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

