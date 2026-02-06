import cv2
import numpy as np
import base64
import requests
from pathlib import Path
from PIL import Image

def load_image(img):
    """
    Loads an image from various sources: path, numpy array, base64, url.
    Returns: BGR numpy array (OpenCV format).
    """
    exact_image = False
    
    if isinstance(img, np.ndarray):
        return img
    
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))[:, :, ::-1] # RGB to BGR

    if isinstance(img, str):
        # Base64
        if img.startswith("data:image/"):
            return load_base64_img(img)
        
        # Path
        if Path(img).is_file():
            image = cv2.imread(img)
            if image is None:
                 raise ValueError(f"Could not read image from path: {img}")
            return image
            
        # URL
        if img.lower().startswith("http"):
             response = requests.get(img)
             image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
             return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    raise ValueError(f"Invalid image input type: {type(img)}")

def load_base64_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def align_face(img, keypoints, target_size=(112, 112)):
    """
    Aligns the face based on 5 keypoints (left eye, right eye, nose, left mouth, right mouth).
    Uses Similarity Transformation.
    """
    if not keypoints or len(keypoints) < 2:
        return cv2.resize(img, target_size)

    # Standard reference points for 112x112 image
    # Note: These are for 112x112. We scale them if target_size is different.
    ref_pts = np.array([
        [38.2946, 51.6963], # left eye
        [73.5318, 51.5014], # right eye
        [56.0252, 71.7366], # nose
        [41.5493, 92.3655], # left mouth
        [70.7299, 92.2041]  # right mouth
    ], dtype=np.float32)

    if target_size != (112, 112):
        ref_pts[:, 0] = ref_pts[:, 0] * target_size[0] / 112
        ref_pts[:, 1] = ref_pts[:, 1] * target_size[1] / 112

    # Map our keypoints to the ref points
    # Reference points order: left_eye, right_eye, nose, left_mouth, right_mouth
    ref_map = {
        'left_eye': [38.2946, 51.6963],
        'right_eye': [73.5318, 51.5014],
        'nose': [56.0252, 71.7366],
        'mouth_left': [41.5493, 92.3655],
        'mouth_right': [70.7299, 92.2041]
    }
    
    src_pts = []
    dst_pts = []
    for name, ref_pt in ref_map.items():
        if name in keypoints:
            src_pts.append(keypoints[name])
            # Scale reference point if needed
            scaled_ref = [
                ref_pt[0] * target_size[0] / 112,
                ref_pt[1] * target_size[1] / 112
            ]
            dst_pts.append(scaled_ref)

    if len(src_pts) < 3:
        # Fallback for 2 points (eyes)
        if 'left_eye' in keypoints and 'right_eye' in keypoints:
             # Basic eye-based alignment could be added here,
             # but estimateAffinePartial2D needs at least 2 points.
             # Actually opencv says 2 points are enough for estimateAffinePartial2D
             pass
        else:
             return cv2.resize(img, target_size)

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    
    # If we have at least 2 points, we can do similarity transform
    if len(src_pts) >= 2:
        tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        if tform is not None:
            aligned_img = cv2.warpAffine(img, tform, target_size, borderValue=0)
            return aligned_img

    # Fallback to simple crop if transform fails or not enough points
    return cv2.resize(img, target_size)

# Avoid circular import by defining simple distance for alignment here or importing carefully
def find_euclidean_distance_cpu(a, b):
    return np.linalg.norm(a - b)
