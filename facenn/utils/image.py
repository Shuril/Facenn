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

def align_face(img, left_eye, right_eye):
    """
    Aligns the face based on eye coordinates.
    """
    # Simple alignment logic implementation placeholder
    # In a full implementation, this uses affine transformations
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 # inverse clocwise
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 # clockwise
        
    a = find_euclidean_distance_cpu(np.array(left_eye), np.array(point_3rd))
    b = find_euclidean_distance_cpu(np.array(right_eye), np.array(point_3rd))
    c = find_euclidean_distance_cpu(np.array(right_eye), np.array(left_eye))
    
    if b != 0 and c != 0:
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a)
        angle = (angle * 180) / np.pi
        
        if direction == -1:
            angle = 90 - angle
        
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
        
    return img

# Avoid circular import by defining simple distance for alignment here or importing carefully
def find_euclidean_distance_cpu(a, b):
    return np.linalg.norm(a - b)
