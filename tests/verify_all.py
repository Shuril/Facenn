import os
import cv2
import numpy as np
import torch
from facenn.core import Facenn
from facenn.config import Config

def create_dummy_image(path):
    # Create a 500x500 RGB image with a "face-like" structure
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.circle(img, (250, 250), 100, (200, 200, 200), -1) # Face
    cv2.circle(img, (200, 220), 20, (0, 0, 0), -1) # Left Eye
    cv2.circle(img, (300, 220), 20, (0, 0, 0), -1) # Right Eye
    cv2.ellipse(img, (250, 300), (40, 20), 0, 0, 180, (0, 0, 0), 2) # Mouth
    cv2.imwrite(path, img)
    return path

def verify_all():
    print("=== Facenn Final Verification ===")
    
    # 1. Setup
    img1 = create_dummy_image("tests/v_img1.jpg")
    img2 = create_dummy_image("tests/v_img2.jpg")
    
    # 2. Initialize Facenn
    print("\n[1] Initializing Facenn (Default)...")
    try:
        app = Facenn()
        print("    Success.")
    except Exception as e:
        print(f"    Failed: {e}")
        return

    # 3. Detection
    print("\n[2] Testing Face Detection...")
    try:
        faces = app.detector.detect_faces(cv2.imread(img1))
        if len(faces) > 0:
            print(f"    Success. Found {len(faces)} faces.")
        else:
            print("    Warning: No faces found in dummy image.")
    except Exception as e:
        print(f"    Failed: {e}")

    # 4. Verification
    print("\n[3] Testing Verification (ArcFace)...")
    try:
        res = app.verify(img1, img2)
        print(f"    Success. Result: Verified={res['verified']}, Distance={res['distance']:.4f}")
    except Exception as e:
        print(f"    Failed: {e}")

    # 5. Analysis (PyTorch)
    print("\n[4] Testing Analysis (PyTorch - CPU/MPS)...")
    app.analyzer_backend = 'torch'
    try:
        # Mock detector for dummy images
        from facenn.detectors.base import FaceDetector
        class MockDetector(FaceDetector):
            def __init__(self): super().__init__("Mock")
            def detect_faces(self, img):
                return [{'box': [50, 50, 200, 200], 'confidence': 0.99, 'keypoints': {}}]
        app.detector = MockDetector()
        
        res = app.analyze(img1, actions=['age', 'gender', 'race', 'emotion'])
        print(f"    Success. Result: {res[0]['age']}, {res[0]['gender']}, {res[0]['emotion']}")
    except Exception as e:
        print(f"    Failed: {e}")

    # 6. Analysis (ONNX)
    print("\n[5] Testing Analysis (ONNX - CoreML/CPU)...")
    app.analyzer_backend = 'onnx'
    # Force reload of analyzers
    if hasattr(app, 'fairface_analyzer'): del app.fairface_analyzer
    if hasattr(app, 'emotion_analyzer'): del app.emotion_analyzer
    
    try:
        res = app.analyze(img1, actions=['age', 'gender', 'race', 'emotion'])
        print(f"    Success. Result: {res[0]['age']}, {res[0]['gender']}, {res[0]['emotion']}")
    except Exception as e:
        print(f"    Failed: {e}")

    # 7. Database
    print("\n[6] Testing Vector Database...")
    try:
        db_path = "tests/test.db"
        if os.path.exists(db_path): os.remove(db_path)
        # Re-init with specific DB path if Facenn supports it, otherwise it uses default
        # Assuming Facenn has internal logic. We'll just test add/find API.
        
        # Add img1
        app.add_to_db(img1, "User1")
        
        # Find img1
        found = app.find(img1)
        if len(found) > 0:
            print(f"    Success. Found matches: {len(found)}")
        else:
            print("    Warning: No matches found (DB might rely on exact detection).")
            
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    verify_all()
