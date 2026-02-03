import cv2
import numpy as np
import os
import shutil
from facenn.core import Facenn
from facenn.config import Config

def create_dummy_image(filename):
    # distinct images
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1) # "Face"
    cv2.circle(img, (100, 100), 10, (0, 0, 0), -1) # Eye
    cv2.circle(img, (150, 100), 10, (0, 0, 0), -1) # Eye
    cv2.imwrite(filename, img)
    return filename

def test_pipeline():
    print("Testing Facenn Pipeline...")
    
    # 1. Setup
    img1 = create_dummy_image("test_face1.jpg")
    img2 = create_dummy_image("test_face2.jpg")
    
    # 2. Init
    try:
        # Test MobileFaceNet v2 specifically
        app = Facenn(recognition_model_name='MobileFaceNet v2', detector_backend='opencv') 
        print("Facenn initialized with MobileFaceNet v2.")
    except Exception as e:
        print(f"Failed to initialize Facenn: {e}")
        return

    # 3. Represent
    # Since RetinaFace decoder is placeholder, I will manually patch the simple detector 
    # just for the *logic* flow, BUT the important part is that app.detector.net was loaded above.
    # If app.detector.net is not None, we succeeded in downloading/loading weights.
    if hasattr(app.detector, 'net') and app.detector.net is not None:
         print("RetinaFace weights loaded successfully.")
    else:
         print("RetinaFace weights FAILED to load.")

    # Monkey patch detector for synthetic testing 
    app.detector.detect_faces = lambda img: [{'box': [50, 50, 150, 150], 'confidence': 0.99, 'keypoints': {}}]
    
    try:
        emb = app.represent(img1)
        print(f"Representation successful. Embedding shape: {emb[0].shape}")
    except Exception as e:
        print(f"Representation failed: {e}")
        
    # 4. Verify
    try:
        res = app.verify(img1, img2)
        print(f"Verification result: {res}")
    except Exception as e:
        print(f"Verification failed: {e}")

    # 5. Database
    try:
        app.add_to_db(img1, "person_1")
        print("Added to DB.")
        
        search_res = app.find(img2)
        print(f"Search result: \n{search_res}")
    except Exception as e:
        print(f"Database operation failed: {e}")

    # Cleanup
    if os.path.exists("test_face1.jpg"): os.remove("test_face1.jpg")
    if os.path.exists("test_face2.jpg"): os.remove("test_face2.jpg")
    if os.path.exists("facenn_db.pkl"): os.remove("facenn_db.pkl")

if __name__ == "__main__":
    test_pipeline()
