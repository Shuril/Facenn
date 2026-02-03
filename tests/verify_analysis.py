from facenn.core import Facenn
import torch
import numpy as np
import cv2
import os

def test_analysis():
    print("Testing Facenn Analysis Pipeline...")
    
    # 1. Create a dummy image with a face-like region
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), -1)
    img_path = "test_analysis.jpg"
    cv2.imwrite(img_path, img)

    # 2. Init
    try:
        app = Facenn(detector_backend='opencv')
        # Monkey patch detector to return our dummy face
        app.detector.detect_faces = lambda img: [{'box': [50, 50, 200, 200]}]
        print("Facenn initialized.")
    except Exception as e:
        print(f"Failed to initialize Facenn: {e}")
        return

    # 3. Analyze
    try:
        results = app.analyze(img_path, actions=['age', 'gender', 'race', 'emotion'])
        print(f"Analysis results: {results}")
        
        if len(results) > 0:
            res = results[0]
            for field in ['age', 'gender', 'race', 'emotion']:
                if field in res:
                    print(f"  {field.capitalize()}: {res[field]}")
                else:
                    print(f"  {field.capitalize()}: FAILED")
        else:
            print("No faces detected/analyzed.")
            
    except Exception as e:
        print(f"Analysis failed: {e}")
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    test_analysis()
