import cv2
import numpy as np
import os
from facenn.detectors.base import FaceDetector
from facenn.config import Config, logger
from facenn.utils.io import download_file_from_url

class YuNetWrapper(FaceDetector):
    def __init__(self):
        super().__init__("YuNet")
        self.net = None
        self.conf_threshold = 0.9
        self.nms_threshold = 0.3
        self.top_k = 5000
        self.load_model()

    def load_model(self):
        # YuNet weights file
        # Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        
        weights_path = Config.get_weights_path("YuNet", "face_detection_yunet_2023mar.onnx")
        
        try:
             download_file_from_url(url, weights_path)
             if os.path.exists(weights_path):
                 self.net = cv2.FaceDetectorYN.create(
                     model=weights_path,
                     config="",
                     input_size=(320, 320), # Will be updated dynamically
                     score_threshold=self.conf_threshold,
                     nms_threshold=self.nms_threshold,
                     top_k=self.top_k,
                     backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                     target_id=cv2.dnn.DNN_TARGET_CPU # OpenCV DNN usually runs best on CPU/OpenCL
                 )
             else:
                 logger.warning("Failed to download YuNet weights.")
        except Exception as e:
             logger.error(f"Error loading YuNet weights: {e}")

    def detect_faces(self, img: np.ndarray):
        if self.net is None: 
            return []
            
        h, w, _ = img.shape
        self.net.setInputSize((w, h))
        
        faces = self.net.detect(img)
        # faces[1] is the result, [0] is a boolean? Check cv2 docs.
        # detect returns (faces, landmarks) tuple in some versions or just faces
        
        # In recent OpenCV:
        # faces = self.net.detect(img)
        # faces is a tuple: (faces, valid_flag) or something similar?
        results = faces[1] if faces[1] is not None else []
        
        output = []
        if results is not None:
            for face in results:
                # Format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rm, y_rm, x_lm, y_lm, conf]
                box = face[0:4].astype(int)
                conf = face[-1]
                
                # Landmarks (5 points)
                landmarks = face[4:14].reshape(5, 2).astype(int)

                output.append({
                    'box': box.tolist(),
                    'confidence': float(conf),
                    'keypoints': {
                        'right_eye': landmarks[0].tolist(),
                        'left_eye': landmarks[1].tolist(),
                        'nose': landmarks[2].tolist(),
                        'mouth_right': landmarks[3].tolist(),
                        'mouth_left': landmarks[4].tolist()
                    }
                })
        return output
