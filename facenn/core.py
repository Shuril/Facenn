import cv2
import numpy as np
import torch
from facenn.config import DEVICE, logger
from facenn.database import FaceDB
from facenn.detectors.retinaface import RetinaFaceWrapper, OpenCVFaceDetector
from facenn.detectors.yunet import YuNetWrapper
from facenn.detectors.centerface import CenterFace
from facenn.detectors.yolo import YOLOvFace
from facenn.models.arcface import ArcFace
from facenn.models.facenet import FaceNet
from facenn.models.vggface import VGGFace
from facenn.models.mobilefacenet import MobileFaceNetV2
from facenn.models.edgeface import EdgeFace
from facenn.models.buffalo_l import Buffalo_L
from facenn.utils.image import load_image, align_face
from facenn.utils.distance import find_cosine_distance

class Facenn:
    def __init__(self, recognition_model_name='ArcFace', detector_backend='opencv', analyzer_backend='onnx', recognition_backend='torch'):
        self.recognition_model_name = recognition_model_name
        self.detector_backend = detector_backend
        self.analyzer_backend = analyzer_backend
        self.recognition_backend = recognition_backend
        
        # Load Recognition Model
        if recognition_model_name == 'ArcFace':
            self.recognition_model = ArcFace()
        elif recognition_model_name == 'FaceNet':
            self.recognition_model = FaceNet()
        elif recognition_model_name == 'VGG-Face':
            self.recognition_model = VGGFace()
        elif recognition_model_name == 'MobileFaceNet' or recognition_model_name == 'MobileFaceNet v2':
            self.recognition_model = MobileFaceNetV2()
        elif recognition_model_name == 'EdgeFace':
            self.recognition_model = EdgeFace()
        elif recognition_model_name == 'Buffalo_L':
            self.recognition_model = Buffalo_L()
        else:
            raise NotImplementedError(f"Model {recognition_model_name} not implemented yet.")
            
        # Set backend
        self.recognition_model.backend = recognition_backend
        
        # Load Detector
        if detector_backend == 'retinaface':
            self.detector = RetinaFaceWrapper()
        elif detector_backend == 'opencv':
            self.detector = OpenCVFaceDetector()
        elif detector_backend == 'yunet':
             self.detector = YuNetWrapper()
        elif detector_backend == 'centerface':
             self.detector = CenterFace()
        elif detector_backend == 'yolov12':
             self.detector = YOLOvFace()
        else:
            raise ValueError(f"Detector {detector_backend} not supported.")
            
        self.db = FaceDB()

    def verify(self, img1_path, img2_path, threshold=0.4):
        """
        Verifies if two images represent the same person.
        """
        emb1 = self.represent(img1_path)
        emb2 = self.represent(img2_path)
        
        if len(emb1) == 0 or len(emb2) == 0:
             return {"verified": False, "reason": "Face not detected in one of the images"}
             
        # Taking the first face found in each image for simplicity in this basic API
        dist = find_cosine_distance(emb1[0], emb2[0])
        dist = float(dist.item()) if isinstance(dist, torch.Tensor) else dist
        
        return {
            "verified": dist < threshold,
            "distance": dist,
            "threshold": threshold,
            "model": self.recognition_model_name
        }

    def find(self, img_path, db_path=None, threshold=0.4):
        """
        Finds a face in the database.
        """
        # If db_path is provided, we might load a specific DB, but here we use the internal one
        # or we could scan a directory of images to build it 'on the fly' like DeepFace does.
        # For this high-perf library, we assume usage of the built-in Vector DB mostly.
        
        emb = self.represent(img_path)
        if not emb:
            return []
            
        return self.db.search(emb[0], threshold=threshold)

    def represent(self, img_path):
        """
        Returns the embedding vector for faces in the image.
        """
        img = load_image(img_path)
        
        # Detect
        faces = self.detector.detect_faces(img)
        if not faces:
            return []

        target_size = self.recognition_model.input_shape
        face_tensors = []
        
        for face in faces:
            # Align face if keypoints are available
            if face.get('keypoints'):
                face_img = align_face(img, face['keypoints'], target_size=target_size)
            else:
                x, y, w, h = face['box']
                # Clip coordinates to image boundaries
                ih, iw = img.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(iw, x+w), min(ih, y+h)
                face_img = img[y1:y2, x1:x2]
                if face_img.size == 0: continue
                face_img = cv2.resize(face_img, target_size)
            
            # Preprocess
            face_img = np.transpose(face_img, (2, 0, 1)) # HWC to CHW
            face_tensor = torch.tensor(face_img).float()
            face_tensor = (face_tensor - 127.5) / 128.0 # Normalize -1 to 1
            face_tensors.append(face_tensor)
            
        if not face_tensors:
            return []
            
        # Batch processing for efficiency
        batch_tensor = torch.stack(face_tensors).to(DEVICE)

        with torch.no_grad():
            embeddings = self.recognition_model.predict(batch_tensor)
            
        # Return as list of individual embeddings
        if len(embeddings.shape) == 1: # Single face case if model returns (D,)
            return [embeddings]

        return [embeddings[i] for i in range(embeddings.size(0))]
        
    def add_to_db(self, img_path, identity):
        """
        Adds a face from an image to the database.
        """
        try:
             embs = self.represent(img_path)
             if embs:
                 self.db.add_face(embs[0], identity)
                 self.db.save()
                 return True
        except Exception as e:
             logger.error(f"Error adding {identity}: {e}")
        return False

    def analyze(self, img_path, actions=['age', 'gender', 'race', 'emotion']):
        """
        Analyzes a face for demographic attributes and emotions.
        """
        img = load_image(img_path)
        faces = self.detector.detect_faces(img)
        
        results = []
        for face in faces:
            x, y, w, h = face['box']
            face_img_raw = img[y:y+h, x:x+w]
            if face_img_raw.size == 0: continue
            
            analysis = {"region": face['box']}
            
            # FairFace actions
            if any(a in actions for a in ['age', 'gender', 'race']):
                if not hasattr(self, 'fairface_analyzer'):
                    if self.analyzer_backend == 'onnx':
                        from facenn.models.analyzers.fairface_onnx import FairFaceONNX
                        self.fairface_analyzer = FairFaceONNX()
                    else:
                        from facenn.models.analyzers.fairface import FairFaceModel
                        self.fairface_analyzer = FairFaceModel()
                
                # Preprocess for FairFace (224x224)
                face_img = cv2.resize(face_img_raw, (224, 224))
                face_img = torch.tensor(face_img).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
                face_img = (face_img - 127.5) / 128.0
                
                res = self.fairface_analyzer.analyze(face_img)
                if 'age' in actions: analysis['age'] = res['age']
                if 'gender' in actions: analysis['gender'] = res['gender']
                if 'race' in actions: analysis['race'] = res['race']
                
            # Emotion action
            if 'emotion' in actions:
                if not hasattr(self, 'emotion_analyzer'):
                    if self.analyzer_backend == 'onnx':
                        from facenn.models.analyzers.emotion_onnx import HSEmotionONNX
                        self.emotion_analyzer = HSEmotionONNX()
                    else:
                        from facenn.models.analyzers.emotion import HSEmotionModel
                        self.emotion_analyzer = HSEmotionModel()
                    
                # Preprocess for HSEmotion (224x224)
                face_img = cv2.resize(face_img_raw, (224, 224))
                face_img = torch.tensor(face_img).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
                face_img = (face_img / 255.0 - 0.5) / 0.5 # Normalization for HSEmotion
                
                analysis['emotion'] = self.emotion_analyzer.analyze(face_img)
                
            results.append(analysis)
            
        return results

