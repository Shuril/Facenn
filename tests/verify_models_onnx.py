import torch
from facenn.core import Facenn
import os

def verify_models():
    print("=== Verifying Recognition Models & ONNX Export ===")
    
    models_to_test = [
        # Light ones first
        'EdgeFace', 
        'MobileFaceNet',
        'ArcFace'
        # 'VGGFace' # Might take long to download, maybe skip for quick test or include if bandwidth OK
    ]
    
    img_path = "tests/test.jpg"
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        try:
            # 1. Initialize Model Manualy to force ops
            print(f"  [Init] Loading {model_name}...")
            # We import dynamically to test factory-like behavior or just use Facenn core mapping
            # Let's use Facenn to get the model instance
            app = Facenn(recognition_model_name=model_name, recognition_backend='torch')
            model = app.recognition_model
            model.load_model()
            
            # Dummy input
            dummy_input = torch.randn(1, 3, model.input_shape[0], model.input_shape[1])
            
            # 2. PyTorch Inference
            res_torch = model.predict(dummy_input)
            print(f"  [PyTorch] Output shape: {res_torch.shape}")
            
            # 3. ONNX Export & Inference
            print(f"  [ONNX] Exporting/Loading...")
            model.backend = 'onnx'
            res_onnx = model.predict(dummy_input)
            print(f"  [ONNX] Output shape: {res_onnx.shape}")
            
            diff = torch.abs(res_torch.cpu() - res_onnx.cpu()).mean()
            print(f"  [Diff] Mean Difference: {diff:.6f}")
            
            print(f"  SUCCESS: {model_name}")
            
        except Exception as e:
            print(f"  FAILURE: {model_name} - {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify_models()
