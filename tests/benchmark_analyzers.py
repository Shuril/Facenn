import time
import torch
import numpy as np
import cv2
from facenn.core import Facenn
import os

def benchmark_analyzers():
    # Setup
    img_path = "tests/test_image.jpg"
    if not os.path.exists(img_path):
        # Create a dummy image if not exists
        dummy = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.putText(dummy, "Test", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(img_path, dummy)

    backends = ['torch', 'onnx']
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    
    actions = ['age', 'gender', 'race', 'emotion']
    
    print(f"{'Backend':<10} | {'Device':<6} | {'Avg Time (ms)':<15}")
    print("-" * 40)

    for backend in backends:
        for device in devices:
            try:
                # Initialize Facenn with chosen backend and device
                app = Facenn(analyzer_backend=backend)
                # Monkey patch DEVICE for this test to force cpu/mps for torch
                import facenn.config
                orig_device = facenn.config.DEVICE
                facenn.config.DEVICE = torch.device(device)
                
                # Warmup
                app.analyze(img_path, actions=actions)
                
                # Run benchmark
                start_time = time.time()
                iterations = 5
                for _ in range(iterations):
                    app.analyze(img_path, actions=actions)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / iterations * 1000
                print(f"{backend:<10} | {device:<6} | {avg_time:>13.2f} ms")
                
                # Restore device
                facenn.config.DEVICE = orig_device
            except Exception as e:
                print(f"{backend:<10} | {device:<6} | Error: {e}")

if __name__ == "__main__":
    benchmark_analyzers()
