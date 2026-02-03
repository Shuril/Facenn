import time
import torch
import numpy as np
from facenn.models.analyzers.fairface import FairFaceModel
from facenn.models.analyzers.fairface_onnx import FairFaceONNX
from facenn.models.analyzers.emotion import HSEmotionModel
from facenn.models.analyzers.emotion_onnx import HSEmotionONNX
from facenn.config import DEVICE
import os

def benchmark_isolated():
    # Setup dummy input (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    print(f"Hardware Device: {DEVICE}")
    print(f"{'Model':<15} | {'Backend':<10} | {'Avg Time (ms)':<15}")
    print("-" * 45)

    scenarios = [
        ("FairFace", FairFaceModel, FairFaceONNX),
        ("HSEmotion", HSEmotionModel, HSEmotionONNX)
    ]

    for name, TorchClass, ONNXClass in scenarios:
        # Torch Benchmark
        try:
            m_torch = TorchClass()
            m_torch.load_model()
            # Warmup
            m_torch.analyze(dummy_input)
            
            start = time.time()
            iters = 50
            for _ in range(iters):
                m_torch.analyze(dummy_input)
            avg_torch = (time.time() - start) / iters * 1000
            print(f"{name:<15} | {'torch':<10} | {avg_torch:>13.2f} ms")
        except Exception as e:
            print(f"{name:<15} | {'torch':<10} | Error: {e}")

        # ONNX Benchmark
        try:
            m_onnx = ONNXClass()
            m_onnx.load_model()
            # Warmup
            m_onnx.analyze(dummy_input)
            
            start = time.time()
            iters = 50
            for _ in range(iters):
                m_onnx.analyze(dummy_input)
            avg_onnx = (time.time() - start) / iters * 1000
            print(f"{name:<15} | {'onnx':<10} | {avg_onnx:>13.2f} ms")
        except Exception as e:
            print(f"{name:<15} | {'onnx':<10} | Error: {e}")

if __name__ == "__main__":
    benchmark_isolated()
