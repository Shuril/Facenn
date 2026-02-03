import torch
import os
import onnx
from facenn.config import DEVICE, Config

def export_to_onnx(model, output_path, input_shape=(1, 3, 112, 112), opset_version=12):
    """
    Export a PyTorch model to ONNX format.
    """
    model.eval()
    model.to("cpu") # Export usually strictly on CPU to avoid device ops in graph
    
    dummy_input = torch.randn(*input_shape).to("cpu")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Successfully exported model to {output_path}")
        
        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified.")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False
