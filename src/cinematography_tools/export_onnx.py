"""
Autonomous ONNX Export script for cinematography-analysis-tools.

Traces the dynamically built native PyTorch model into a pure mathematical
ONNX graph binary. This allows incredibly fast GPU/CPU execution using
C++ engines (like onnxruntime) without needing the Python environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import load_model

def export_to_onnx(output_path: str = "resnet50_cinematography.onnx", model_path: str = None):
    """Bake the active PyTorch environment weights into a standalone .onnx file."""
    output_obj = Path(output_path)
    
    # Load the pure PyTorch model architecture and weights
    print(f"Loading native PyTorch network...")
    model, _ = load_model(path_base=model_path, device="cpu")
    model.eval()
    
    # Create a dummy input tensor that exactly mimics a processed image
    # Standard format: (Batch_Size, Channels, Height, Width) -> (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Tracing nodes into ONNX graph constraint: {output_obj.resolve()}")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_obj,
        export_params=True,
        opset_version=14,              # Latest widely supported ONNX opset
        do_constant_folding=True,      # Extremely aggressive optimization
        input_names=["input_tensor"],
        output_names=["classification_logits"],
        dynamic_axes={
            "input_tensor": {0: "batch_size"},         # Allow batch inferencing dynamically
            "classification_logits": {0: "batch_size"}
        }
    )
    
    import os
    size_mb = os.path.getsize(output_obj) / (1024 * 1024)
    print(f"✅ Executed Successfully! Baked static Graph size: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export the PyTorch model to ONNX.")
    parser.add_argument("--output", "-o", default="resnet50_cinematography.onnx", help="Path to save the ONNX binary.")
    parser.add_argument("--model-path", "-m", default=None, help="Directory containing the pytorch weights.")
    
    args = parser.parse_args()
    export_to_onnx(output_path=args.output, model_path=args.model_path)


if __name__ == "__main__":
    main()
