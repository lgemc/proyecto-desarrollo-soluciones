#!/usr/bin/env python3
"""
Script to convert PyTorch ResNet model to ONNX format for optimized inference.
"""
import torch
import torch.onnx
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from animal_classification.models.resnet_classifier import ResnetClassifier


def convert_model_to_onnx(
    pytorch_model_path: Path,
    onnx_model_path: Path,
    num_classes: int = 4,
    input_shape: tuple = (1, 3, 224, 224)
):
    """Convert PyTorch model to ONNX format."""
    print(f"Loading PyTorch model from: {pytorch_model_path}")

    # Load the PyTorch model
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')
    model = ResnetClassifier(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(*input_shape)

    print(f"Converting to ONNX format...")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    # Save class names alongside ONNX model
    class_names = checkpoint.get('class_names', ['Buffalo', 'Elephant', 'Rhino', 'Zebra'])
    metadata_path = onnx_model_path.with_suffix('.json')

    import json
    with open(metadata_path, 'w') as f:
        json.dump({
            'class_names': class_names,
            'input_shape': list(input_shape),
            'num_classes': num_classes
        }, f, indent=2)

    print(f"ONNX model saved to: {onnx_model_path}")
    print(f"Metadata saved to: {metadata_path}")

    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful!")
    except ImportError:
        print("Warning: onnx package not available for verification")
    except Exception as e:
        print(f"Warning: ONNX model verification failed: {e}")


if __name__ == "__main__":
    pytorch_model_path = project_root / "models" / "animal-classifier-resnet.pth"
    onnx_model_path = project_root / "models" / "animal-classifier-resnet.onnx"

    if not pytorch_model_path.exists():
        print(f"Error: PyTorch model not found at {pytorch_model_path}")
        sys.exit(1)

    # Create models directory if it doesn't exist
    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)

    convert_model_to_onnx(pytorch_model_path, onnx_model_path)
    print("Model conversion completed!")