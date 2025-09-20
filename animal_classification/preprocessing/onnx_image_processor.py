from PIL import Image
import numpy as np
import io

def preprocess_image_for_onnx(image_bytes: bytes) -> np.ndarray:
    """
    Lightweight image preprocessing for ONNX inference.
    Uses only PIL and numpy - no torch dependencies.

    Args:
        image_bytes: Raw image data as bytes

    Returns:
        np.ndarray: Preprocessed image array ready for ONNX inference [1, 3, 224, 224]
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize and crop to 224x224 (ResNet input size)
    image = image.resize((256, 256), Image.LANCZOS)

    # Center crop to 224x224
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32)

    # Normalize using ImageNet means and stds
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

    image_array = (image_array - mean) / std

    # Convert from HWC to CHW format and add batch dimension
    image_array = np.transpose(image_array, (2, 0, 1))  # HWC to CHW
    image_array = np.expand_dims(image_array, axis=0)    # Add batch dimension

    # Ensure the final array is float32
    return image_array.astype(np.float32)