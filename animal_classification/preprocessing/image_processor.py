from PIL import Image
import io
import torch
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """
    Process image bytes and apply ResNet preprocessing transforms.
    
    Args:
        image_bytes: Raw image data as bytes
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for ResNet inference
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply transforms and add batch dimension
    tensor = preprocess(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor