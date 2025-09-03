import torch
import torch.nn.functional as F
import os
from pathlib import Path
from typing import Union, List, Dict

from animal_classification.models.resnet_classifier import ResnetClassifier
from animal_classification.preprocessing.image_processor import process_image_bytes


class ResNetInference:
    def __init__(self, model_path: Union[str, Path], class_names: List[str] = None):
        self.model_path = Path(model_path)
        self.model = None
        self.class_names = class_names or ['Buffalo', 'Elephant', 'Rhino', 'Zebra']
        self._loaded = False

    def setup(self):
        if self._loaded:
            return

        self._load_model()
        self._loaded = True

    def teardown(self):
        self.model = None
        self._loaded = False
        torch.cuda.empty_cache()
    
    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file '{self.model_path}' not found")
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        num_classes = len(self.class_names)
        self.model = ResnetClassifier(num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
    
    def predict_from_path(self, image_path: Union[str, Path]) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file '{image_path}' not found")
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        return self.predict_from_bytes(image_bytes)
    
    def predict_from_bytes(self, image_bytes: bytes) -> str:
        input_tensor = process_image_bytes(image_bytes)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
        
        return self.class_names[predicted_class_idx]
    
    def predict_with_confidence_from_bytes(self, image_bytes: bytes) -> Dict[str, Union[str, int, float]]:
        input_tensor = process_image_bytes(image_bytes)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)
            
        return {
            "classification_label": self.class_names[predicted_class_idx.item()],
            "class": predicted_class_idx.item(),
            "confidence": confidence.item()
        }