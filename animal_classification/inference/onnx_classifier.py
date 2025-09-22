import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Union, List, Dict

from animal_classification.preprocessing.onnx_image_processor import preprocess_image_for_onnx


class ONNXInference:
    def __init__(self, model_path: Union[str, Path], class_names: List[str] = None):
        self.model_path = Path(model_path)
        self.session = None
        self.class_names = class_names or ['Buffalo', 'Elephant', 'Rhino', 'Zebra']
        self._loaded = False

    def setup(self):
        if self._loaded:
            return

        self._load_model()
        self._load_metadata()
        self._loaded = True

    def teardown(self):
        self.session = None
        self._loaded = False

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file '{self.model_path}' not found")

        # Create ONNX Runtime session with optimizations
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )

    def _load_metadata(self):
        metadata_path = self.model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.class_names = metadata.get('class_names', self.class_names)

    def predict_from_path(self, image_path: Union[str, Path]) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file '{image_path}' not found")

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        return self.predict_from_bytes(image_bytes)

    def predict_from_bytes(self, image_bytes: bytes) -> str:
        input_array = preprocess_image_for_onnx(image_bytes)

        # Run inference
        outputs = self.session.run(None, {'input': input_array})
        predicted_class_idx = np.argmax(outputs[0], axis=1)[0]

        return self.class_names[predicted_class_idx]

    def predict_with_confidence_from_bytes(self, image_bytes: bytes) -> Dict[str, Union[str, int, float]]:
        input_array = preprocess_image_for_onnx(image_bytes)

        # Run inference
        outputs = self.session.run(None, {'input': input_array})
        logits = outputs[0]

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        predicted_class_idx = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][predicted_class_idx]

        return {
            "classification_label": self.class_names[predicted_class_idx],
            "class": int(predicted_class_idx),
            "confidence": float(confidence)
        }