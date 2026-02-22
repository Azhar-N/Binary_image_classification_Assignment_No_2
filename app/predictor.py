"""
Model loading and inference logic.
Loaded once at app startup and shared via FastAPI lifespan.
"""

import os
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from src.model import CatDogModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/cat_dog_model.pt")

# ImageNet normalization (same as training)
INFER_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

CLASSES = ["cat", "dog"]  # Index 0 = cat, Index 1 = dog


class Predictor:
    """Wraps the model for single-image inference."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.loaded = True

    def _load_model(self, path: str) -> CatDogModel:
        model = CatDogModel(pretrained=False)
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """
        Given a PIL Image, return label + probabilities.

        Returns:
            {
                "label": "cat" | "dog",
                "confidence": float,
                "cat_probability": float,
                "dog_probability": float,
            }
        """
        tensor = INFER_TRANSFORMS(image.convert("RGB")).unsqueeze(0).to(self.device)
        logit = self.model(tensor)                 # shape: [1, 1]
        dog_prob = torch.sigmoid(logit).item()     # Probability of "dog"
        cat_prob = 1.0 - dog_prob

        label = CLASSES[int(dog_prob >= 0.5)]
        confidence = dog_prob if label == "dog" else cat_prob

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "cat_probability": round(cat_prob, 4),
            "dog_probability": round(dog_prob, 4),
        }
