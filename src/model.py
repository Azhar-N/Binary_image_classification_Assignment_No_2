"""
Model definition for Cats vs Dogs binary classification.
Uses ResNet-18 as a backbone with a custom binary output head.
"""

import torch
import torch.nn as nn
from torchvision import models


class CatDogModel(nn.Module):
    """ResNet-18 fine-tuned for binary cat/dog classification."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Optionally freeze backbone for transfer learning
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the classifier head with a binary output
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1),  # Single logit → sigmoid → probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # Returns raw logit


def get_model(pretrained: bool = True, freeze_backbone: bool = False) -> CatDogModel:
    """Factory function to create and return the model."""
    return CatDogModel(pretrained=pretrained, freeze_backbone=freeze_backbone)


def load_model(checkpoint_path: str, device: str = "cpu") -> CatDogModel:
    """Load a model from a saved checkpoint file."""
    model = CatDogModel(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and checkpoint dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


if __name__ == "__main__":
    model = get_model(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Model output shape: {out.shape}")  # Expected: [2, 1]
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
