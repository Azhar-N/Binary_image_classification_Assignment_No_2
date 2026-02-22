"""
Unit tests for model inference functions.
Uses a randomly initialized model (no checkpoint required) to test inference shape/output.
"""

import sys
from pathlib import Path
import pytest
import torch
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CatDogModel, get_model
from src.utils import get_val_transforms, compute_accuracy


class TestModelArchitecture:
    def test_output_shape(self):
        """Model should output [batch, 1] logits."""
        model = get_model(pretrained=False)
        model.eval()
        dummy = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (4, 1)

    def test_output_is_finite(self):
        """Logits should not be NaN or Inf."""
        model = get_model(pretrained=False)
        model.eval()
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert torch.isfinite(out).all()

    def test_sigmoid_in_range(self):
        """Sigmoid of logits should be in [0, 1]."""
        model = get_model(pretrained=False)
        model.eval()
        dummy = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy)
        probs = torch.sigmoid(logits)
        assert (probs >= 0).all() and (probs <= 1).all()


class TestTransforms:
    def test_val_transform_output_shape(self):
        """Val transform should produce [3, 224, 224] tensor."""
        transform = get_val_transforms()
        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_normalized(self):
        """After normalization, pixel values should not be strictly in [0, 1]."""
        transform = get_val_transforms()
        # Black image (0,0,0): after ToTensor → 0.0, then (0.0 - mean) / std < 0 for all channels
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        tensor = transform(img)
        # Black image normalized with ImageNet stats goes below 0 (e.g. R: -0.485/0.229 ≈ -2.12)
        assert tensor.min().item() < 0.0


class TestComputeAccuracy:
    def test_all_correct(self):
        """Perfect predictions should give accuracy 1.0."""
        preds = torch.tensor([[5.0], [-5.0], [5.0]])   # logits
        labels = torch.tensor([[1.0], [0.0], [1.0]])
        assert compute_accuracy(preds, labels) == pytest.approx(1.0)

    def test_all_wrong(self):
        """All wrong predictions should give accuracy 0.0."""
        preds = torch.tensor([[-5.0], [5.0]])
        labels = torch.tensor([[1.0], [0.0]])
        assert compute_accuracy(preds, labels) == pytest.approx(0.0)

    def test_half_correct(self):
        """Half correct should give accuracy 0.5."""
        preds = torch.tensor([[5.0], [5.0]])
        labels = torch.tensor([[1.0], [0.0]])
        assert compute_accuracy(preds, labels) == pytest.approx(0.5)


class TestInferencePipeline:
    """Integration-style test: image → transform → model → prediction."""

    def test_end_to_end_inference(self):
        """Should produce a valid prediction dict structure."""
        model = get_model(pretrained=False)
        model.eval()
        transform = get_val_transforms()

        # Create a random PIL image
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            logit = model(tensor)

        dog_prob = torch.sigmoid(logit).item()
        cat_prob = 1.0 - dog_prob
        label = "dog" if dog_prob >= 0.5 else "cat"

        assert label in ("cat", "dog")
        assert 0.0 <= dog_prob <= 1.0
        assert 0.0 <= cat_prob <= 1.0
        assert abs(dog_prob + cat_prob - 1.0) < 1e-5
