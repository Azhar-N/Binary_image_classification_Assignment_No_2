"""
Shared utilities: transforms, metrics, visualization.
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers/CI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
from torchvision import transforms


# ── Image Transforms ──────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmentation pipeline for training images."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic pipeline for validation/test/inference images."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary accuracy from logits and 0/1 labels."""
    predicted = (torch.sigmoid(preds) >= 0.5).long().squeeze()
    correct = (predicted == labels.squeeze()).sum().item()
    return correct / len(labels)


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses: list, val_losses: list) -> bytes:
    """Return PNG bytes of train/val loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss", marker="o")
    ax.plot(val_losses, label="Val Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_confusion_matrix(y_true: list, y_pred: list, class_names: list = ["Cat", "Dog"]) -> bytes:
    """Return PNG bytes of a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
