"""
Training script for Cats vs Dogs binary classifier.
Logs all parameters, metrics, and artifacts to MLflow.

Usage:
    python src/train.py --data-dir data/processed --epochs 10 --lr 0.001
"""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.model import get_model
from src.utils import (
    get_train_transforms,
    get_val_transforms,
    compute_accuracy,
    plot_loss_curves,
    plot_confusion_matrix,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MLFLOW_EXPERIMENT = "cats-vs-dogs"


def get_dataloaders(data_dir: Path, batch_size: int, num_workers: int = 2):
    train_ds = ImageFolder(root=str(data_dir / "train"), transform=get_train_transforms())
    val_ds = ImageFolder(root=str(data_dir / "val"), transform=get_val_transforms())
    test_ds = ImageFolder(root=str(data_dir / "test"), transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    # ImageFolder maps classes alphabetically: cat=0, dog=1
    print(f"Class mapping: {train_ds.class_to_idx}")
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(outputs.detach().cpu(), labels.cpu())

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        total_acc += compute_accuracy(outputs.cpu(), labels.cpu())

        preds = (torch.sigmoid(outputs) >= 0.5).long().squeeze().cpu().tolist()
        true = labels.long().squeeze().cpu().tolist()
        all_preds.extend(preds if isinstance(preds, list) else [preds])
        all_labels.extend(true if isinstance(true, list) else [true])

    return total_loss / len(loader), total_acc / len(loader), all_preds, all_labels


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        Path(args.data_dir), args.batch_size
    )

    model = get_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=args.run_name):
        # Log hyperparams
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "model": "ResNet-18",
            "pretrained": True,
            "optimizer": "Adam",
        })

        train_losses, val_losses = [], []

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            print(f"Epoch {epoch:02d}/{args.epochs} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Final test evaluation
        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        print(f"\nTest  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

        # ── Save model checkpoint ──────────────────────────────────────────────
        checkpoint_path = MODEL_DIR / "cat_dog_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "test_acc": test_acc,
            "epochs": args.epochs,
        }, checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path), artifact_path="model")

        # ── Log loss curves ────────────────────────────────────────────────────
        loss_png = plot_loss_curves(train_losses, val_losses)
        loss_path = MODEL_DIR / "loss_curves.png"
        loss_path.write_bytes(loss_png)
        mlflow.log_artifact(str(loss_path), artifact_path="charts")

        # ── Log confusion matrix ───────────────────────────────────────────────
        cm_png = plot_confusion_matrix(test_labels, test_preds)
        cm_path = MODEL_DIR / "confusion_matrix.png"
        cm_path.write_bytes(cm_png)
        mlflow.log_artifact(str(cm_path), artifact_path="charts")

        # ── Log model with MLflow ──────────────────────────────────────────────
        mlflow.pytorch.log_model(model, artifact_path="pytorch_model")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cat vs Dog classifier")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--run-name", type=str, default="baseline-resnet18")
    args = parser.parse_args()
    train(args)
