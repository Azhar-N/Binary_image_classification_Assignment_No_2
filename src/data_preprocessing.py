"""
Data Preprocessing Script for Cats vs Dogs Classification
- Downloads/prepares dataset
- Resizes images to 224x224 RGB
- Splits into train/val/test (80/10/10)
- Applies augmentation to training set
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
IMAGE_SIZE = (224, 224)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42
CLASSES = ["cat", "dog"]


def resize_image(src_path: Path, dst_path: Path, size: tuple = IMAGE_SIZE) -> None:
    """Resize a single image to the target size in RGB mode."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        img.save(dst_path)


def get_image_files(directory: Path, extensions: tuple = (".jpg", ".jpeg", ".png")) -> list:
    """Recursively collect image file paths from a directory."""
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def split_files(files: list, ratios: dict, seed: int = RANDOM_SEED) -> dict:
    """Split a list of files into train/val/test sets."""
    random.seed(seed)
    shuffled = files.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train: n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def preprocess_dataset(raw_dir: Path = RAW_DATA_DIR, out_dir: Path = PROCESSED_DATA_DIR) -> dict:
    """
    Main preprocessing pipeline.

    Expects raw_dir to have subdirectories named 'cat' and 'dog'
    (or any structure where filenames contain 'cat'/'dog').

    Returns a dict with split counts per class.
    """
    stats = {}

    for cls in CLASSES:
        # Support both flat structure (files named cat.xxx) and subdirectory structure
        cls_raw_dir = raw_dir / cls
        if cls_raw_dir.exists():
            files = get_image_files(cls_raw_dir)
        else:
            # Fallback: find files in raw_dir whose name starts with cls
            files = [f for f in get_image_files(raw_dir) if f.stem.lower().startswith(cls)]

        if not files:
            print(f"[WARNING] No files found for class '{cls}' in {raw_dir}")
            continue

        splits = split_files(files, SPLIT_RATIOS)
        stats[cls] = {split: len(paths) for split, paths in splits.items()}

        for split, paths in splits.items():
            for src in paths:
                dst = out_dir / split / cls / src.name
                resize_image(src, dst)

        print(f"[{cls}] → train: {stats[cls]['train']}, val: {stats[cls]['val']}, test: {stats[cls]['test']}")

    return stats


def create_dummy_dataset(out_dir: Path = RAW_DATA_DIR, n_per_class: int = 20) -> None:
    """
    Create a tiny dummy dataset (solid-color images) for testing/CI purposes.
    Produces n_per_class images per class in out_dir/<class>/.
    """
    colors = {"cat": (200, 100, 80), "dog": (80, 120, 200)}
    for cls, color in colors.items():
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = Image.fromarray(
                np.full((64, 64, 3), color, dtype=np.uint8), mode="RGB"
            )
            img.save(cls_dir / f"{cls}_{i:04d}.jpg")
    print(f"[Dummy dataset] Created {n_per_class} images per class in '{out_dir}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Cats vs Dogs dataset")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--dummy", action="store_true", help="Create dummy data for testing")
    args = parser.parse_args()

    if args.dummy:
        create_dummy_dataset(args.raw_dir)

    stats = preprocess_dataset(args.raw_dir, args.out_dir)
    print("\nPreprocessing complete. Stats:", stats)
