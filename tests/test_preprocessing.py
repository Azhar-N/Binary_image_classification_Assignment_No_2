"""
Unit tests for data_preprocessing.py functions.
Tests: resize_image, get_image_files, split_files, create_dummy_dataset
"""

import os
import pytest
from pathlib import Path
from PIL import Image
import tempfile

# Ensure src is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import (
    resize_image,
    get_image_files,
    split_files,
    create_dummy_dataset,
    preprocess_dataset,
    IMAGE_SIZE,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for each test."""
    return tmp_path


class TestResizeImage:
    def test_resize_to_224(self, temp_dir):
        """Image should be saved at 224×224."""
        src = temp_dir / "src.jpg"
        dst = temp_dir / "dst.jpg"
        # Create a 50×80 image
        Image.new("RGB", (50, 80), color=(128, 64, 32)).save(src)
        resize_image(src, dst)
        with Image.open(dst) as img:
            assert img.size == IMAGE_SIZE

    def test_converts_to_rgb(self, temp_dir):
        """RGBA/L/P images should be converted to RGB."""
        src = temp_dir / "rgba.png"
        dst = temp_dir / "rgb.jpg"
        Image.new("RGBA", (100, 100), color=(10, 20, 30, 128)).save(src)
        resize_image(src, dst)
        with Image.open(dst) as img:
            assert img.mode == "RGB"

    def test_creates_parent_dirs(self, temp_dir):
        """Destination parent directories should be created automatically."""
        src = temp_dir / "src.jpg"
        dst = temp_dir / "a" / "b" / "c" / "dst.jpg"
        Image.new("RGB", (100, 100)).save(src)
        resize_image(src, dst)
        assert dst.exists()


class TestGetImageFiles:
    def test_finds_jpeg_and_png(self, temp_dir):
        """Should collect .jpg, .jpeg, and .png files."""
        for name in ["a.jpg", "b.jpeg", "c.png", "d.txt", "e.csv"]:
            (temp_dir / name).touch()
        files = get_image_files(temp_dir)
        assert len(files) == 3

    def test_recursive_search(self, temp_dir):
        """Should search nested directories."""
        sub = temp_dir / "nested"
        sub.mkdir()
        (temp_dir / "top.jpg").touch()
        (sub / "deep.png").touch()
        files = get_image_files(temp_dir)
        assert len(files) == 2

    def test_empty_directory(self, temp_dir):
        """Should return empty list for empty directory."""
        assert get_image_files(temp_dir) == []


class TestSplitFiles:
    def test_split_ratios(self):
        """80/10/10 split should produce correct proportions."""
        files = list(range(100))
        splits = split_files(files, {"train": 0.8, "val": 0.1, "test": 0.1})
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_no_data_leakage(self):
        """No file should appear in more than one split."""
        files = list(range(200))
        splits = split_files(files, {"train": 0.8, "val": 0.1, "test": 0.1})
        all_files = splits["train"] + splits["val"] + splits["test"]
        assert len(all_files) == len(set(all_files))

    def test_reproducibility(self):
        """Same seed should produce same splits."""
        files = list(range(50))
        s1 = split_files(files, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
        s2 = split_files(files, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
        assert s1["train"] == s2["train"]


class TestCreateDummyDataset:
    def test_creates_images(self, temp_dir):
        """Dummy dataset should create n images per class."""
        create_dummy_dataset(temp_dir, n_per_class=5)
        cat_files = list((temp_dir / "cat").glob("*.jpg"))
        dog_files = list((temp_dir / "dog").glob("*.jpg"))
        assert len(cat_files) == 5
        assert len(dog_files) == 5

    def test_images_are_valid(self, temp_dir):
        """Created images should be openable."""
        create_dummy_dataset(temp_dir, n_per_class=3)
        for f in (temp_dir / "cat").glob("*.jpg"):
            with Image.open(f) as img:
                assert img.mode == "RGB"
