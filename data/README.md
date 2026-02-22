This directory will contain the dataset tracked by DVC.

Structure:
  data/raw/cat/    ← cat images (*.jpg)
  data/raw/dog/    ← dog images (*.jpg)

To add the Kaggle dataset:
  1. Download cats-vs-dogs dataset from Kaggle
  2. Extract into data/raw/ with cat/ and dog/ subdirectories
  3. Run: python src/data_preprocessing.py --raw-dir data/raw --out-dir data/processed
  4. Run: dvc add data/raw data/processed
  5. Run: git add data/raw.dvc data/processed.dvc
  6. Run: git commit -m "data: add DVC-tracked dataset"

For testing (no real dataset):
  python src/data_preprocessing.py --dummy
