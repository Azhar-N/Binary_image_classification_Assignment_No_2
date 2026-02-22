# Cats vs Dogs MLOps Pipeline

> **BITS Pilani MLOPS Assignment 2** — Binary image classification with end-to-end MLOps.

---

## Project Structure

```
Assignment2/
├── src/                    # M1: Data preprocessing, model, training, utils
├── app/                    # M2: FastAPI inference service
├── tests/                  # M3: Unit tests (pytest)
├── .github/workflows/      # M3: GitHub Actions CI/CD
├── deployment/             # M4: Docker Compose + smoke tests
├── monitoring/             # M5: Prometheus config + simulation script
├── models/                 # Saved model checkpoints
├── data/                   # DVC-versioned dataset (raw + processed)
├── Dockerfile              # M2: Container spec
├── dvc.yaml                # M1: DVC pipeline
├── params.yaml             # M1: DVC parameters
└── requirements.txt        # Pinned Python dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

Place Kaggle Cats vs Dogs images in `data/raw/` using the structure:
```
data/raw/
├── cat/   (*.jpg images of cats)
└── dog/   (*.jpg images of dogs)
```

Or generate a **dummy dataset for testing**:
```bash
python src/data_preprocessing.py --dummy
```

### 3. Preprocess data (via DVC)

```bash
dvc repro preprocess
```

Or manually:
```bash
# python src/data_preprocessing.py --raw-dir data/raw --out-dir data/processed
python -m src.train --epochs 10 --batch-size 32 --lr 0.001
```

### 4. Train the model

```bash
python src/train.py --epochs 10 --batch-size 32 --lr 0.001
```

View MLflow UI:
```bash
mlflow ui   # → http://localhost:5000
```

### 5. Run the API locally

```bash
uvicorn app.main:app --reload --port 8000
```

Test endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Predict (replace with your image)
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/cat.jpg"

# Prometheus metrics
curl http://localhost:8000/metrics
```

### 6. Build and run Docker image

```bash
docker build -t catdog-classifier:latest .
# docker run -p 8000:8000 -v $(pwd)/models:/app/models catdog-classifier:latest
docker run -p 8000:8000 -v ${PWD}/models:/app/models catdog-classifier:latest
```

### 7. Deploy with Docker Compose

```bash
cd deployment
# DOCKERHUB_USERNAME=yourusername docker compose up -d
DOCKERHUB_USERNAME=yourusername docker compose up -d
bash smoke_test.sh
```

### 8. Run unit tests

```bash
pytest tests/ -v
```

### 9. Simulate batch requests (M5)

```bash
python monitoring/simulate_requests.py --n 50 --url http://localhost:8000
```

---

## Milestones

| Milestone | Description | Status |
|---|---|---|
| M1 | Model Dev + Experiment Tracking (MLflow + DVC) | ✅ |
| M2 | FastAPI + Docker containerization | ✅ |
| M3 | GitHub Actions CI (test → build → push) | ✅ |
| M4 | Docker Compose CD + smoke tests | ✅ |
| M5 | Prometheus metrics + structured logging | ✅ |

---

## CI/CD Setup (GitHub Actions)

No secrets required! The pipeline uses GitHub's built-in `GITHUB_TOKEN` to authenticate with **GitHub Container Registry (GHCR)**.

> **One-time setup:** Go to `Settings → Actions → General → Workflow permissions` and enable **"Read and write permissions"**.

The pipeline runs automatically on every push to `main`:
1. Runs all unit tests (pytest)
2. Builds Docker image (multi-stage)
3. Pushes to `ghcr.io/azhar-n/catdog-classifier`
4. Deploys with Docker Compose
5. Runs smoke tests

---

## API Reference

### `GET /health`
Returns service and model status.
```json
{"status": "ok", "model_loaded": true, "version": "1.0.0"}
```

### `POST /predict`
Upload an image file. Returns classification result.
```json
{
  "label": "cat",
  "confidence": 0.9234,
  "cat_probability": 0.9234,
  "dog_probability": 0.0766
}
```

### `GET /metrics`
Prometheus-format metrics endpoint.

---

## Model Architecture

- **Backbone**: ResNet-18 (ImageNet pretrained)
- **Head**: Dropout(0.3) → Linear(512 → 1)
- **Loss**: BCEWithLogitsLoss
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: StepLR (step=5, gamma=0.5)
- **Input**: 224×224 RGB, ImageNet normalization
