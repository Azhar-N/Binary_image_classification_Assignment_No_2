# MLOps Assignment 2 — Complete Run Guide

**Use case:** Binary image classification (Cats vs Dogs) for a pet adoption platform.  
**Stack:** PyTorch · MLflow · DVC · FastAPI · Docker · GitHub Actions · Docker Compose · Prometheus

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.10+ | `python --version` |
| Git | any | `git --version` |
| DVC | 3.x | `python -m dvc version` |
| Docker Desktop | any | `docker --version` |
| curl | any | `curl --version` |

---

## Quick Reference — Ports

| Service | URL |
|---------|-----|
| FastAPI REST API | http://localhost:8000 |
| Swagger / API Docs | http://localhost:8000/docs |
| Prometheus Metrics | http://localhost:8000/metrics |
| MLflow UI | http://localhost:5000 |
| Prometheus Dashboard | http://localhost:9090 |

---

## Setup

```powershell
# 1. Navigate to project root
cd "c:\Users\azhar\BITS\Sem3\MLOPS\Assignment2"

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## M1 — Model Development & Experiment Tracking

### Step 1.1 — Verify dataset

The Kaggle Cats vs Dogs dataset lives in `archive/PetImages/`:

```
archive/PetImages/
├── Cat/   ← 12,499 images
└── Dog/   ← 12,499 images
```

Data is already preprocessed into `data/processed/` (split 80/10/10) and locked in `dvc.lock`.

To check DVC pipeline status:

```powershell
python -m dvc status
# Outputs: Data and pipelines are up to date.
```

### Step 1.2 — Re-run preprocessing (only if data changed)

```powershell
python -m dvc repro preprocess
```

This calls `src/data_preprocessing.py` which:
- Reads images from `data/raw/cat/` and `data/raw/dog/`
- Resizes all images to **224×224 RGB**
- Splits into **80% train / 10% val / 10% test**
- Writes to `data/processed/{train,val,test}/{cat,dog}/`

### Step 1.3 — Train the model

```powershell
python src/train.py --epochs 10 --batch-size 32 --lr 0.001 --run-name baseline-resnet18
```

**What happens:**
- Loads ResNet-18 (pretrained on ImageNet), replaces head with binary output
- Trains with Adam optimizer, BCEWithLogitsLoss
- Logs per-epoch metrics to MLflow
- Saves model to `models/cat_dog_model.pt`
- Saves `models/loss_curves.png` and `models/confusion_matrix.png`

**Expected output:**
```
Using device: cpu   (or cuda if GPU available)
Epoch 01/10 | Train Loss: 0.2341 Acc: 0.9105 | Val Loss: 0.1823 Acc: 0.9312
Epoch 02/10 | Train Loss: 0.1654 Acc: 0.9389 | Val Loss: 0.1392 Acc: 0.9487
...
Epoch 10/10 | Train Loss: 0.0821 Acc: 0.9723 | Val Loss: 0.1234 Acc: 0.9576
Test Loss: 0.1298 | Test Acc: 0.9512
MLflow Run ID: <run_id>
Model saved to: models/cat_dog_model.pt
```

### Step 1.4 — View MLflow experiment tracking

```powershell
# Start MLflow UI (in a separate terminal)
mlflow ui --port 5000
```

Open **http://localhost:5000** and verify:

| What to check | Where to find it |
|---------------|-----------------|
| Experiment name `cats-vs-dogs` | Left sidebar |
| Run name `baseline-resnet18` | Runs table |
| Parameters: `epochs`, `batch_size`, `learning_rate`, `model`, `optimizer`, `pretrained`, `weight_decay` | Run detail → Parameters |
| Metrics: `train_loss`, `train_acc`, `val_loss`, `val_acc` (per epoch) | Run detail → Metrics → view charts |
| Artifacts: `model/cat_dog_model.pt`, `charts/loss_curves.png`, `charts/confusion_matrix.png` | Run detail → Artifacts |

### Step 1.5 — Track dataset with DVC

```powershell
python -m dvc add data/raw data/processed
git add data/raw.dvc data/processed.dvc
git commit -m "data: track dataset and processed splits with DVC"
```

To push data to a remote (optional):
```powershell
python -m dvc remote add -d myremote /tmp/dvc-remote
python -m dvc push
```

✅ **M1 complete** — Git for code, DVC for data, MLflow for experiments, ResNet-18 model trained and saved.

---

## M2 — Model Packaging & Containerization

### Step 2.1 — Run the API locally

```powershell
# Start FastAPI server (in a separate terminal)
uvicorn app.main:app --reload --port 8000
```

**Expected startup log:**
```json
{"time": "...", "level": "INFO", "message": "Loading model..."}
{"time": "...", "level": "INFO", "message": "Model loaded successfully"}
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2.2 — Test the health endpoint

```powershell
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Step 2.3 — Test the predict endpoint

Using any JPEG image (cat or dog):

```powershell
# Windows PowerShell
curl -X POST http://localhost:8000/predict `
     -F "file=@path\to\image.jpg;type=image/jpeg"
```

**Expected response:**
```json
{
  "label": "cat",
  "confidence": 0.9721,
  "cat_probability": 0.9721,
  "dog_probability": 0.0279
}
```

Or use the **interactive Swagger UI** at **http://localhost:8000/docs**:
1. Click `POST /predict` → Try it out
2. Upload any image file
3. Click Execute

### Step 2.4 — Build the Docker image

```powershell
docker build -t catdog-classifier:latest .
```

**Expected output:**
```
[+] Building 120.5s
 => [builder 1/5] FROM python:3.10-slim
 => [builder 4/5] RUN pip install --no-cache-dir --user -r requirements.txt
 => [runtime 3/4] COPY app/ ./app/
 => exporting to image
 => naming to docker.io/library/catdog-classifier:latest
```

### Step 2.5 — Run and verify the Docker container

```powershell
docker run -d -p 8000:8000 -v ${PWD}/models:/app/models --name catdog-test catdog-classifier:latest

# Wait ~15s for startup, then test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@path\to\test.jpg;type=image/jpeg"

# View container logs
docker logs catdog-test

# Cleanup
docker stop catdog-test && docker rm catdog-test
```

✅ **M2 complete** — FastAPI with `/health` + `/predict` + `/metrics`, Dockerfile built and tested, `requirements.txt` with pinned versions.

---

## M3 — CI Pipeline (Tests + Build + Push)

### Step 3.1 — Run unit tests locally

```powershell
python -m pytest tests/ -v --tb=short
```

**Expected output (13 tests):**
```
tests/test_preprocessing.py::TestResizeImage::test_resize_to_224          PASSED
tests/test_preprocessing.py::TestResizeImage::test_converts_to_rgb        PASSED
tests/test_preprocessing.py::TestResizeImage::test_creates_parent_dirs    PASSED
tests/test_preprocessing.py::TestGetImageFiles::test_finds_jpeg_and_png   PASSED
tests/test_preprocessing.py::TestGetImageFiles::test_recursive_search     PASSED
tests/test_preprocessing.py::TestGetImageFiles::test_empty_directory      PASSED
tests/test_preprocessing.py::TestSplitFiles::test_split_ratios            PASSED
tests/test_preprocessing.py::TestSplitFiles::test_no_data_leakage        PASSED
tests/test_preprocessing.py::TestSplitFiles::test_reproducibility         PASSED
tests/test_preprocessing.py::TestCreateDummyDataset::test_creates_images  PASSED
tests/test_preprocessing.py::TestCreateDummyDataset::test_images_are_valid PASSED
tests/test_inference.py::TestModelArchitecture::test_output_shape         PASSED
tests/test_inference.py::TestModelArchitecture::test_output_is_finite     PASSED
tests/test_inference.py::TestModelArchitecture::test_sigmoid_in_range     PASSED
tests/test_inference.py::TestTransforms::test_val_transform_output_shape  PASSED
tests/test_inference.py::TestTransforms::test_val_transform_normalized    PASSED
tests/test_inference.py::TestComputeAccuracy::test_all_correct            PASSED
tests/test_inference.py::TestComputeAccuracy::test_all_wrong              PASSED
tests/test_inference.py::TestComputeAccuracy::test_half_correct           PASSED
tests/test_inference.py::TestInferencePipeline::test_end_to_end_inference PASSED
===== 20 passed in X.XXs =====
```

### Step 3.2 — Push to GitHub and trigger CI

```powershell
# Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/<your-username>/mlops-assignment2.git
git branch -M main
git push -u origin main
```

**Add secrets in GitHub:**  
Go to: `Repository → Settings → Secrets and variables → Actions → New repository secret`

| Secret Name | Value |
|-------------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub → Account Settings → Security → New Access Token |

**CI pipeline stages** (`.github/workflows/ci-cd.yml`):

```
push to main
    │
    ▼
[Job 1: test]
  - Checkout code
  - Set up Python 3.10
  - pip install -r requirements.txt
  - pytest tests/ -v --junit-xml=test-results.xml
  - Upload test results artifact
    │
    ▼ (only if tests pass)
[Job 2: build-and-push]
  - docker buildx build
  - docker push → Docker Hub (sha-<hash> + latest tags)
    │
    ▼
[Job 3: deploy]
  - docker compose up -d
  - bash deployment/smoke_test.sh
  - docker compose down
```

Watch it run at: `https://github.com/<your-username>/mlops-assignment2/actions`

✅ **M3 complete** — pytest for preprocessing & inference, GitHub Actions CI auto-runs on every push, Docker image pushed to registry.

---

## M4 — CD Pipeline & Deployment

### Step 4.1 — Deploy with Docker Compose

```powershell
cd deployment

# Deploy catdog-api + prometheus
docker compose up -d

# Verify both containers are running and healthy
docker compose ps
```

**Expected output:**
```
NAME          IMAGE                           COMMAND        STATUS
catdog-api    localuser/catdog-classifier:latest  "python..."   Up (healthy)
prometheus    prom/prometheus:v2.50.1             "/bin/..."   Up
```

### Step 4.2 — Run smoke tests

> **Note:** Requires Git Bash or WSL on Windows to run bash scripts.

```bash
# Run from Git Bash or WSL
cd "c:/Users/azhar/BITS/Sem3/MLOPS/Assignment2"
bash deployment/smoke_test.sh
```

**Expected output (exit code 0 = success):**
```
============================================
  Smoke Tests — Cats vs Dogs API
  Target: http://localhost:8000
============================================
[INFO] Waiting for service to start...
[INFO] Service is up after 0s

--- Test 1: GET /health ---
[PASS] GET /health returned HTTP 200
[PASS] Model reported as loaded

--- Test 2: POST /predict ---
[PASS] POST /predict returned HTTP 200
[PASS] Prediction label is valid: 'dog'

============================================
  Results: 4 passed, 0 failed
============================================
```

If any test fails, the script exits with code 1, which **fails the CI/CD pipeline**.

### Step 4.3 — Verify CD triggers automatically

After pushing to `main`, the GitHub Actions `deploy` job:
1. Checks out the repo
2. Sets `IMAGE_TAG=sha-<commit_hash>`
3. Runs `docker compose up -d --wait`
4. Runs `bash deployment/smoke_test.sh`
5. Tears down the compose stack (cleanup in CI)

✅ **M4 complete** — Docker Compose deployment, smoke tests gate the pipeline, CD auto-triggers on main branch push.

---

## M5 — Monitoring, Logs & Metrics

### Step 5.1 — View structured request logs

Every prediction request is logged in pseudo-JSON format. View them in the terminal running uvicorn or via Docker:

```powershell
# If running via Docker Compose
docker compose logs -f catdog-api
```

**Expected log format:**
```json
{"time": "2026-02-22 14:15:00,123", "level": "INFO", "message": "Loading model..."}
{"time": "2026-02-22 14:15:02,456", "level": "INFO", "message": "predict | label=cat confidence=0.9721 latency=0.045s file=cat001.jpg"}
{"time": "2026-02-22 14:15:10,789", "level": "INFO", "message": "predict | label=dog confidence=0.8833 latency=0.038s file=dog042.jpg"}
```

Each log entry contains: `label`, `confidence`, `latency`, `filename`.

### Step 5.2 — Check Prometheus metrics endpoint

```powershell
curl http://localhost:8000/metrics
```

**Expected output (excerpt):**
```
# HELP catdog_request_total Total number of requests
# TYPE catdog_request_total counter
catdog_request_total{endpoint="/health",status="200"} 3.0
catdog_request_total{endpoint="/predict",status="200"} 10.0

# HELP catdog_request_latency_seconds Request latency in seconds
# TYPE catdog_request_latency_seconds histogram
catdog_request_latency_seconds_bucket{endpoint="/predict",le="0.05"} 4.0
catdog_request_latency_seconds_bucket{endpoint="/predict",le="0.1"} 8.0
catdog_request_latency_seconds_sum{endpoint="/predict"} 0.694
catdog_request_latency_seconds_count{endpoint="/predict"} 10.0

# HELP catdog_prediction_label_total Count of predicted labels
catdog_prediction_label_total{label="cat"} 6.0
catdog_prediction_label_total{label="dog"} 4.0
```

### Step 5.3 — View Prometheus dashboard

```powershell
start http://localhost:9090
```

In the Prometheus UI, query:

| Query | What it shows |
|-------|--------------|
| `catdog_request_total` | Total requests per endpoint |
| `rate(catdog_request_total[5m])` | Requests per second (5-min window) |
| `catdog_request_latency_seconds_sum / catdog_request_latency_seconds_count` | Average latency |
| `catdog_prediction_label_total` | Cat vs Dog prediction counts |

### Step 5.4 — Run batch simulation (post-deployment)

```powershell
python monitoring/simulate_requests.py --n 50 --url http://localhost:8000
```

**Expected output:**
```
[Health] {'status': 'ok', 'model_loaded': True, 'version': '1.0.0'}

  [1/50] label=dog conf=0.7832 latency=44.2ms
  [2/50] label=cat conf=0.9108 latency=39.7ms
  ...
  [50/50] label=dog conf=0.6521 latency=41.8ms

==================================================
  Simulation Summary
==================================================
  Total requests  : 50
  Successful      : 50
  Cat predictions : 27
  Dog predictions : 23
  Avg latency     : 42.3 ms
  Min latency     : 35.1 ms
  Max latency     : 98.4 ms
  P95 latency     : 87.6 ms
```

After running, re-check `http://localhost:8000/metrics` — counters should have increased by 50.

✅ **M5 complete** — JSON structured logging, Prometheus metrics (request count + latency + prediction counts), post-deployment batch simulation.

---

## Full Pipeline End-to-End Checklist

Run through this checklist to verify every requirement:

### M1 — Model Development & Experiment Tracking
- [ ] `git log --oneline` shows at least 1 commit
- [ ] `python -m dvc status` reports pipeline up to date
- [ ] `models/cat_dog_model.pt` exists and is ~44 MB
- [ ] `models/loss_curves.png` and `models/confusion_matrix.png` exist
- [ ] MLflow UI at http://localhost:5000 shows experiment **cats-vs-dogs** with logged params + metrics

### M2 — Packaging & Containerization
- [ ] `curl http://localhost:8000/health` returns `{"status":"ok","model_loaded":true}`
- [ ] `POST /predict` with an image returns `{"label":"cat/dog", "confidence":...}`
- [ ] `docker images` shows `catdog-classifier:latest`
- [ ] `docker ps` shows the container running with status `(healthy)`

### M3 — CI Pipeline
- [ ] `python -m pytest tests/ -v` shows all tests passing
- [ ] `.github/workflows/ci-cd.yml` exists with test, build, and deploy jobs
- [ ] GitHub Actions tab shows a green pipeline run after push

### M4 — CD Pipeline & Deployment
- [ ] `docker compose ps` (from `deployment/`) shows `catdog-api` and `prometheus` running
- [ ] `bash deployment/smoke_test.sh` outputs `Results: 4 passed, 0 failed`
- [ ] CD job in GitHub Actions ran and completed successfully

### M5 — Monitoring & Logs
- [ ] Uvicorn/Docker logs show JSON-formatted predict entries with latency
- [ ] `curl http://localhost:8000/metrics` returns Prometheus-format metrics
- [ ] `catdog_request_total`, `catdog_request_latency_seconds`, `catdog_prediction_label_total` all visible
- [ ] `python monitoring/simulate_requests.py --n 50` completes successfully

---

## Troubleshooting

### API returns 503 — Model not loaded
```powershell
# Check if model file exists
ls models/cat_dog_model.pt
# If missing, re-run training:
python src/train.py --epochs 10
```

### Docker container exits immediately
```powershell
docker logs catdog-test
# Common cause: model file not mounted
# Fix: ensure -v ${PWD}/models:/app/models in docker run
```

### DVC repro fails
```powershell
# Check if raw data exists
ls data/raw/cat | Select-Object -First 5
# If empty, place images in data/raw/cat/ and data/raw/dog/
```

### pytest — Module not found
```powershell
# Make sure venv is active and deps installed
.venv\Scripts\Activate
pip install -r requirements.txt
python -m pytest tests/ -v
```

### MLflow artifacts missing from UI
```powershell
# Re-run training — test_acc and artifacts will be logged fresh
python src/train.py --epochs 10 --run-name baseline-resnet18-v2
```

### Prometheus shows no targets
Make sure Docker Compose is running (not just uvicorn standalone):
```powershell
cd deployment
docker compose up -d
# Then visit http://localhost:9090/targets — catdog-api should be UP
```
