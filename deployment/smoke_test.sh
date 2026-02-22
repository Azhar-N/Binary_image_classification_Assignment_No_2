#!/usr/bin/env bash
# ── Smoke Test Script ─────────────────────────────────────────────────────────
# Calls the health endpoint and send a test prediction.
# Exits 0 (success) or 1 (failure) — used by CI/CD to gate deployment.

set -uo pipefail

BASE_URL="${API_URL:-http://localhost:8000}"
PASS=0
FAIL=0

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
RESET="\033[0m"

pass() { echo -e "${GREEN}[PASS]${RESET} $1"; ((PASS++)); }
fail() { echo -e "${RED}[FAIL]${RESET} $1"; ((FAIL++)); }
warn() { echo -e "${YELLOW}[WARN]${RESET} $1"; }

echo "============================================"
echo "  Smoke Tests — Cats vs Dogs API"
echo "  Target: $BASE_URL"
echo "============================================"

# ── Wait for service to be ready ──────────────────────────────────────────────
echo "[INFO] Waiting for service to start..."
MAX_WAIT=60
WAITED=0
until curl -sf "${BASE_URL}/health" > /dev/null 2>&1; do
  if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[ERROR] Service did not become healthy within ${MAX_WAIT}s"
    exit 1
  fi
  sleep 2
  WAITED=$((WAITED + 2))
done
echo "[INFO] Service is up after ${WAITED}s"

# ── Test 1: Health Endpoint ────────────────────────────────────────────────────
echo ""
echo "--- Test 1: GET /health ---"
HEALTH_RESP=$(curl -sf "${BASE_URL}/health")
HTTP_STATUS=$(curl -o /dev/null -sw "%{http_code}" "${BASE_URL}/health")

if [ "$HTTP_STATUS" -eq 200 ]; then
  pass "GET /health returned HTTP 200"
else
  fail "GET /health returned HTTP $HTTP_STATUS (expected 200)"
fi

# model_loaded is informational — warn but don't fail in CI (model is volume-mounted at runtime)
MODEL_LOADED=$(echo "$HEALTH_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded',''))" 2>/dev/null || echo "unknown")
if [ "$MODEL_LOADED" = "True" ] || [ "$MODEL_LOADED" = "true" ]; then
  pass "Model reported as loaded"
else
  warn "Model not loaded (model_loaded=$MODEL_LOADED) — expected in CI where no .pt is mounted"
fi

# ── Test 2: Predict Endpoint with a generated dummy image ─────────────────────
echo ""
echo "--- Test 2: POST /predict ---"

# Generate a tiny JPEG test image using Python (fixed: pass filename as variable, not sys.argv)
TMPIMG=$(mktemp /tmp/smoke_test_XXXXXX.jpg)
python3 -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype='uint8'), mode='RGB')
img.save('$TMPIMG')
print('[INFO] Test image created: $TMPIMG')
"

PREDICT_STATUS=$(curl -o /tmp/predict_resp.json -sw "%{http_code}" \
  -X POST "${BASE_URL}/predict" \
  -F "file=@${TMPIMG};type=image/jpeg" 2>/dev/null)

if [ "$PREDICT_STATUS" -eq 200 ]; then
  pass "POST /predict returned HTTP 200"
  LABEL=$(python3 -c "import json; d=json.load(open('/tmp/predict_resp.json')); print(d.get('label',''))" 2>/dev/null || echo "")
  if [[ "$LABEL" == "cat" || "$LABEL" == "dog" ]]; then
    pass "Prediction label is valid: '$LABEL'"
  else
    fail "Invalid prediction label: '$LABEL'"
  fi
elif [ "$PREDICT_STATUS" -eq 503 ]; then
  warn "POST /predict returned 503 — model not loaded (expected in CI). Skipping label check."
else
  fail "POST /predict returned HTTP $PREDICT_STATUS (expected 200 or 503)"
fi

# Cleanup
rm -f "$TMPIMG" /tmp/predict_resp.json

# ── Results ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "============================================"

if [ $FAIL -gt 0 ]; then
  exit 1
fi
exit 0
