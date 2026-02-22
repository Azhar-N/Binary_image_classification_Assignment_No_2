"""
Post-deployment batch simulation for M5.
Sends N requests (mix of cat/dog images) to the running API,
collects predictions vs simulated true labels, and logs a performance report.

Usage:
    python monitoring/simulate_requests.py --n 50 --url http://localhost:8000
"""

import argparse
import time
import random
import io
import json
from datetime import datetime
import requests
import numpy as np
from PIL import Image


# Simulated ground truth: orange-tinted images → "cat", blue-tinted → "dog"
LABEL_PROFILES = {
    "cat": (200, 120, 80),   # warm/orange tone
    "dog": (80,  120, 200),  # cool/blue tone
}


def make_dummy_image(color: tuple) -> bytes:
    """Generate a synthetic 224×224 JPEG image in memory with a fixed color."""
    arr = np.full((224, 224, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_simulation(base_url: str, n: int):
    predict_url = f"{base_url}/predict"
    health_url  = f"{base_url}/health"

    # Check health first
    resp = requests.get(health_url, timeout=5)
    resp.raise_for_status()
    print(f"[Health] {resp.json()}\n")

    results   = []   # (true_label, predicted_label, confidence, latency)
    latencies = []

    labels_cycle = (["cat"] * (n // 2)) + (["dog"] * (n - n // 2))
    random.shuffle(labels_cycle)

    for i, true_label in enumerate(labels_cycle):
        color     = LABEL_PROFILES[true_label]
        img_bytes = make_dummy_image(color)

        start = time.perf_counter()
        try:
            r = requests.post(
                predict_url,
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
                timeout=10,
            )
            latency = time.perf_counter() - start

            if r.status_code == 200:
                data = r.json()
                pred_label = data["label"]
                confidence = data["confidence"]
                results.append((true_label, pred_label, confidence, latency))
                latencies.append(latency)

                match = "✓" if pred_label == true_label else "✗"
                print(
                    f"  [{i+1:02d}/{n}] true={true_label:<3}  pred={pred_label:<3}  "
                    f"conf={confidence:.4f}  latency={latency*1000:.1f}ms  {match}"
                )
            else:
                print(f"  [{i+1:02d}/{n}] ERROR: HTTP {r.status_code}")
        except Exception as e:
            print(f"  [{i+1:02d}/{n}] EXCEPTION: {e}")

    # ── Performance Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Post-Deployment Performance Report")
    print(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 55)

    if not results:
        print("  No successful predictions recorded.")
        return

    total        = len(results)
    correct      = sum(1 for t, p, _, __ in results if t == p)
    accuracy     = correct / total

    cat_total    = sum(1 for t, _, __, ___ in results if t == "cat")
    cat_correct  = sum(1 for t, p, _, __ in results if t == "cat" and p == "cat")
    dog_total    = sum(1 for t, _, __, ___ in results if t == "dog")
    dog_correct  = sum(1 for t, p, _, __ in results if t == "dog" and p == "dog")

    avg_conf     = sum(c for _, __, c, ___ in results) / total
    avg_lat      = sum(l for _, __, ___, l in results) / total
    p95_lat      = sorted(l for _, __, ___, l in results)[int(0.95 * total)]

    print(f"  Total requests     : {n}")
    print(f"  Successful         : {total}")
    print(f"  Overall Accuracy   : {accuracy*100:.1f}%  ({correct}/{total})")
    print(f"  Cat accuracy       : {cat_correct}/{cat_total}" if cat_total else "  Cat accuracy: N/A")
    print(f"  Dog accuracy       : {dog_correct}/{dog_total}" if dog_total else "  Dog accuracy: N/A")
    print(f"  Avg confidence     : {avg_conf:.4f}")
    print(f"  Avg latency        : {avg_lat*1000:.1f} ms")
    print(f"  P95 latency        : {p95_lat*1000:.1f} ms")
    print(f"  Min latency        : {min(l for _,__,___,l in results)*1000:.1f} ms")
    print(f"  Max latency        : {max(l for _,__,___,l in results)*1000:.1f} ms")
    print("=" * 55)

    # ── Save JSON report ───────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_requests": n,
        "successful": total,
        "accuracy": round(accuracy, 4),
        "cat_accuracy": round(cat_correct / cat_total, 4) if cat_total else None,
        "dog_accuracy": round(dog_correct / dog_total, 4) if dog_total else None,
        "avg_confidence": round(avg_conf, 4),
        "avg_latency_ms": round(avg_lat * 1000, 2),
        "p95_latency_ms": round(p95_lat * 1000, 2),
    }
    report_path = "monitoring/performance_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate batch inference requests")
    parser.add_argument("--n",   type=int, default=50, help="Number of requests")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    args = parser.parse_args()
    run_simulation(args.url, args.n)
