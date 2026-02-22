"""
Post-deployment batch simulation for M5.
Sends N requests (mix of cat/dog images) to the running API,
collects predictions, and logs a simple performance report.

Usage:
    python monitoring/simulate_requests.py --n 50 --url http://localhost:8000
"""

import argparse
import time
import random
import io
import requests
import numpy as np
from PIL import Image


def make_dummy_image(color: tuple = None) -> bytes:
    """Generate a synthetic 224×224 JPEG image in memory."""
    if color is None:
        color = tuple(random.randint(0, 255) for _ in range(3))
    arr = np.full((224, 224, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_simulation(base_url: str, n: int):
    predict_url = f"{base_url}/predict"
    health_url = f"{base_url}/health"

    # Check health first
    resp = requests.get(health_url, timeout=5)
    resp.raise_for_status()
    print(f"[Health] {resp.json()}\n")

    results = []
    latencies = []

    for i in range(n):
        img_bytes = make_dummy_image()
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
                results.append(data["label"])
                latencies.append(latency)
                print(f"  [{i+1}/{n}] label={data['label']} "
                      f"conf={data['confidence']:.4f} "
                      f"latency={latency*1000:.1f}ms")
            else:
                print(f"  [{i+1}/{n}] ERROR: HTTP {r.status_code}")
        except Exception as e:
            print(f"  [{i+1}/{n}] EXCEPTION: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if latencies:
        cat_count = results.count("cat")
        dog_count = results.count("dog")
        print("\n" + "=" * 50)
        print("  Simulation Summary")
        print("=" * 50)
        print(f"  Total requests  : {n}")
        print(f"  Successful      : {len(latencies)}")
        print(f"  Cat predictions : {cat_count}")
        print(f"  Dog predictions : {dog_count}")
        print(f"  Avg latency     : {1000*sum(latencies)/len(latencies):.1f} ms")
        print(f"  Min latency     : {1000*min(latencies):.1f} ms")
        print(f"  Max latency     : {1000*max(latencies):.1f} ms")
        print(f"  P95 latency     : {1000*sorted(latencies)[int(0.95*len(latencies))]:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate batch inference requests")
    parser.add_argument("--n", type=int, default=50, help="Number of requests")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    args = parser.parse_args()
    run_simulation(args.url, args.n)
