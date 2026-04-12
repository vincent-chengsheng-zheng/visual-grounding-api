"""
scripts/benchmark_latency.py

Benchmark inference latency for all models.
Measures single-image and batch throughput.

Usage:
    python scripts/benchmark_latency.py

Output:
    results/latency_benchmark.json
"""

import base64
import json
import time
import requests
import sys

API_URL = "http://localhost:8000"

def check_server():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def benchmark(image_b64, text, model, n_warmup=2, n_runs=10):
    """Run n_runs predictions and return latency stats."""
    payload = {"image": image_b64, "text": text, "model": model}

    # Warmup
    for _ in range(n_warmup):
        requests.post(f"{API_URL}/predict", json=payload, timeout=120)

    # Benchmark
    latencies = []
    inference_times = []
    for _ in range(n_runs):
        t0 = time.time()
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=120)
        roundtrip = (time.time() - t0) * 1000
        data = r.json()
        latencies.append(roundtrip)
        inference_times.append(data["inference_time_ms"])

    return {
        "model": model,
        "n_runs": n_runs,
        "roundtrip_ms": {
            "mean": round(sum(latencies) / len(latencies), 1),
            "min": round(min(latencies), 1),
            "max": round(max(latencies), 1),
        },
        "gpu_inference_ms": {
            "mean": round(sum(inference_times) / len(inference_times), 1),
            "min": round(min(inference_times), 1),
            "max": round(max(inference_times), 1),
        },
        "api_overhead_ms": round(
            (sum(latencies) - sum(inference_times)) / len(latencies), 1
        ),
    }

def main():
    if not check_server():
        print("Error: API server not running at", API_URL)
        print("Start it with: uvicorn api.server:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # Load a test image
    import glob
    examples = glob.glob("results/main/examples/*.png")
    if not examples:
        print("Error: No example images found in results/main/examples/")
        sys.exit(1)

    with open(examples[0], "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    text = "the person"
    models = requests.get(f"{API_URL}/health").json()["models_loaded"]

    print(f"Benchmarking {len(models)} models, 10 runs each...\n")

    results = {}
    for model in models:
        print(f"  {model}...", end=" ", flush=True)
        stats = benchmark(image_b64, text, model)
        results[model] = stats
        print(f"gpu={stats['gpu_inference_ms']['mean']}ms  roundtrip={stats['roundtrip_ms']['mean']}ms")

    # Save
    output = {
        "benchmark": results,
        "config": {
            "image": examples[0],
            "text": text,
            "n_warmup": 2,
            "n_runs": 10,
        },
    }
    with open("results/latency_benchmark.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: results/latency_benchmark.json")

if __name__ == "__main__":
    main()
