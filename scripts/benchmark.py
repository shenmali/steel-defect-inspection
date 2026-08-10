"""Generate a reproducible PyTorch inference benchmark report."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from steel_inspection.inference.pytorch import PyTorchPredictor


def make_report(backend: str, timings_ms: list[float], peak_gpu_memory_mb: float) -> dict[str, float | str]:
    return {"backend": backend, "mean_latency_ms": float(np.mean(timings_ms)), "p95_latency_ms": float(np.percentile(timings_ms, 95)), "fps": 1000.0 / float(np.mean(timings_ms)), "peak_gpu_memory_mb": peak_gpu_memory_mb}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks.json"))
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    predictor = PyTorchPredictor(args.checkpoint)
    for _ in range(20): predictor.predict(image)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(args.runs):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started = perf_counter(); predictor.predict(image)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timings.append((perf_counter() - started) * 1000)
    peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    report = {"environment": {"torch": torch.__version__}, "backends": [make_report("pytorch", timings, peak)]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
