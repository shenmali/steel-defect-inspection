"""Benchmark equivalent PyTorch and TensorRT GPU inference paths."""
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


class TensorRTEngineRunner:
    """Run a fixed-shape TensorRT engine using PyTorch-owned CUDA buffers."""

    def __init__(self, engine_path: Path) -> None:
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError("TensorRT Python bindings are required for --engine") from error
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT benchmarking requires CUDA")
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Unable to load TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        self._input_name = self._engine.get_tensor_name(0)
        self._output_name = self._engine.get_tensor_name(1)
        input_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        output_shape = tuple(self._engine.get_tensor_shape(self._output_name))
        self._input = torch.empty(input_shape, device="cuda", dtype=torch.float32)
        self._output = torch.empty(output_shape, device="cuda", dtype=torch.float32)
        self._stream = torch.cuda.Stream()
        if not self._context.set_tensor_address(self._input_name, self._input.data_ptr()):
            raise RuntimeError("Unable to bind TensorRT input buffer")
        if not self._context.set_tensor_address(self._output_name, self._output.data_ptr()):
            raise RuntimeError("Unable to bind TensorRT output buffer")

    def __call__(self, image: torch.Tensor) -> None:
        with torch.cuda.stream(self._stream):
            self._input.copy_(image, non_blocking=True)
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT inference failed")
        self._stream.synchronize()


def measure(callable_backend, runs: int) -> list[float]:
    for _ in range(20):
        callable_backend()
    timings = []
    for _ in range(runs):
        started = perf_counter()
        callable_backend()
        timings.append((perf_counter() - started) * 1000)
    return timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks.json"))
    parser.add_argument("--engine", type=Path, action="append", default=[], help="TensorRT engine to compare")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    predictor = PyTorchPredictor(args.checkpoint)
    model_input = predictor._preprocess(image)
    def run_pytorch():
        with torch.inference_mode():
            predictor.model(model_input)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    timings = measure(run_pytorch, args.runs)
    peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    backends = [make_report("pytorch", timings, peak)]
    for engine_path in args.engine:
        runner = TensorRTEngineRunner(engine_path)
        backends.append(make_report(f"tensorrt:{engine_path.name}", measure(lambda: runner(model_input), args.runs), 0.0))
    report = {"environment": {"torch": torch.__version__}, "backends": backends}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
