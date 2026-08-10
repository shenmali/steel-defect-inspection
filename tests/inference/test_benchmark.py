import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2] / "scripts"))
from benchmark import make_report


def test_benchmark_report_has_required_metrics():
    report = make_report("pytorch", [1.0, 2.0], 42.0)
    assert set(report) == {"backend", "mean_latency_ms", "p95_latency_ms", "fps", "peak_gpu_memory_mb"}
    assert report["backend"] == "pytorch"
    assert report["fps"] > 0
