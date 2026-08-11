from pathlib import Path


def test_dockerfile_installs_opencv_runtime_libraries() -> None:
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")

    assert "libgl1" in dockerfile
    assert "libglib2.0-0" in dockerfile


def test_tensorrt_dockerfile_pins_the_engine_runtime() -> None:
    dockerfile = (Path(__file__).parents[1] / "Dockerfile.tensorrt").read_text(encoding="utf-8")

    assert dockerfile.splitlines()[0] == "FROM steel-defect-inspection"
    assert "tensorrt-cu12==11.2.1.2" in dockerfile
    assert "cuda-python==13.3.1" in dockerfile
