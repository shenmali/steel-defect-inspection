from pathlib import Path


def test_dockerfile_installs_opencv_runtime_libraries() -> None:
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")

    assert "libgl1" in dockerfile
    assert "libglib2.0-0" in dockerfile
