"""Behavioral tests for the prediction HTTP contract."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A minimal four-channel checkpoint for exercising the real HTTP path."""
    model = torch.nn.Conv2d(3, 4, kernel_size=1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(1.0)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": [256, 512],
            "class_names": ["class_1", "class_2", "class_3", "class_4"],
        },
        path,
    )
    return path


@pytest.fixture
def client(checkpoint: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Create an app backed by a tiny deterministic model."""
    from steel_inspection.api.main import create_app
    from steel_inspection.inference import pytorch

    monkeypatch.setattr(pytorch, "create_model", lambda: torch.nn.Conv2d(3, 4, kernel_size=1))
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app(checkpoint)) as test_client:
        yield test_client


@pytest.fixture
def png_bytes() -> bytes:
    """A valid colour image with non-model source dimensions."""
    image = np.full((32, 64, 3), (10, 100, 200), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_predict_returns_contract_and_persists_annotation(client: TestClient, png_bytes: bytes, tmp_path: Path):
    """Catches endpoints that omit result metadata or fail to save the PNG overlay."""
    response = client.post("/predict", files={"image": ("sheet.png", png_bytes, "image/png")})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {"has_defect", "defects", "latency_ms", "backend", "annotated_image"}
    assert payload["has_defect"] is True
    assert payload["backend"] == "pytorch"
    assert len(payload["defects"]) == 4
    assert (tmp_path / payload["annotated_image"]).is_file()


def test_predict_rejects_non_image(client: TestClient):
    """Catches upload validation that accepts an unsupported media type."""
    response = client.post("/predict", files={"image": ("notes.txt", b"no", "text/plain")})

    assert response.status_code == 415


def test_predict_rejects_corrupt_image_data(client: TestClient):
    """Catches image decoding errors that would otherwise become server errors."""
    response = client.post("/predict", files={"image": ("broken.png", b"not a PNG", "image/png")})

    assert response.status_code == 422


def test_predict_reports_unavailable_model_as_service_unavailable(tmp_path: Path):
    """Catches unavailable model artifacts being exposed as a generic server error."""
    from steel_inspection.api.main import create_app

    with TestClient(create_app(tmp_path / "missing.pt")) as unavailable_client:
        response = unavailable_client.post(
            "/predict", files={"image": ("sheet.png", b"not needed", "image/png")}
        )

    assert response.status_code == 503
