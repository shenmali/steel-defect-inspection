"""Behavioral tests for the prediction HTTP contract."""

import asyncio
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from steel_inspection.api.storage import AnnotationStorageError, UploadTooLargeError, read_limited_upload, store_annotation
from steel_inspection.inference.types import PredictionResult


class _BoundedReadUpload:
    """Upload double that fails if the service asks for the complete body at once."""

    def __init__(self, body: bytes, maximum_read_size: int) -> None:
        self._body = body
        self._maximum_read_size = maximum_read_size

    async def read(self, size: int = -1) -> bytes:
        if size < 0 or size > self._maximum_read_size:
            raise AssertionError(f"Upload read was not bounded: {size}")
        chunk, self._body = self._body[:size], self._body[size:]
        return chunk


def _prediction_result() -> PredictionResult:
    """Return a small annotation-ready inference result."""
    return PredictionResult(
        has_defect=False,
        defects=[],
        latency_ms=0.0,
        backend="pytorch",
        mask=np.zeros((4, 4, 4), dtype=np.uint8),
    )


def _multipart_body(payload: bytes, boundary: bytes = b"streaming-boundary") -> bytes:
    """Create a small single-file multipart body for direct ASGI tests."""
    return (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="image"; filename="sheet.png"\r\n'
        b"Content-Type: image/png\r\n\r\n"
        + payload
        + b"\r\n--"
        + boundary
        + b"--\r\n"
    )


def _post_asgi_in_chunks(app, body: bytes, boundary: bytes, chunk_size: int) -> tuple[int, int, int]:
    """Post a multipart body directly to ASGI and count body chunks consumed by the app."""
    chunks = [body[index : index + chunk_size] for index in range(0, len(body), chunk_size)]
    consumed = 0
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        nonlocal consumed
        if consumed == len(chunks):
            return {"type": "http.disconnect"}
        body_chunk = chunks[consumed]
        consumed += 1
        return {"type": "http.request", "body": body_chunk, "more_body": consumed < len(chunks)}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/predict",
        "raw_path": b"/predict",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"multipart/form-data; boundary=" + boundary)],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }

    asyncio.run(app(scope, receive, send))

    response_start = next(message for message in messages if message["type"] == "http.response.start")
    return int(response_start["status"]), consumed, len(chunks)


def test_local_request_body_limiter_stops_reading_before_the_full_body() -> None:
    """Catches a request limiter that is unavailable under FastAPI's minimum Starlette dependency."""
    from steel_inspection.api.body_limit import RequestBodyLimitMiddleware

    chunks = [b"abc", b"def", b"ghi"]
    consumed = 0
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        nonlocal consumed
        chunk = chunks[consumed]
        consumed += 1
        return {"type": "http.request", "body": chunk, "more_body": consumed < len(chunks)}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    async def downstream(scope, receive, send) -> None:
        while (message := await receive()).get("more_body", False):
            pass

    scope = {"type": "http", "headers": []}
    asyncio.run(RequestBodyLimitMiddleware(downstream, max_body_size=4)(scope, receive, send))

    assert consumed == 2
    assert sent[0]["status"] == 413


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
def annotating_client(checkpoint: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Create a deterministic app where annotation persistence has been explicitly enabled."""
    from steel_inspection.api.main import create_app
    from steel_inspection.inference import pytorch

    monkeypatch.setattr(pytorch, "create_model", lambda: torch.nn.Conv2d(3, 4, kernel_size=1))
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app(checkpoint, save_annotations=True), raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def png_bytes() -> bytes:
    """A valid colour image with non-model source dimensions."""
    image = np.full((32, 64, 3), (10, 100, 200), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_predict_returns_contract_without_persisting_annotation_by_default(
    client: TestClient, png_bytes: bytes, tmp_path: Path
):
    """Catches endpoints that retain annotation files without an explicit opt-in."""
    response = client.post("/predict", files={"image": ("sheet.png", png_bytes, "image/png")})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {"has_defect", "defects", "latency_ms", "backend", "annotated_image"}
    assert payload["has_defect"] is True
    assert payload["backend"] == "pytorch"
    assert len(payload["defects"]) == 4
    assert payload["annotated_image"] is None
    assert not (tmp_path / "artifacts/results").exists()


def test_predict_rejects_non_image(client: TestClient):
    """Catches upload validation that accepts an unsupported media type."""
    response = client.post("/predict", files={"image": ("notes.txt", b"no", "text/plain")})

    assert response.status_code == 415


def test_predict_rejects_an_upload_larger_than_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Catches routes that retain more than the configured upload limit before rejecting it."""
    from steel_inspection.api import main as api

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)

    response = client.post("/predict", files={"image": ("x.png", b"12345", "image/png")})

    assert response.status_code == 413


def test_asgi_request_limit_stops_before_full_multipart_body_is_spooled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches request caps that run only after FastAPI has parsed every multipart chunk."""
    from steel_inspection.api import main as api

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 100)
    boundary = b"streaming-boundary"
    body = _multipart_body(b"x" * 256, boundary)

    status_code, consumed, total_chunks = _post_asgi_in_chunks(
        api.create_app(tmp_path / "missing.pt"), body, boundary, chunk_size=24
    )

    assert status_code == 413
    assert consumed < total_chunks


def test_read_limited_upload_rejects_before_an_unbounded_body_read() -> None:
    """Catches a reader that calls UploadFile.read() without a bounded chunk size."""
    upload = _BoundedReadUpload(b"12345", maximum_read_size=5)

    with pytest.raises(UploadTooLargeError):
        asyncio.run(read_limited_upload(upload, limit=4))


def test_store_annotation_removes_expired_pngs_and_trims_png_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches retention that leaves expired data, exceeds capacity, or deletes non-PNG files."""
    monkeypatch.chdir(tmp_path)
    results_dir = Path("artifacts/results")
    results_dir.mkdir(parents=True)
    expired = results_dir / "expired.png"
    expired.write_bytes(b"expired")
    expired_at = (expired.stat().st_mtime - timedelta(hours=25).total_seconds())
    os.utime(expired, (expired_at, expired_at))
    oversized_existing = results_dir / "older.png"
    oversized_existing.write_bytes(b"x" * 950)
    unrelated = results_dir / "keep.txt"
    unrelated.write_bytes(b"do not delete")
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    result = _prediction_result()

    saved = store_annotation(
        source,
        result,
        results_dir=results_dir,
        enabled=True,
        ttl=timedelta(hours=24),
        maximum_total_bytes=1024,
    )

    assert saved is not None and saved.is_file()
    assert not expired.exists()
    assert not oversized_existing.exists()
    assert unrelated.read_bytes() == b"do not delete"
    assert sum(path.stat().st_size for path in results_dir.glob("*.png")) <= 1024


def test_store_annotation_rejects_noncanonical_directory_before_cleanup(tmp_path: Path) -> None:
    """Catches retention that can delete files from a caller-controlled results directory."""
    untrusted_results_dir = tmp_path / "outside-results"
    untrusted_results_dir.mkdir()
    retained = untrusted_results_dir / "retained.png"
    retained.write_bytes(b"x" * 950)

    with pytest.raises(AnnotationStorageError):
        store_annotation(
            np.zeros((4, 4, 3), dtype=np.uint8),
            _prediction_result(),
            results_dir=untrusted_results_dir,
            enabled=True,
            maximum_total_bytes=1,
        )

    assert retained.is_file()


def test_store_annotation_rejects_results_symlink_before_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches retention that follows artifacts/results links to delete another directory's files."""
    target = tmp_path / "redirected-results"
    target.mkdir()
    retained = target / "retained.png"
    retained.write_bytes(b"x" * 950)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    try:
        (artifacts / "results").symlink_to(target, target_is_directory=True)
    except OSError as error:
        junction = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(artifacts / "results"), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if junction.returncode != 0:
            pytest.skip(f"Symlinks and junctions are unavailable in this test environment: {error}")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AnnotationStorageError):
        store_annotation(
            np.zeros((4, 4, 3), dtype=np.uint8),
            _prediction_result(),
            results_dir=Path("artifacts/results"),
            enabled=True,
            maximum_total_bytes=1,
        )

    assert retained.is_file()


def test_store_annotation_rejects_redirected_artifacts_parent_before_creating_results_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches retention that creates artifacts/results through a redirected artifacts parent."""
    target = tmp_path / "redirected-artifacts"
    target.mkdir()
    artifacts = tmp_path / "artifacts"
    try:
        artifacts.symlink_to(target, target_is_directory=True)
    except OSError as error:
        junction = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(artifacts), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if junction.returncode != 0:
            pytest.skip(f"Symlinks and junctions are unavailable in this test environment: {error}")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AnnotationStorageError):
        store_annotation(
            np.zeros((4, 4, 3), dtype=np.uint8),
            _prediction_result(),
            results_dir=Path("artifacts/results"),
            enabled=True,
        )

    assert not (target / "results").exists()


def test_store_annotation_rejects_an_oversized_new_png_before_deleting_existing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches capacity handling that deletes retained annotations before rejecting an oversized new one."""
    monkeypatch.chdir(tmp_path)
    results_dir = Path("artifacts/results")
    results_dir.mkdir(parents=True)
    retained = results_dir / "retained.png"
    retained.write_bytes(b"retained")

    with pytest.raises(AnnotationStorageError):
        store_annotation(
            np.zeros((4, 4, 3), dtype=np.uint8),
            _prediction_result(),
            results_dir=results_dir,
            enabled=True,
            maximum_total_bytes=1,
        )

    assert retained.is_file()


def test_predict_hides_annotation_write_failures(
    annotating_client: TestClient, png_bytes: bytes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches annotation write errors that escape as tracebacks or expose filesystem paths."""
    from steel_inspection.api import main as api

    blocked_results_dir = tmp_path / "blocked-results"
    blocked_results_dir.write_bytes(b"not a directory")
    monkeypatch.setattr(api, "RESULTS_DIR", blocked_results_dir)

    response = annotating_client.post("/predict", files={"image": ("sheet.png", png_bytes, "image/png")})

    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to store annotated image"}
    assert str(blocked_results_dir) not in response.text


def test_health_reports_ready_backend(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}


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
