"""FastAPI application exposing one image-segmentation endpoint."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status

from steel_inspection.inference.annotate import annotate_mask
from steel_inspection.inference.pytorch import ModelUnavailableError, PyTorchPredictor
from steel_inspection.inference.types import PredictionResult


SUPPORTED_IMAGE_TYPES = frozenset({"image/bmp", "image/jpeg", "image/png", "image/webp"})
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
RESULTS_DIR = Path("artifacts/results")


def create_app(model_path: Path, backend: str = "pytorch") -> FastAPI:
    """Create a prediction service and load its backend exactly once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = None
        app.state.backend_error = None
        try:
            if backend != "pytorch":
                raise ModelUnavailableError(f"Inference backend is unavailable: {backend}")
            app.state.predictor = PyTorchPredictor(Path(model_path))
        except ModelUnavailableError as error:
            app.state.backend_error = str(error)
        yield

    app = FastAPI(title="Steel Defect Inspection API", lifespan=lifespan)
    app.state.predictor = None
    app.state.backend_error = None

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Report readiness without exposing implementation details."""
        if app.state.backend_error is not None or app.state.predictor is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model backend is unavailable")
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(image: UploadFile = File(...)) -> dict[str, object]:
        """Validate one image upload, segment it, and persist its annotation."""
        if app.state.backend_error is not None or app.state.predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=app.state.backend_error or "Inference backend is unavailable",
            )
        if image.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Upload a PNG, JPEG, WEBP, or BMP image",
            )

        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Image upload is empty")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image exceeds 10 MiB")
        decoded = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Image data is unreadable")

        result = app.state.predictor.predict(decoded)
        annotated_path = _store_annotation(decoded, result)
        return _response_payload(result, annotated_path)

    return app


def _store_annotation(source_bgr: np.ndarray, result: PredictionResult) -> Path:
    """Encode the annotated result into the predictable local results directory."""
    annotated = annotate_mask(source_bgr, result.mask)
    success, encoded = cv2.imencode(".png", annotated)
    if not success:
        raise RuntimeError("Unable to encode annotated PNG")
    path = RESULTS_DIR / f"{uuid4()}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return path


def _response_payload(result: PredictionResult, annotated_path: Path) -> dict[str, object]:
    """Translate inference domain objects into the public JSON contract."""
    return {
        "has_defect": result.has_defect,
        "defects": [
            {
                "class_name": defect.class_name,
                "confidence": defect.confidence,
                "coverage": defect.coverage,
            }
            for defect in result.defects
        ],
        "latency_ms": result.latency_ms,
        "backend": result.backend,
        "annotated_image": annotated_path.as_posix(),
    }


app = create_app(Path(os.environ.get("STEEL_INSPECTION_MODEL", "artifacts/checkpoints/best.pt")))
