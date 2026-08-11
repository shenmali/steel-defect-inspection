"""FastAPI application exposing one image-segmentation endpoint."""

from contextlib import asynccontextmanager
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from steel_inspection.api.storage import AnnotationStorageError, UploadTooLargeError, read_limited_upload, store_annotation
from steel_inspection.inference.pytorch import ModelUnavailableError, PyTorchPredictor
from steel_inspection.inference.types import PredictionResult


SUPPORTED_IMAGE_TYPES = frozenset({"image/bmp", "image/jpeg", "image/png", "image/webp"})
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
RESULTS_DIR = Path("artifacts/results")


def create_app(model_path: Path, backend: str = "pytorch", save_annotations: bool = False) -> FastAPI:
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
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_UPLOAD_BYTES)
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
        """Validate one image upload and segment it."""
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

        try:
            contents = await read_limited_upload(image, MAX_UPLOAD_BYTES)
        except UploadTooLargeError:
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="Image exceeds 10 MiB") from None
        if not contents:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Image upload is empty")
        decoded = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Image data is unreadable")

        result = app.state.predictor.predict(decoded)
        try:
            annotated_path = store_annotation(
                decoded,
                result,
                results_dir=RESULTS_DIR,
                enabled=save_annotations,
            )
        except AnnotationStorageError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to store annotated image",
            ) from None
        return _response_payload(result, annotated_path)

    return app


def _response_payload(result: PredictionResult, annotated_path: Path | None) -> dict[str, object]:
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
        "annotated_image": annotated_path.as_posix() if annotated_path is not None else None,
    }


app = create_app(Path(os.environ.get("STEEL_INSPECTION_MODEL", "artifacts/checkpoints/best.pt")))
