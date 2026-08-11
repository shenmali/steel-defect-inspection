"""Select an available inference adapter during application startup."""

from pathlib import Path
from typing import Protocol

import numpy as np

from steel_inspection.inference.pytorch import ModelUnavailableError, PyTorchPredictor
from steel_inspection.inference.tensorrt import TensorRTPredictor
from steel_inspection.inference.types import PredictionResult


class Predictor(Protocol):
    """The interface shared by model-serving backends."""

    backend: str

    def predict(self, image_bgr: np.ndarray) -> PredictionResult:
        """Return segmentation results for a BGR source image."""


def create_predictor(backend: str, model_path: Path, engine_path: Path) -> Predictor:
    """Construct the requested predictor, with startup-only auto fallback."""
    if backend == "pytorch":
        return PyTorchPredictor(model_path)
    if backend == "tensorrt":
        return TensorRTPredictor(engine_path)
    if backend == "auto":
        try:
            return TensorRTPredictor(engine_path)
        except ModelUnavailableError:
            return PyTorchPredictor(model_path)
    raise ModelUnavailableError(f"Unsupported backend: {backend}")
