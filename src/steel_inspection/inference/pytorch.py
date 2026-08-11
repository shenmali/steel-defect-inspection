"""PyTorch model adapter with fixed-geometry image preprocessing."""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from steel_inspection.config import CLASS_NAMES, IMAGE_SIZE
from steel_inspection.inference.types import Defect, PredictionResult
from steel_inspection.training.evaluate import load_checkpoint
from steel_inspection.training.model import create_model


class ModelUnavailableError(RuntimeError):
    """Raised when a requested model artifact cannot be used for inference."""


class PyTorchPredictor:
    """Load one trained checkpoint and predict four binary defect masks."""

    backend = "pytorch"

    def __init__(self, model_path: Path, device: str | torch.device | None = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise ModelUnavailableError(f"Model checkpoint is unavailable: {self.model_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            checkpoint = load_checkpoint(self.model_path, self.device)
            self._validate_metadata(checkpoint)
            model = create_model()
            model.load_state_dict(checkpoint["model_state_dict"])
            self.model = model.to(self.device).eval()
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ModelUnavailableError(f"Unable to load model checkpoint {self.model_path}: {error}") from error

    @staticmethod
    def _validate_metadata(checkpoint: dict[str, object]) -> None:
        """Reject checkpoints trained with a geometry or class-order mismatch."""
        image_size = checkpoint.get("image_size")
        class_names = checkpoint.get("class_names")
        if image_size != list(IMAGE_SIZE):
            raise ValueError(f"Checkpoint image_size must be {list(IMAGE_SIZE)}, got {image_size}")
        if class_names != list(CLASS_NAMES):
            raise ValueError(f"Checkpoint class_names must be {list(CLASS_NAMES)}, got {class_names}")

    def predict(self, image_bgr: np.ndarray) -> PredictionResult:
        """Segment a BGR source image and return masks at its original dimensions."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image")
        source_height, source_width = image_bgr.shape[:2]
        image = self._preprocess(image_bgr)

        started_at = perf_counter()
        with torch.inference_mode():
            logits = self.model(image)
            probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
        latency_ms = (perf_counter() - started_at) * 1000

        if probabilities.shape != (len(CLASS_NAMES), *IMAGE_SIZE):
            raise RuntimeError(
                f"Model output must be {(len(CLASS_NAMES), *IMAGE_SIZE)}, got {tuple(probabilities.shape)}"
            )

        masks = probabilities >= 0.5
        source_masks = np.stack(
            [
                cv2.resize(mask.astype(np.uint8), (source_width, source_height), interpolation=cv2.INTER_NEAREST)
                for mask in masks
            ]
        )
        defects = [
            Defect(
                class_name=class_name,
                confidence=float(probabilities[index].max()),
                coverage=float(source_masks[index].mean()),
            )
            for index, class_name in enumerate(CLASS_NAMES)
            if source_masks[index].any()
        ]
        return PredictionResult(
            has_defect=bool(defects),
            defects=defects,
            latency_ms=latency_ms,
            backend=self.backend,
            mask=source_masks,
        )

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR input to a normalized RGB tensor at the fixed model size."""
        target_height, target_width = IMAGE_SIZE
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        array = np.ascontiguousarray(resized.transpose(2, 0, 1))
        return torch.from_numpy(array).float().div(255.0).unsqueeze(0).to(self.device)
