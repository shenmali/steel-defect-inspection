"""Rendering of model masks over original steel-sheet images."""

import numpy as np

from steel_inspection.config import CLASS_NAMES


CLASS_COLOURS_BGR = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
)


def annotate_mask(image_bgr: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay transparent class-coloured masks without changing image dimensions."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image")
    if masks.shape != (len(CLASS_NAMES), *image_bgr.shape[:2]):
        raise ValueError(f"Mask shape must be {(len(CLASS_NAMES), *image_bgr.shape[:2])}, got {masks.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    annotated = image_bgr.copy()
    for class_index, colour in enumerate(CLASS_COLOURS_BGR):
        selected = masks[class_index].astype(bool)
        if selected.any():
            annotated[selected] = (
                image_bgr[selected].astype(np.float32) * (1.0 - alpha)
                + np.asarray(colour, dtype=np.float32) * alpha
            ).astype(np.uint8)
    return annotated
