"""Stable domain types produced by inference backends."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Defect:
    """A detected defect class and its image-level summary."""

    class_name: str
    confidence: float
    coverage: float


@dataclass(frozen=True)
class PredictionResult:
    """The segmentation output shared by API and future inference backends."""

    has_defect: bool
    defects: list[Defect]
    latency_ms: float
    backend: str
    mask: np.ndarray
