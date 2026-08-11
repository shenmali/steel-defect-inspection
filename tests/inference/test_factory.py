"""Behavioral tests for selecting an inference backend at startup."""

from pathlib import Path

import pytest

from steel_inspection.inference import factory
from steel_inspection.inference.pytorch import ModelUnavailableError


class _PyTorchPredictor:
    """Lightweight stand-in for the checkpoint-backed adapter."""

    backend = "pytorch"

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path


def _unavailable_predictor(engine_path: Path) -> None:
    raise ModelUnavailableError(f"TensorRT engine is unavailable: {engine_path}")


def test_auto_falls_back_to_pytorch_when_tensorrt_is_unavailable(monkeypatch, tmp_path):
    """Catches auto selection that fails startup instead of using the checkpoint adapter."""
    monkeypatch.setattr(factory, "TensorRTPredictor", _unavailable_predictor)
    monkeypatch.setattr(factory, "PyTorchPredictor", _PyTorchPredictor)

    predictor = factory.create_predictor("auto", tmp_path / "model.pt", tmp_path / "engine")

    assert predictor.backend == "pytorch"


def test_explicit_tensorrt_propagates_unavailability(monkeypatch, tmp_path):
    """Catches explicit TensorRT selection silently changing the requested backend."""
    monkeypatch.setattr(factory, "TensorRTPredictor", _unavailable_predictor)

    with pytest.raises(ModelUnavailableError):
        factory.create_predictor("tensorrt", tmp_path / "model.pt", tmp_path / "engine")
