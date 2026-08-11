"""Checkpoint persistence tests."""

import torch
import pytest

from steel_inspection.config import CLASS_NAMES
from steel_inspection.training.evaluate import evaluate_model, load_checkpoint, save_checkpoint


def test_checkpoint_preserves_model_metadata(tmp_path):
    """Catches checkpoints that lose the image shape or class labels."""
    model = torch.nn.Conv2d(3, 4, kernel_size=1)

    save_checkpoint(tmp_path / "best.pt", model, (256, 512), CLASS_NAMES)
    payload = load_checkpoint(tmp_path / "best.pt", device="cpu")

    assert payload["image_size"] == [256, 512]
    assert payload["class_names"] == ["class_1", "class_2", "class_3", "class_4"]


def test_evaluate_model_returns_overall_and_per_class_metrics():
    """Catches evaluation output that omits class-specific held-out scores."""
    logits = torch.tensor([[[[20.0, -20.0]], [[20.0, -20.0]], [[-20.0, -20.0]], [[20.0, 20.0]]]])
    targets = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]], [[1.0, 1.0]]]])

    scores = evaluate_model(torch.nn.Identity(), [(logits, targets)], device="cpu", class_names=CLASS_NAMES)

    assert scores == {
        "overall": {"dice": 0.75, "iou": pytest.approx(0.6)},
        "per_class": {
            "class_1": {"dice": 1.0, "iou": 1.0},
            "class_2": {"dice": 0.0, "iou": 0.0},
            "class_3": {"dice": 1.0, "iou": 1.0},
            "class_4": {"dice": 1.0, "iou": 1.0},
        },
    }
