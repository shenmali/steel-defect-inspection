import torch
import pytest

from steel_inspection.training.evaluate import dice_iou


def test_dice_iou_returns_one_for_identical_full_masks():
    """Detects metrics that compare raw logits instead of sigmoid probabilities."""
    scores = dice_iou(torch.full((1, 4, 2, 2), 20.0), torch.ones((1, 4, 2, 2)))

    assert scores["dice"] == 1.0
    assert scores["iou"] == 1.0


def test_dice_iou_uses_unsmoothed_partial_overlap_scores():
    logits = torch.tensor([[[[20.0, 20.0, -20.0]]]])
    targets = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    scores = dice_iou(logits, targets)

    assert scores["dice"] == 0.5
    assert scores["iou"] == pytest.approx(1 / 3)


def test_dice_iou_counts_both_empty_masks_as_perfect_match():
    scores = dice_iou(torch.full((1, 1, 2, 2), -20.0), torch.zeros((1, 1, 2, 2)))

    assert scores == {"dice": 1.0, "iou": 1.0}
