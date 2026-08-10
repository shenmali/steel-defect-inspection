import torch

from steel_inspection.training.evaluate import dice_iou


def test_dice_iou_returns_one_for_identical_full_masks():
    """Detects metrics that compare raw logits instead of sigmoid probabilities."""
    scores = dice_iou(torch.full((1, 4, 2, 2), 20.0), torch.ones((1, 4, 2, 2)))

    assert scores["dice"] == 1.0
    assert scores["iou"] == 1.0
