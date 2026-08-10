"""Segmentation evaluation metrics."""

import torch


def dice_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Return mean Dice and IoU after thresholding sigmoid probabilities."""
    predictions = torch.sigmoid(logits) >= threshold
    expected = targets >= threshold
    dimensions = tuple(range(1, predictions.ndim))
    intersection = (predictions & expected).sum(dim=dimensions, dtype=torch.float32)
    prediction_total = predictions.sum(dim=dimensions, dtype=torch.float32)
    expected_total = expected.sum(dim=dimensions, dtype=torch.float32)
    union = (predictions | expected).sum(dim=dimensions, dtype=torch.float32)

    both_empty = (prediction_total == 0) & (expected_total == 0)
    dice = torch.where(
        both_empty,
        torch.ones_like(intersection),
        (2 * intersection) / (prediction_total + expected_total),
    )
    iou = torch.where(
        both_empty,
        torch.ones_like(intersection),
        intersection / union,
    )
    return {"dice": dice.mean().item(), "iou": iou.mean().item()}
