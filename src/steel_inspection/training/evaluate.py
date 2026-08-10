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

    dice = (2 * intersection + 1.0) / (prediction_total + expected_total + 1.0)
    iou = (intersection + 1.0) / (union + 1.0)
    return {"dice": dice.mean().item(), "iou": iou.mean().item()}
