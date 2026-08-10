"""Segmentation evaluation metrics and checkpoint persistence."""

from pathlib import Path
from typing import Iterable, Sequence

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    image_size: tuple[int, int],
    class_names: Sequence[str],
) -> None:
    """Persist model weights and the metadata needed to recreate them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": list(image_size),
            "class_names": list(class_names),
        },
        path,
    )


def load_checkpoint(path: Path, device: str | torch.device) -> dict[str, object]:
    """Load a checkpoint onto the requested device."""
    return torch.load(Path(path), map_location=device, weights_only=True)


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: str | torch.device,
    class_names: Sequence[str],
) -> dict[str, dict[str, float] | dict[str, dict[str, float]]]:
    """Return sample-weighted overall and per-class segmentation scores."""
    model.eval()
    totals = {"dice": 0.0, "iou": 0.0}
    class_totals = {name: {"dice": 0.0, "iou": 0.0} for name in class_names}
    sample_count = 0

    for images, targets in batches:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        batch_size = targets.shape[0]
        overall_scores = dice_iou(logits, targets)
        for metric_name, value in overall_scores.items():
            totals[metric_name] += value * batch_size
        for class_index, class_name in enumerate(class_names):
            class_scores = dice_iou(logits[:, class_index : class_index + 1], targets[:, class_index : class_index + 1])
            for metric_name, value in class_scores.items():
                class_totals[class_name][metric_name] += value * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    return {
        "overall": {metric_name: value / sample_count for metric_name, value in totals.items()},
        "per_class": {
            class_name: {metric_name: value / sample_count for metric_name, value in scores.items()}
            for class_name, scores in class_totals.items()
        },
    }


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
