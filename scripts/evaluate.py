"""Evaluate a saved steel-defect checkpoint on the held-out test split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from steel_inspection.config import CLASS_NAMES, IMAGE_SIZE
from steel_inspection.training.dataset import SteelDefectDataset
from steel_inspection.training.evaluate import evaluate_model, load_checkpoint
from steel_inspection.training.model import create_model


CHECKPOINT_PATH = Path("artifacts/checkpoints/best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Split-manifest CSV path")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing source images")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=None)
    return parser.parse_args()


def limited(dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], limit: int | None) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    """Use the leading examples for a deterministic smoke-run limit."""
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("--limit-test must be positive")
    return Subset(dataset, range(min(limit, len(dataset))))


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_checkpoint(args.checkpoint, device)
    image_size = tuple(payload["image_size"])
    class_names = tuple(payload["class_names"])
    if image_size != IMAGE_SIZE:
        raise ValueError(f"checkpoint image_size must be {IMAGE_SIZE}, got {image_size}")
    if class_names != CLASS_NAMES:
        raise ValueError(f"checkpoint class_names must be {CLASS_NAMES}, got {class_names}")

    test_dataset = limited(SteelDefectDataset(args.manifest, args.images, "test", IMAGE_SIZE), args.limit_test)
    if not len(test_dataset):
        raise ValueError("test split must contain at least one image")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = create_model().to(device)
    model.load_state_dict(payload["model_state_dict"])
    scores = evaluate_model(model, test_loader, device, class_names)
    print(json.dumps(scores, sort_keys=True))


if __name__ == "__main__":
    main()
