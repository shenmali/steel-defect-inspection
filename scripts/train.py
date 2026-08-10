"""Train the steel-defect segmentation model and save its best checkpoint."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from steel_inspection.config import CLASS_NAMES, IMAGE_SIZE
from steel_inspection.training.dataset import SteelDefectDataset
from steel_inspection.training.evaluate import evaluate_model, save_checkpoint
from steel_inspection.training.model import create_model


CHECKPOINT_PATH = Path("artifacts/checkpoints/best.pt")
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Split-manifest CSV path")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing source images")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    return parser.parse_args()


def set_seed() -> None:
    """Make the train/validation split traversal and model initialization repeatable."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def limited(dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], limit: int | None) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    """Use the leading examples for a deterministic smoke-run limit."""
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("dataset limits must be positive")
    return Subset(dataset, range(min(limit, len(dataset))))


def main() -> None:
    args = parse_args()
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = limited(SteelDefectDataset(args.manifest, args.images, "train", IMAGE_SIZE), args.limit_train)
    val_dataset = limited(SteelDefectDataset(args.manifest, args.images, "val", IMAGE_SIZE), args.limit_val)
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("train and validation splits must each contain at least one image")

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = create_model().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_dice = float("-inf")

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
                loss = loss_function(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size

        validation = evaluate_model(model, val_loader, device, CLASS_NAMES)
        validation_dice = validation["overall"]["dice"]
        if validation_dice > best_dice:
            best_dice = validation_dice
            save_checkpoint(args.checkpoint, model, IMAGE_SIZE, CLASS_NAMES)
        print(
            f"epoch={epoch} train_loss={total_loss / sample_count:.6f} "
            f"val_dice={validation_dice:.6f} best_val_dice={best_dice:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
