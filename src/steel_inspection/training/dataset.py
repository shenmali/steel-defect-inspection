"""PyTorch dataset for split-manifest steel-defect annotations."""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from steel_inspection.config import CLASS_NAMES
from steel_inspection.data.rle import decode_rle


class SteelDefectDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load RGB images and one binary mask per defect class from a manifest."""

    def __init__(
        self,
        manifest_path: Path,
        image_dir: Path,
        split: str,
        image_size: tuple[int, int],
    ) -> None:
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_path = self.image_dir / row["image_id"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        height, width = image_bgr.shape[:2]
        target_height, target_width = self.image_size
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float() / 255.0

        masks = [
            cv2.resize(
                decode_rle(row.get(f"{class_name}_rle"), (height, width)),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
            for class_name in CLASS_NAMES
        ]
        mask = torch.from_numpy(np.ascontiguousarray(np.stack(masks))).float()
        return image, mask
