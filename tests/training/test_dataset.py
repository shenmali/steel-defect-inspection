import csv

import cv2
import numpy as np
import torch

from steel_inspection.training.dataset import SteelDefectDataset


def _write_manifest(path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "split",
                "class_1_rle",
                "class_2_rle",
                "class_3_rle",
                "class_4_rle",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "sheet.png",
                "split": "train",
                "class_1_rle": "1 1",
                "class_2_rle": "",
                "class_3_rle": "",
                "class_4_rle": "",
            }
        )


def test_dataset_returns_rgb_tensor_and_four_masks(tmp_path):
    """Detects changes that return BGR/non-normalized images or omit a class mask."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    source_image = np.zeros((2, 3, 3), dtype=np.uint8)
    source_image[:, :, 2] = 255
    assert cv2.imwrite(str(image_dir / "sheet.png"), source_image)
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path)

    image, mask = SteelDefectDataset(manifest_path, image_dir, "train", (256, 512))[0]

    assert image.shape == (3, 256, 512)
    assert image.dtype == torch.float32
    assert image[0, 0, 0].item() == 1.0
    assert image[2, 0, 0].item() == 0.0
    assert mask.shape == (4, 256, 512)
    assert mask.dtype == torch.float32
    assert mask[0, 0, 0].item() == 1.0
    assert mask[1:].sum().item() == 0.0
