"""End-to-end coverage for the training and evaluation commands."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from steel_inspection.config import CLASS_NAMES


def _write_split_fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest_path = tmp_path / "splits.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "split", *(f"{name}_rle" for name in CLASS_NAMES)],
        )
        writer.writeheader()
        for filename, split in (("train.png", "train"), ("val.png", "val"), ("test.png", "test")):
            assert cv2.imwrite(str(image_dir / filename), np.zeros((4, 6, 3), dtype=np.uint8))
            writer.writerow({"image_id": filename, "split": split, **{f"{name}_rle": "" for name in CLASS_NAMES}})
    return manifest_path, image_dir


def test_training_and_evaluation_commands_create_checkpoint_and_print_metrics(tmp_path):
    """Catches CLI wiring that skips checkpoint persistence or class metric JSON output."""
    manifest_path, image_dir = _write_split_fixture(tmp_path)
    checkpoint_path = tmp_path / "best.pt"
    repository_root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--manifest",
            str(manifest_path),
            "--images",
            str(image_dir),
            "--max-epochs",
            "1",
            "--batch-size",
            "1",
            "--checkpoint",
            str(checkpoint_path),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert checkpoint_path.is_file()

    evaluation = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--manifest",
            str(manifest_path),
            "--images",
            str(image_dir),
            "--batch-size",
            "1",
            "--checkpoint",
            str(checkpoint_path),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    scores = json.loads(evaluation.stdout)

    assert set(scores) == {"overall", "per_class"}
    assert set(scores["overall"]) == {"dice", "iou"}
    assert set(scores["per_class"]) == set(CLASS_NAMES)
    for class_scores in scores["per_class"].values():
        assert set(class_scores) == {"dice", "iou"}
