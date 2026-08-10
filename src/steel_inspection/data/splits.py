"""Deterministic validation and splitting for Severstal annotations."""

import csv
import random
from pathlib import Path

from steel_inspection.config import CLASS_NAMES

_INPUT_COLUMNS = {"ImageId_ClassId", "EncodedPixels"}
_OUTPUT_COLUMNS = ["image_id", "split", *(f"{name}_rle" for name in CLASS_NAMES)]


def build_split_manifest(
    csv_path: Path, image_dir: Path, output_path: Path, seed: int
) -> dict[str, int]:
    """Validate annotations and write a reproducible 60/20/20 split manifest."""
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    annotations = _read_annotations(csv_path)
    _validate_images(annotations, image_dir)

    image_ids = sorted(annotations)
    random.Random(seed).shuffle(image_ids)
    split_by_image = _split_names(image_ids)
    rows = [_manifest_row(image_id, split_by_image[image_id], annotations[image_id]) for image_id in image_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}


def _read_annotations(csv_path: Path) -> dict[str, dict[str, str | None]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not _INPUT_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("CSV must contain ImageId_ClassId and EncodedPixels columns")

        annotations: dict[str, dict[str, str | None]] = {}
        for row in reader:
            image_and_class = row["ImageId_ClassId"]
            if not image_and_class or "_" not in image_and_class:
                raise ValueError(f"Invalid ImageId_ClassId value: {image_and_class!r}")
            image_id, class_number = image_and_class.rsplit("_", maxsplit=1)
            class_name = f"class_{class_number}"
            if class_name not in CLASS_NAMES:
                raise ValueError(f"Unsupported class in ImageId_ClassId: {image_and_class!r}")
            image_annotations = annotations.setdefault(image_id, {name: None for name in CLASS_NAMES})
            image_annotations[class_name] = row["EncodedPixels"] or None
    return annotations


def _validate_images(annotations: dict[str, dict[str, str | None]], image_dir: Path) -> None:
    missing = [image_id for image_id in annotations if not (image_dir / image_id).is_file()]
    if missing:
        raise FileNotFoundError(f"Referenced images are missing: {', '.join(sorted(missing))}")


def _split_names(image_ids: list[str]) -> dict[str, str]:
    total = len(image_ids)
    train_count = int(total * 0.6)
    val_count = max(1, int(total * 0.2)) if total >= 3 else 0
    test_count = total - train_count - val_count
    if total >= 3 and test_count == 0:
        train_count -= 1
        test_count = 1
    return {
        **{image_id: "train" for image_id in image_ids[:train_count]},
        **{image_id: "val" for image_id in image_ids[train_count : train_count + val_count]},
        **{image_id: "test" for image_id in image_ids[train_count + val_count : train_count + val_count + test_count]},
    }


def _manifest_row(
    image_id: str, split: str, annotations: dict[str, str | None]
) -> dict[str, str | None]:
    return {
        "image_id": image_id,
        "split": split,
        **{f"{name}_rle": annotations[name] for name in CLASS_NAMES},
    }
