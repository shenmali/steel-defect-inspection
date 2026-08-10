import csv

import pytest

from steel_inspection.data.splits import build_split_manifest


def _write_fixture_csv(path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageId_ClassId", "EncodedPixels"])
        writer.writeheader()
        writer.writerows(
            [
                {"ImageId_ClassId": "sheet_a.jpg_1", "EncodedPixels": "1 2"},
                {"ImageId_ClassId": "sheet_a.jpg_3", "EncodedPixels": ""},
                {"ImageId_ClassId": "sheet_b.jpg_2", "EncodedPixels": "3 1"},
                {"ImageId_ClassId": "sheet_c.jpg_4", "EncodedPixels": "5 2"},
                {"ImageId_ClassId": "sheet_d.jpg_1", "EncodedPixels": ""},
            ]
        )


def test_build_split_manifest_is_deterministic(tmp_path):
    csv_path = tmp_path / "train.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_fixture_csv(csv_path)
    for name in ("sheet_a.jpg", "sheet_b.jpg", "sheet_c.jpg", "sheet_d.jpg"):
        (image_dir / name).touch()

    output_path = tmp_path / "splits.csv"
    counts = build_split_manifest(csv_path, image_dir, output_path, seed=7)

    assert counts == {"train": 2, "val": 1, "test": 1}

    other_output_path = tmp_path / "other-splits.csv"
    build_split_manifest(csv_path, image_dir, other_output_path, seed=7)
    assert output_path.read_text(encoding="utf-8") == other_output_path.read_text(encoding="utf-8")

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "image_id",
        "split",
        "class_1_rle",
        "class_2_rle",
        "class_3_rle",
        "class_4_rle",
    ]
    assert next(row for row in rows if row["image_id"] == "sheet_a.jpg")["class_3_rle"] == ""


def test_build_split_manifest_rejects_missing_referenced_images(tmp_path):
    csv_path = tmp_path / "train.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_fixture_csv(csv_path)

    with pytest.raises(FileNotFoundError, match="sheet_a.jpg"):
        build_split_manifest(csv_path, image_dir, tmp_path / "splits.csv", seed=7)
