"""Create a validated deterministic Severstal split manifest."""

import argparse
import csv
from pathlib import Path

from steel_inspection.config import CLASS_NAMES
from steel_inspection.data.splits import build_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Severstal training CSV")
    parser.add_argument("--images", type=Path, required=True, help="Training image directory")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest CSV")
    parser.add_argument("--seed", type=int, required=True, help="Seed for deterministic splits")
    return parser.parse_args()


def class_counts(manifest_path: Path) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for name in CLASS_NAMES:
                if row[f"{name}_rle"]:
                    counts[name] += 1
    return counts


def main() -> None:
    args = parse_args()
    split_counts = build_split_manifest(args.csv, args.images, args.output, args.seed)
    print(f"Split counts: {split_counts}")
    print(f"Class counts: {class_counts(args.output)}")


if __name__ == "__main__":
    main()
