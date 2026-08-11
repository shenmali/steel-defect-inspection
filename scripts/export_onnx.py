"""Export a trained checkpoint to a validated ONNX model."""
import argparse
from pathlib import Path

import onnx
import torch

from steel_inspection.config import IMAGE_SIZE
from steel_inspection.training.evaluate import load_checkpoint
from steel_inspection.training.model import create_model


def export_onnx(checkpoint_path: Path, output_path: Path) -> None:
    model = create_model()
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sample = torch.zeros((1, 3, *IMAGE_SIZE), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, sample, output_path, input_names=["image"], output_names=["logits"], opset_version=17)
    onnx.checker.check_model(onnx.load(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output)
    print(args.output)
