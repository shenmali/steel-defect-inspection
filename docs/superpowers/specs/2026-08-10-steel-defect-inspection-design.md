# Steel Surface Defect Inspection — Design

## Goal

Build a portfolio-ready industrial visual-inspection service that detects and segments defects in steel-sheet images. The project demonstrates computer vision, GPU inference optimisation, API design, reproducibility, and performance measurement relevant to industrial AI roles.

## Scope

### Included

- Train a PyTorch U-Net segmentation model on the Severstal Steel Defect Detection dataset.
- Predict whether a defect exists, its class, and a pixel-level mask.
- Export the selected model to ONNX and build a TensorRT FP16 engine.
- Benchmark PyTorch GPU inference against TensorRT FP16 on the local NVIDIA RTX 4090.
- Provide a FastAPI image-inference endpoint that returns JSON metrics and an annotated result image.
- Package the service with Docker and document setup, evaluation, and benchmarks in English.

### Excluded from the first release

- Live camera, PLC, MES, or factory-floor integration.
- Cloud hosting and user authentication.
- A claim that the public benchmark is validated for a production factory line.

## Data

Use the Severstal Steel Defect Detection dataset. It contains steel-sheet images and run-length-encoded masks for four defect classes. Data preparation will decode masks, validate image-mask pairs, create reproducible train/validation/test splits, and record split counts and class balance.

The README must state that public benchmark data is representative, not evidence of production readiness. A deployment would require validation and likely retraining using the target camera, illumination, steel grade, and defect distribution.

## Architecture

```text
Dataset → validation/splits → PyTorch training/evaluation → best checkpoint
  → ONNX export → TensorRT FP16 engine → FastAPI inference endpoint
  → annotated image + JSON response + benchmark report
```

Components have single responsibilities:

- `data`: download instructions, RLE-mask decoding, validation, and split generation.
- `training`: datasets, augmentation, U-Net training, checkpoints, and evaluation.
- `export`: ONNX export and TensorRT engine generation.
- `inference`: image preprocessing, model adapters, thresholding, and annotation.
- `api`: FastAPI request validation and response composition.
- `benchmarks`: warm-up-aware latency, FPS, and GPU-memory measurements.

## API contract

`POST /predict` accepts one supported image upload. It returns:

- `has_defect`: boolean;
- `defects`: detected classes, confidence-like scores, and mask coverage;
- `latency_ms` and inference backend;
- a reference to the generated annotated image.

Unsupported, corrupt, or oversized images receive a clear 4xx response. The service fails at startup with an actionable error if its model artifact or requested inference backend is unavailable.

## Evaluation and benchmarks

Report Dice and IoU overall and per defect class, together with precision/recall where meaningful. Benchmark single-image latency, throughput in FPS, and GPU memory for PyTorch GPU and TensorRT FP16 under the same preprocessing and input size. Include environment versions, warm-up policy, batch size, and measurement repetitions so results are reproducible.

## Testing

- Unit tests for RLE-mask decoding and image-mask pairing.
- Tests for dataset split reproducibility and API response schema.
- Tests for invalid image uploads and missing model artifacts.
- A small end-to-end inference smoke test using a fixture image.

## Portfolio deliverables

- Public GitHub repository with an English README, architecture overview, commands, metrics, and limitations.
- Short demo video/GIF showing upload, predicted mask, and response.
- Benchmark table comparing PyTorch GPU and TensorRT FP16.
- Docker-based local run instructions.
