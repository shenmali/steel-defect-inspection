# Steel Defect Inspection

An individual learning project that demonstrates GPU-accelerated steel-surface defect segmentation. It trains a four-class PyTorch segmentation model, serves predictions through FastAPI, exports ONNX, and benchmarks GPU inference.

## Prerequisites

Use 64-bit PowerShell and Python 3.10, 3.11, or 3.12. NVIDIA GPU inference requires a compatible driver and CUDA 12.1-compatible PyTorch.

## Setup and test

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -v
```

## Data preparation

Download the Steel Defect Detection dataset from Kaggle and place `train.csv` and `train_images/` under `data/`. The dataset is not included in this repository. Create a reproducible split manifest:

```powershell
python scripts/prepare_data.py --csv data/train.csv --images data/train_images --output artifacts/splits.csv --seed 42
```

## Train and evaluate

```powershell
python scripts/train.py --manifest artifacts/splits.csv --images data/train_images --max-epochs 20
python scripts/evaluate.py --manifest artifacts/splits.csv --images data/train_images --checkpoint artifacts/checkpoints/best.pt
```

The evaluation command emits JSON with overall and per-class Dice/IoU. Train and evaluate on held-out data before comparing changes.

## Export and benchmark

```powershell
python scripts/export_onnx.py --checkpoint artifacts/checkpoints/best.pt --output artifacts/model.onnx
python scripts/build_trt.py --onnx artifacts/model.onnx --output artifacts/model-fp16.engine
python scripts/benchmark.py --checkpoint artifacts/checkpoints/best.pt --image data/train_images/<image-file> --runs 100
```

TensorRT engine creation requires the TensorRT Python bindings. The benchmark performs 20 warm-up runs, then measures 100 single-image inferences and writes `artifacts/benchmarks.json`.

## API

Start the API after a trained checkpoint exists:

```powershell
$env:STEEL_INSPECTION_MODEL='artifacts/checkpoints/best.pt'
python -m uvicorn steel_inspection.api.main:app --host 0.0.0.0 --port 8000
```

Send a prediction request:

```powershell
$imagePath = (Get-ChildItem . -Filter *.png -File | Select-Object -First 1).FullName
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -Form @{ image = Get-Item $imagePath }
```

`POST /predict` accepts PNG, JPEG, WEBP, and BMP images up to 10 MiB. It returns detected classes, coverage, inference latency, and the path to an annotated PNG. `GET /health` returns 200 only when the model backend is ready.

## Docker

Build and run with GPU access, mounting local model artifacts read/write for generated annotations:

```powershell
docker build -t steel-defect-inspection .
docker run --rm --gpus all -p 8000:8000 -v ${PWD}/artifacts:/app/artifacts steel-defect-inspection
```

## Limitations

This is a learning and portfolio project. Public benchmark results do not establish readiness for a factory deployment. A real deployment needs validation on the target camera, lighting, material, operating conditions, and defect distribution before any operational use.

The `data/` and `artifacts/` directories contain local datasets and generated
outputs and are intentionally not tracked by Git.
