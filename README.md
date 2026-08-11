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
python scripts/build_trt.py --onnx artifacts/model.onnx --output artifacts/model.engine
python scripts/benchmark.py --checkpoint artifacts/checkpoints/best.pt --image data/train_images/<image-file> --engine artifacts/model.engine --engine artifacts/model-fp16.engine --runs 100
```

TensorRT engine creation requires the TensorRT Python bindings. The benchmark performs 20 warm-up runs, then measures 100 GPU-only single-image inferences for PyTorch and every supplied TensorRT engine. It writes a comparable JSON report to `artifacts/benchmarks.json`.

TensorRT 11 engines use the precision encoded in their ONNX graph. Build `model-fp16.engine` from a ModelOpt-converted `model-fp16.onnx`; keep ModelOpt in a separate virtual environment because its current PyTorch requirement may differ from this project's training environment.

## API

Start the API after a trained checkpoint exists:

```powershell
$env:STEEL_INSPECTION_BACKEND='auto' # auto, tensorrt, or pytorch
$env:STEEL_INSPECTION_MODEL='artifacts/checkpoints/best.pt'
$env:STEEL_INSPECTION_ENGINE='artifacts/model-fp16.engine'
$env:STEEL_INSPECTION_SAVE_ANNOTATIONS='false'
python -m uvicorn steel_inspection.api.main:app --host 0.0.0.0 --port 8000
```

Send a prediction request:

```powershell
$imagePath = (Get-ChildItem . -Filter *.png -File | Select-Object -First 1).FullName
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -Form @{ image = Get-Item $imagePath }
```

`POST /predict` accepts PNG, JPEG, WEBP, and BMP images up to 10 MiB. It returns detected classes, coverage, inference latency, and `annotated_image`, which is `null` by default. `annotated_image` contains a saved PNG path only when `STEEL_INSPECTION_SAVE_ANNOTATIONS=true`. `GET /health` returns 200 only when the model backend is ready. Configuration is read once when the process imports the API module: `auto` tries the configured TensorRT engine and falls back to PyTorch at startup only; `tensorrt` never falls back and reports an unready 503 when unavailable; `pytorch` loads only the checkpoint.

## Docker

The base image is a PyTorch-serving image; it does **not** install TensorRT. It is suitable for the PyTorch-only internal deployment below. Do not expose this service directly to the internet: keep it on an internal factory network and restrict ingress with the host firewall or a reverse proxy.

Build the image:

```powershell
docker build -t steel-defect-inspection .
```

Run PyTorch-only with a read-only checkpoint mount and annotation persistence disabled:

```powershell
docker run --rm --gpus all -p 127.0.0.1:8000:8000 `
  -e STEEL_INSPECTION_BACKEND=pytorch `
  -e STEEL_INSPECTION_MODEL=/models/best.pt `
  -e STEEL_INSPECTION_SAVE_ANNOTATIONS=false `
  -v ${PWD}/artifacts/checkpoints/best.pt:/models/best.pt:ro `
  steel-defect-inspection
```

For GPU TensorRT `auto`, first build an **internal derivative image** that includes the TensorRT runtime matching the engine and GPU driver, then smoke-test it with a real engine. Only then run it with read-only model and engine mounts:

```powershell
docker run --rm --gpus all -p 127.0.0.1:8000:8000 `
  -e STEEL_INSPECTION_BACKEND=auto `
  -e STEEL_INSPECTION_MODEL=/models/best.pt `
  -e STEEL_INSPECTION_ENGINE=/models/model-fp16.engine `
  -e STEEL_INSPECTION_SAVE_ANNOTATIONS=false `
  -v ${PWD}/artifacts/checkpoints/best.pt:/models/best.pt:ro `
  -v ${PWD}/artifacts/model-fp16.engine:/models/model-fp16.engine:ro `
  steel-defect-inspection-tensorrt
```

`auto` falls back to PyTorch during startup if TensorRT cannot load. Use `STEEL_INSPECTION_BACKEND=tensorrt` when the deployment must fail readiness instead of accepting that fallback.

## Local validation evidence

On 2026-08-11, local GPU validation used an NVIDIA GeForce RTX 4090 (driver 591.86), PyTorch 2.5.1+cu121 with CUDA available, and TensorRT 11.2.1.2. Benchmarking a real local image against `artifacts/checkpoints/best.pt` and `artifacts/model-fp16.engine` reported PyTorch mean latency 4.998 ms (200.07 FPS; p95 7.051 ms) and FP16 TensorRT mean latency 1.004 ms (995.57 FPS; p95 1.267 ms).

A direct `TensorRTPredictor` and PyTorch comparison using the same real image, checkpoint, and engine produced masks with equal shapes and agreement of 1.0. The TensorRT adapter, backend-selection, and API-contract checks also passed with `python -m pytest tests/inference/test_tensorrt.py tests/inference/test_factory.py tests/api/test_predict.py -q` (24 passed).

On the same host, the PyTorch base image was rebuilt successfully and a GPU container smoke test passed: `GET /health` returned `{"status":"ok"}` and `POST /predict` on a real Severstal image returned a PyTorch result with `annotated_image: null`. The base image still does not validate TensorRT container serving; that requires the separately built derivative image described above and a real-engine smoke test.

## Limitations

This is a learning and portfolio project. Public benchmark results do not establish readiness for a factory deployment. A real deployment needs validation on the target camera, lighting, material, operating conditions, and defect distribution before any operational use.

The `data/` and `artifacts/` directories contain local datasets and generated
outputs and are intentionally not tracked by Git.
