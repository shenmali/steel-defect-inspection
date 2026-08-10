# Steel Surface Defect Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portfolio-ready API that segments defects in steel-sheet images and compares PyTorch GPU inference with TensorRT FP16 on the local RTX 4090.

**Architecture:** A focused Python package separates data preparation, training, model export, inference, API routing, and benchmarking. A U-Net checkpoint is exported to ONNX and optionally compiled to TensorRT; the API uses a backend adapter to provide one consistent prediction response.

**Tech Stack:** Python 3.12, PyTorch, segmentation-models-pytorch, OpenCV, NumPy, FastAPI, Uvicorn, ONNX, TensorRT, pytest, Docker.

## Global Constraints

- Run project commands in 64-bit PowerShell; `[Environment]::Is64BitProcess` must return `True`.
- Use the Severstal Steel Defect Detection dataset locally; never commit images, masks, checkpoints, ONNX files, TensorRT engines, or credentials.
- Fix input dimensions at `256 × 512` for the first release.
- Use four output channels in the fixed order `['class_1', 'class_2', 'class_3', 'class_4']`.
- Report Dice and IoU overall and per class; report single-image latency, FPS, and peak GPU memory for both backends.
- Public-facing documentation must state that this benchmark is not validation for a real production line.

---

## File Structure

```text
steel-defect-inspection/
  src/steel_inspection/
    config.py                 # shared constants and settings
    data/rle.py               # Severstal RLE decoding and validation
    data/splits.py            # deterministic split manifest creation
    training/dataset.py       # PyTorch dataset and preprocessing
    training/model.py         # U-Net model factory
    training/evaluate.py      # Dice and IoU calculations
    inference/types.py        # shared prediction result types
    inference/pytorch.py      # PyTorch backend adapter
    inference/annotate.py     # masks and result-image rendering
    api/main.py               # FastAPI app and POST /predict
  scripts/prepare_data.py     # data validation and split command
  scripts/train.py            # training command
  scripts/export_onnx.py      # checkpoint to ONNX command
  scripts/build_trt.py        # ONNX to TensorRT FP16 engine command
  scripts/benchmark.py        # reproducible backend benchmark command
  tests/                      # unit, API, and smoke tests
  artifacts/                  # ignored checkpoints, ONNX, engines, reports
  data/                       # ignored Kaggle dataset
  Dockerfile
  requirements.txt
  README.md
```

## Task 1: Repository and reproducible Python environment

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `src/steel_inspection/__init__.py`
- Create: `tests/__init__.py`
- Create: `README.md`

**Interfaces:**
- Produces an installable `steel_inspection` package and a `pytest` test command.

- [ ] **Step 1: Write the failing import test**

```python
# tests/test_package.py
def test_package_imports():
    import steel_inspection
    assert steel_inspection.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_package.py -v`

Expected: FAIL because `steel_inspection` is unavailable.

- [ ] **Step 3: Create the minimal package and dependencies**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.build_meta"

[project]
name = "steel-defect-inspection"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
```

```python
# src/steel_inspection/__init__.py
__version__ = "0.1.0"
```

Include `data/`, `artifacts/`, `.venv/`, `__pycache__/`, and `.env` in `.gitignore`. Pin test and runtime dependencies in `requirements.txt` after verifying CUDA-compatible PyTorch installation.

- [ ] **Step 4: Install and run the test**

Run: `python -m pip install -e .; python -m pytest tests/test_package.py -v`

Expected: PASS.

- [ ] **Step 5: Initialize and commit**

Run: `git init; git add .gitignore requirements.txt pyproject.toml src tests README.md; git commit -m "chore: scaffold inspection package"`

## Task 2: RLE decoding and deterministic data manifest

**Files:**
- Create: `src/steel_inspection/config.py`
- Create: `src/steel_inspection/data/__init__.py`
- Create: `src/steel_inspection/data/rle.py`
- Create: `src/steel_inspection/data/splits.py`
- Create: `scripts/prepare_data.py`
- Create: `tests/data/test_rle.py`
- Create: `tests/data/test_splits.py`

**Interfaces:**
- Produces `decode_rle(encoded: str | None, shape: tuple[int, int]) -> np.ndarray`.
- Produces `build_split_manifest(csv_path: Path, image_dir: Path, output_path: Path, seed: int) -> dict[str, int]`.

- [ ] **Step 1: Write failing RLE and split tests**

```python
def test_decode_rle_returns_expected_fortran_mask():
    mask = decode_rle("1 2 5 1", (2, 3))
    assert mask.tolist() == [[1, 0, 1], [1, 0, 0]]

def test_build_split_manifest_is_deterministic(tmp_path):
    counts = build_split_manifest(csv_path, image_dir, tmp_path / "splits.csv", seed=7)
    assert counts == {"train": 2, "val": 1, "test": 1}
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m pytest tests/data/test_rle.py tests/data/test_splits.py -v`

Expected: FAIL because the functions do not exist.

- [ ] **Step 3: Implement RLE decoding and manifest validation**

```python
def decode_rle(encoded: str | None, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if not encoded or encoded != encoded:
        return mask.reshape(shape, order="F")
    starts_and_lengths = np.asarray(encoded.split(), dtype=np.int64).reshape(-1, 2)
    for start, length in starts_and_lengths:
        mask[start - 1:start - 1 + length] = 1
    return mask.reshape(shape, order="F")
```

Require every referenced image to exist. Pivot the CSV into one row per image with four nullable RLE columns, then assign image names to `train`, `val`, and `test` using a seeded shuffle. Write a CSV manifest containing `image_id`, `split`, and `class_1_rle` through `class_4_rle`.

- [ ] **Step 4: Run tests and exercise the command**

Run: `python -m pytest tests/data/test_rle.py tests/data/test_splits.py -v; python scripts/prepare_data.py --csv data/train.csv --images data/train_images --output artifacts/splits.csv --seed 42`

Expected: tests PASS and command prints split counts and class counts.

- [ ] **Step 5: Commit**

Run: `git add src scripts tests; git commit -m "feat: add Severstal data preparation"`

## Task 3: Dataset, model factory, and segmentation metrics

**Files:**
- Create: `src/steel_inspection/training/__init__.py`
- Create: `src/steel_inspection/training/dataset.py`
- Create: `src/steel_inspection/training/model.py`
- Create: `src/steel_inspection/training/evaluate.py`
- Create: `tests/training/test_dataset.py`
- Create: `tests/training/test_evaluate.py`

**Interfaces:**
- Produces `SteelDefectDataset(manifest_path: Path, image_dir: Path, split: str, image_size: tuple[int, int])`.
- Produces `create_model() -> torch.nn.Module` with four logits channels.
- Produces `dice_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]`.

- [ ] **Step 1: Write failing tests**

```python
def test_dataset_returns_rgb_tensor_and_four_masks(fixture_manifest, fixture_images):
    image, mask = SteelDefectDataset(fixture_manifest, fixture_images, "train", (256, 512))[0]
    assert image.shape == (3, 256, 512)
    assert mask.shape == (4, 256, 512)

def test_dice_iou_returns_one_for_identical_full_masks():
    scores = dice_iou(torch.full((1, 4, 2, 2), 20.0), torch.ones((1, 4, 2, 2)))
    assert scores["dice"] == 1.0
    assert scores["iou"] == 1.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/training/test_dataset.py tests/training/test_evaluate.py -v`

Expected: FAIL because the dataset and metrics modules do not exist.

- [ ] **Step 3: Implement the minimum training primitives**

Use OpenCV to read BGR images, convert to RGB, resize images and nearest-neighbour masks to `256 × 512`, and scale images to `float32` in `[0, 1]`. Decode all four class masks per sample. Build U-Net with a ResNet-34 encoder and `classes=4`; return raw logits. Apply `sigmoid` only in metric and inference code.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/training/test_dataset.py tests/training/test_evaluate.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src tests; git commit -m "feat: add segmentation training primitives"`

## Task 4: Training and checkpoint evaluation commands

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/evaluate.py`
- Create: `tests/training/test_checkpoint.py`
- Modify: `src/steel_inspection/training/evaluate.py`

**Interfaces:**
- Consumes `SteelDefectDataset`, `create_model`, and `dice_iou`.
- Produces `artifacts/checkpoints/best.pt` containing `model_state_dict`, `image_size`, and `class_names`.

- [ ] **Step 1: Write the failing checkpoint round-trip test**

```python
def test_checkpoint_preserves_model_metadata(tmp_path):
    save_checkpoint(tmp_path / "best.pt", model, (256, 512), CLASS_NAMES)
    payload = load_checkpoint(tmp_path / "best.pt", device="cpu")
    assert payload["image_size"] == [256, 512]
    assert payload["class_names"] == ["class_1", "class_2", "class_3", "class_4"]
```

- [ ] **Step 2: Run the test to verify failure**

Run: `python -m pytest tests/training/test_checkpoint.py -v`

Expected: FAIL because checkpoint helpers do not exist.

- [ ] **Step 3: Implement training and evaluation**

Train with `BCEWithLogitsLoss`, AdamW, mixed precision on CUDA, fixed seed, validation Dice selection, and a `--max-epochs` argument. Save only the best validation checkpoint. The evaluation command loads the checkpoint and prints JSON containing overall and per-class Dice/IoU for the held-out test split.

- [ ] **Step 4: Run test and a smoke training job**

Run: `python -m pytest tests/training/test_checkpoint.py -v; python scripts/train.py --manifest artifacts/splits.csv --images data/train_images --max-epochs 1 --limit-train 32 --limit-val 16`

Expected: test PASS; training completes one epoch and writes `artifacts/checkpoints/best.pt`.

- [ ] **Step 5: Commit**

Run: `git add src scripts tests; git commit -m "feat: add training and evaluation commands"`

## Task 5: PyTorch inference, annotation, and API contract

**Files:**
- Create: `src/steel_inspection/inference/__init__.py`
- Create: `src/steel_inspection/inference/types.py`
- Create: `src/steel_inspection/inference/pytorch.py`
- Create: `src/steel_inspection/inference/annotate.py`
- Create: `src/steel_inspection/api/__init__.py`
- Create: `src/steel_inspection/api/main.py`
- Create: `tests/api/test_predict.py`

**Interfaces:**
- Produces `PredictionResult(has_defect: bool, defects: list[Defect], latency_ms: float, backend: str, mask: np.ndarray)`.
- Produces `create_app(model_path: Path, backend: str = "pytorch") -> FastAPI`.
- Produces `POST /predict` multipart endpoint with field name `image`.

- [ ] **Step 1: Write failing API tests**

```python
def test_predict_returns_contract(client, png_bytes):
    response = client.post("/predict", files={"image": ("sheet.png", png_bytes, "image/png")})
    assert response.status_code == 200
    assert set(response.json()) >= {"has_defect", "defects", "latency_ms", "backend", "annotated_image"}

def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"image": ("notes.txt", b"no", "text/plain")})
    assert response.status_code == 415
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/api/test_predict.py -v`

Expected: FAIL because the application factory does not exist.

- [ ] **Step 3: Implement the PyTorch adapter and endpoint**

Load checkpoint metadata once at app startup. Decode uploaded bytes with OpenCV; reject unreadable and non-image inputs. Run preprocessing at the fixed model dimensions, threshold sigmoid logits at `0.5`, resize masks back to the source dimensions, and render class-coloured transparent overlays. Store annotated PNGs under `artifacts/results/` with UUID filenames. Return 503 at startup if the model path is unavailable.

- [ ] **Step 4: Run tests and a manual API smoke test**

Run: `python -m pytest tests/api/test_predict.py -v; uvicorn steel_inspection.api.main:app --port 8000`

In a second terminal run: `Invoke-RestMethod -Uri http://127.0.0.1:8000/docs`

Expected: tests PASS and Swagger UI endpoint is reachable.

- [ ] **Step 5: Commit**

Run: `git add src tests; git commit -m "feat: add PyTorch inference API"`

## Task 6: ONNX export, TensorRT backend, and benchmark report

**Files:**
- Create: `scripts/export_onnx.py`
- Create: `scripts/build_trt.py`
- Create: `src/steel_inspection/inference/tensorrt.py`
- Create: `scripts/benchmark.py`
- Create: `tests/inference/test_onnx_export.py`
- Create: `tests/inference/test_benchmark.py`

**Interfaces:**
- Produces `artifacts/model.onnx` with input name `image` and output name `logits`.
- Produces `artifacts/model-fp16.engine` when TensorRT is available.
- Produces `artifacts/benchmarks.json` with `backend`, `mean_latency_ms`, `p95_latency_ms`, `fps`, and `peak_gpu_memory_mb`.

- [ ] **Step 1: Write failing export and report-schema tests**

```python
def test_exported_onnx_has_expected_io_names(tmp_path, checkpoint):
    export_onnx(checkpoint, tmp_path / "model.onnx")
    model = onnx.load(tmp_path / "model.onnx")
    assert model.graph.input[0].name == "image"
    assert model.graph.output[0].name == "logits"

def test_benchmark_report_has_required_metrics():
    assert set(make_report("pytorch", [1.0, 2.0], 42.0)) == {"backend", "mean_latency_ms", "p95_latency_ms", "fps", "peak_gpu_memory_mb"}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/inference/test_onnx_export.py tests/inference/test_benchmark.py -v`

Expected: FAIL because export and report functions do not exist.

- [ ] **Step 3: Implement export, engine creation, and benchmark logic**

Export one `(1, 3, 256, 512)` input with ONNX opset 17 and verify it using `onnx.checker.check_model`. Build the engine with explicit batch and FP16 enabled. Benchmark exactly 20 warm-up inferences followed by 100 timed inferences, synchronising CUDA before and after each measurement. Capture PyTorch peak allocated memory or the process GPU memory measurement for TensorRT. Write a single JSON report containing both backend records and environment versions.

- [ ] **Step 4: Run tests and the GPU benchmark**

Run in 64-bit PowerShell:

```powershell
python -m pytest tests/inference/test_onnx_export.py tests/inference/test_benchmark.py -v
python scripts/export_onnx.py --checkpoint artifacts/checkpoints/best.pt --output artifacts/model.onnx
python scripts/build_trt.py --onnx artifacts/model.onnx --output artifacts/model-fp16.engine
$sampleImage = (Get-ChildItem data/train_images -File | Select-Object -First 1).FullName
python scripts/benchmark.py --image $sampleImage --runs 100
```

Expected: tests PASS, ONNX validation succeeds, TensorRT engine is produced, and the benchmark JSON contains both backends.

- [ ] **Step 5: Commit**

Run: `git add src scripts tests; git commit -m "feat: add TensorRT export and benchmarks"`

## Task 7: Docker, documentation, and final verification

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Modify: `README.md`
- Create: `tests/test_e2e_smoke.py`

**Interfaces:**
- Consumes a prepared checkpoint and, optionally, a TensorRT engine mounted beneath `/app/artifacts`.
- Produces a documented `POST /predict` service on port `8000`.

- [ ] **Step 1: Write the failing end-to-end smoke test**

```python
def test_service_predicts_fixture_image(app_client, fixture_png):
    response = app_client.post("/predict", files={"image": ("fixture.png", fixture_png, "image/png")})
    assert response.status_code == 200
    assert isinstance(response.json()["has_defect"], bool)
```

- [ ] **Step 2: Run the smoke test to verify current state**

Run: `python -m pytest tests/test_e2e_smoke.py -v`

Expected: PASS after Tasks 1–6; if it fails, fix the failing interface before adding Docker files.

- [ ] **Step 3: Write packaging and portfolio documentation**

Use a CUDA-compatible base image. The Dockerfile installs pinned dependencies, exposes port `8000`, and runs Uvicorn. README sections must include: problem statement, dataset source and license note, architecture, 64-bit PowerShell prerequisite, setup commands, train/evaluate/export/benchmark commands, API example, metrics table template, limitations, and production validation warning.

- [ ] **Step 4: Run full verification**

Run: `python -m pytest -v; docker build -t steel-defect-inspection .; docker run --rm --gpus all -p 8000:8000 -v ${PWD}/artifacts:/app/artifacts steel-defect-inspection`

Expected: all tests PASS, Docker image builds, and `http://127.0.0.1:8000/docs` is reachable.

- [ ] **Step 5: Commit**

Run: `git add Dockerfile .dockerignore README.md tests; git commit -m "docs: package inspection service"`

## Plan Self-Review

- Spec coverage: data preparation (Task 2), model training and metrics (Tasks 3–4), ONNX/TensorRT and performance comparison (Task 6), API and error handling (Task 5), tests (Tasks 1–7), Docker and portfolio documentation (Task 7).
- Scope: live camera/PLC, cloud deployment, and production-line validation are explicitly excluded.
- Type consistency: `SteelDefectDataset`, `create_model`, `dice_iou`, checkpoint metadata, `PredictionResult`, and `POST /predict` are defined before their consumers.
- Placeholder scan: no deferred implementation markers or unresolved command values are used.
