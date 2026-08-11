# Production Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the steel-inspection model with safe automatic TensorRT/PyTorch selection and bounded request/storage resources.

**Architecture:** Add a TensorRT predictor that returns the existing `PredictionResult`, plus a backend factory used by FastAPI startup. Keep API transport safeguards and result retention in focused helpers so they can be tested without CUDA. Docker documentation distinguishes the base PyTorch image from an internal GPU/TensorRT deployment.

**Tech Stack:** Python 3.12, FastAPI, PyTorch CUDA 12.1, TensorRT 11, OpenCV, pytest.

## Global Constraints

- `STEEL_INSPECTION_BACKEND` values are exactly `auto`, `tensorrt`, and `pytorch`; default is `auto`.
- Explicit `tensorrt` never falls back; `auto` falls back only at startup when TensorRT cannot load.
- Uploads are bounded at 10 MiB before retaining the complete body.
- Annotation persistence is opt-in, 24-hour TTL, 1 GiB capacity, and never deletes outside `artifacts/results`.
- Retain the existing `/predict` response schema.

---

### Task 1: Add a TensorRT inference adapter and backend factory

**Files:**
- Create: `src/steel_inspection/inference/tensorrt.py`
- Create: `src/steel_inspection/inference/factory.py`
- Modify: `src/steel_inspection/inference/__init__.py`
- Test: `tests/inference/test_factory.py`

**Interfaces:**
- Consumes: `PredictionResult`, `Defect`, `CLASS_NAMES`, `IMAGE_SIZE`, and a serialized fixed-shape engine.
- Produces: `TensorRTPredictor(engine_path: Path)` with `predict(image_bgr: np.ndarray) -> PredictionResult`; `create_predictor(backend: str, model_path: Path, engine_path: Path) -> Predictor`.

- [ ] **Step 1: Write failing factory tests**

```python
def test_auto_falls_back_to_pytorch_when_tensorrt_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(factory, "TensorRTPredictor", _unavailable_predictor)
    predictor = factory.create_predictor("auto", tmp_path / "model.pt", tmp_path / "engine")
    assert predictor.backend == "pytorch"

def test_explicit_tensorrt_propagates_unavailability(monkeypatch, tmp_path):
    monkeypatch.setattr(factory, "TensorRTPredictor", _unavailable_predictor)
    with pytest.raises(ModelUnavailableError):
        factory.create_predictor("tensorrt", tmp_path / "model.pt", tmp_path / "engine")
```

- [ ] **Step 2: Run focused tests and observe failure**

Run: `python -m pytest tests/inference/test_factory.py -q`

Expected: import failure because `factory` does not exist.

- [ ] **Step 3: Implement minimal adapter and factory**

```python
def create_predictor(backend: str, model_path: Path, engine_path: Path):
    if backend == "pytorch":
        return PyTorchPredictor(model_path)
    if backend == "tensorrt":
        return TensorRTPredictor(engine_path)
    if backend == "auto":
        try:
            return TensorRTPredictor(engine_path)
        except ModelUnavailableError:
            return PyTorchPredictor(model_path)
    raise ModelUnavailableError(f"Unsupported backend: {backend}")
```

`TensorRTPredictor` must deserialize the engine once, bind persistent `torch.float32` CUDA input/output tensors by name, execute with `execute_async_v3`, apply sigmoid, resize masks to the source image, and construct `PredictionResult(backend="tensorrt")`.

- [ ] **Step 4: Run factory and existing inference tests**

Run: `python -m pytest tests/inference -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/steel_inspection/inference tests/inference
git commit -m "feat: select TensorRT or PyTorch inference backend"
```

### Task 2: Bound uploads and annotation retention

**Files:**
- Create: `src/steel_inspection/api/storage.py`
- Modify: `src/steel_inspection/api/main.py`
- Modify: `tests/api/test_predict.py`

**Interfaces:**
- Produces: `read_limited_upload(upload: UploadFile, limit: int) -> bytes`; `store_annotation(...) -> Path | None`.

- [ ] **Step 1: Write failing API tests**

```python
def test_predict_rejects_an_upload_larger_than_limit(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    response = client.post("/predict", files={"image": ("x.png", b"12345", "image/png")})
    assert response.status_code == 413

def test_predict_does_not_persist_annotation_by_default(client, png_bytes, tmp_path):
    response = client.post("/predict", files={"image": ("x.png", png_bytes, "image/png")})
    assert response.json()["annotated_image"] is None
    assert not (tmp_path / "artifacts/results").exists()
```

- [ ] **Step 2: Run focused tests and observe failure**

Run: `python -m pytest tests/api/test_predict.py -q`

Expected: persistence/default and bounded-read assertions fail.

- [ ] **Step 3: Implement bounded reads and retention**

```python
async def read_limited_upload(upload: UploadFile, limit: int) -> bytes:
    chunks, size = [], 0
    while chunk := await upload.read(min(1024 * 1024, limit + 1 - size)):
        size += len(chunk)
        if size > limit:
            raise UploadTooLargeError
        chunks.append(chunk)
    return b"".join(chunks)
```

Before saving, delete only regular PNG files inside the resolved results directory older than 24 hours, then trim oldest regular files until total size plus the new PNG is at most 1 GiB. Convert write errors to HTTP 500 without exposing paths.

- [ ] **Step 4: Run API tests**

Run: `python -m pytest tests/api/test_predict.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/steel_inspection/api tests/api/test_predict.py
git commit -m "feat: bound uploads and annotation storage"
```

### Task 3: Wire environment configuration and document internal deployment

**Files:**
- Modify: `src/steel_inspection/api/main.py`
- Modify: `Dockerfile`
- Modify: `README.md`
- Test: `tests/api/test_predict.py`

**Interfaces:**
- `create_app(model_path, backend="auto", engine_path=...)` loads via `create_predictor`.
- Environment variables are read once at module initialization.

- [ ] **Step 1: Write failing startup tests**

```python
def test_explicit_unavailable_tensorrt_marks_health_unready(tmp_path):
    with TestClient(create_app(tmp_path / "model.pt", backend="tensorrt", engine_path=tmp_path / "missing.engine")) as client:
        assert client.get("/health").status_code == 503
```

- [ ] **Step 2: Run test and observe failure**

Run: `python -m pytest tests/api/test_predict.py::test_explicit_unavailable_tensorrt_marks_health_unready -q`

Expected: `create_app` does not accept the engine path.

- [ ] **Step 3: Wire factory and update operator instructions**

Document two Docker runs: PyTorch-only and GPU TensorRT `auto` with read-only model/engine mounts, `--gpus all`, `STEEL_INSPECTION_SAVE_ANNOTATIONS=false`, and an internal-network warning. Do not claim TensorRT works in the base image unless its runtime is actually included and smoke-tested.

- [ ] **Step 4: Run full suite and Docker syntax/build check**

Run: `python -m pytest -q`

Expected: PASS.

Run: `docker build -t steel-defect-inspection .`

Expected: image build exits 0.

- [ ] **Step 5: Commit**

```powershell
git add src/steel_inspection/api Dockerfile README.md tests/api/test_predict.py
git commit -m "feat: configure production inference serving"
```

### Task 4: Run GPU release checks and update deployment evidence

**Files:**
- Modify: `README.md`

**Interfaces:**
- Uses generated `artifacts/checkpoints/best.pt`, `artifacts/model-fp16.engine`, and one `data/train_images` sample; generated artifacts remain ignored.

- [ ] **Step 1: Add documented release commands**

```powershell
python scripts/benchmark.py --checkpoint artifacts/checkpoints/best.pt --image data/train_images/<image> --engine artifacts/model-fp16.engine --runs 100
docker run --rm --gpus all -e STEEL_INSPECTION_BACKEND=auto -e STEEL_INSPECTION_ENGINE=/app/artifacts/model-fp16.engine -v ${PWD}/artifacts:/app/artifacts steel-defect-inspection
```

- [ ] **Step 2: Run local release checks**

Run the benchmark and call `/health` plus `/predict` against the GPU container with a real image.

Expected: benchmark JSON names the selected backend; health is 200; predict response has `backend: "tensorrt"` when the TensorRT image/runtime is configured.

- [ ] **Step 3: Record observed results without committing data or engine artifacts**

Update README with only reproducible command and measured environment/result summary; do not add `data/` or `artifacts/` to Git.

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "docs: record production serving validation"
```
