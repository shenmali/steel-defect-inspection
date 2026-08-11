# Production Serving Design

## Goal

Make the internal-factory API safely select PyTorch or TensorRT inference while bounding upload memory and optional result-file storage.

## Scope

- `STEEL_INSPECTION_BACKEND` accepts `auto`, `tensorrt`, or `pytorch`.
- `auto` uses TensorRT only when CUDA and `STEEL_INSPECTION_ENGINE` are usable; otherwise it uses the PyTorch checkpoint from `STEEL_INSPECTION_MODEL`.
- `tensorrt` fails readiness with HTTP 503 if its engine or CUDA runtime is unavailable. `pytorch` retains the existing checkpoint behavior.
- TensorRT produces the existing `PredictionResult` contract, including masks resized to source-image dimensions and `backend="tensorrt"`.
- Upload reading stops at `MAX_UPLOAD_BYTES + 1`; a larger request returns HTTP 413 without retaining the full body in process memory.
- `STEEL_INSPECTION_SAVE_ANNOTATIONS` defaults to `false`. When enabled, files live in `artifacts/results`, files older than 24 hours are removed before a write, and the directory is trimmed to 1 GiB by deleting oldest files first.
- A TensorRT-capable container is documented separately from the base PyTorch container. The service is intended for an internal factory network; ingress restrictions are enforced by the host firewall or reverse proxy, not app-level authentication.

## Architecture

The API owns backend selection through a small factory. `PyTorchPredictor` remains unchanged as one implementation. A new `TensorRTPredictor` owns engine deserialization, persistent CUDA tensors, input preprocessing, execution, output conversion, and the same domain result contract. The factory returns a predictor or an explicit unavailable error, which keeps `/health` and `/predict` behavior consistent.

Upload reading uses `UploadFile.read()` in bounded chunks. Annotation retention is a focused storage helper: it is invoked only when saving is enabled and never removes files outside its configured results directory.

## Configuration and failure behavior

| Setting | Default | Behavior |
| --- | --- | --- |
| `STEEL_INSPECTION_BACKEND` | `auto` | Selects `auto`, `tensorrt`, or `pytorch`; any other value is unavailable. |
| `STEEL_INSPECTION_MODEL` | `artifacts/checkpoints/best.pt` | PyTorch checkpoint used by `pytorch` and as the `auto` fallback. |
| `STEEL_INSPECTION_ENGINE` | `artifacts/model-fp16.engine` | TensorRT engine used by `tensorrt` and attempted first by `auto`. |
| `STEEL_INSPECTION_SAVE_ANNOTATIONS` | `false` | Enables bounded PNG persistence only when exactly `true` (case-insensitive). |

`auto` never silently reports TensorRT when it is using PyTorch. Explicit `tensorrt` configuration never falls back: readiness is 503 so factory operators can correct the motor or GPU configuration.

## Verification

- Unit tests cover backend selection, explicit TensorRT failure, auto fallback, chunked oversized uploads, disabled annotation storage, TTL cleanup, capacity cleanup, and storage failures.
- API tests preserve existing response fields for both predictor implementations.
- A local GPU release check runs an FP16 engine against one source image and compares TensorRT and PyTorch masks within an agreed numerical tolerance.
- Docker documentation provides PowerShell commands for internal-only deployment with model and engine mounts.

## Non-goals

- Internet exposure, identity management, multi-tenant authorization, or TLS termination.
- Automatic engine rebuilding or ModelOpt conversion inside the serving container.
- Database/object-storage retention or a production monitoring stack.
