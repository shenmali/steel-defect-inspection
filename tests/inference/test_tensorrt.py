"""Behavioral tests for the TensorRT inference adapter."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from steel_inspection.config import CLASS_NAMES, IMAGE_SIZE
from steel_inspection.inference.pytorch import ModelUnavailableError
from steel_inspection.inference.tensorrt import TensorRTPredictor


class _FakeContext:
    def __init__(self, execute_result: bool = True) -> None:
        self.execute_result = execute_result
        self.addresses: dict[str, int] = {}
        self.tensors: dict[int, torch.Tensor] = {}

    def set_tensor_address(self, name: str, address: int) -> bool:
        self.addresses[name] = address
        return True

    def execute_async_v3(self, stream: int) -> bool:
        if not self.execute_result:
            return False
        self.tensors[self.addresses["output"]].copy_(
            torch.tensor([[[[-1.0]], [[2.0]], [[-1.0]], [[-1.0]]]], dtype=torch.float32)
        )
        return True


class _FakeEngine:
    def __init__(self, tensor_modes, tensor_shapes, float32: object, execute_result: bool = True) -> None:
        self.num_io_tensors = 2
        self._tensor_modes = tensor_modes
        self._tensor_shapes = tensor_shapes
        self._float32 = float32
        self.context = _FakeContext(execute_result)

    def get_tensor_name(self, index: int) -> str:
        return ("input", "output")[index]

    def get_tensor_mode(self, name: str) -> object:
        return self._tensor_modes[name]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self._tensor_shapes[name]

    def get_tensor_dtype(self, name: str) -> object:
        return self._float32

    def create_execution_context(self) -> _FakeContext:
        return self.context


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch, engine: _FakeEngine, *, input_mode: object, output_mode: object, float32: object
) -> None:
    class Logger:
        ERROR = 0

        def __init__(self, severity: int) -> None:
            self.severity = severity

    class Runtime:
        def __init__(self, logger: Logger) -> None:
            self.logger = logger

        def deserialize_cuda_engine(self, serialized: bytes) -> _FakeEngine:
            assert serialized == b"engine"
            return engine

    fake_tensorrt = SimpleNamespace(
        DataType=SimpleNamespace(FLOAT=float32),
        Logger=Logger,
        Runtime=Runtime,
        TensorIOMode=SimpleNamespace(INPUT=input_mode, OUTPUT=output_mode),
    )
    original_empty = torch.empty

    def cpu_backed_cuda_empty(shape, *, device: str, dtype: torch.dtype) -> torch.Tensor:
        assert device == "cuda"
        assert dtype == torch.float32
        tensor = original_empty(shape, dtype=dtype)
        engine.context.tensors[tensor.data_ptr()] = tensor
        return tensor

    class Stream:
        cuda_stream = 1

        def synchronize(self) -> None:
            return None

    monkeypatch.setitem(__import__("sys").modules, "tensorrt", fake_tensorrt)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch, "empty", cpu_backed_cuda_empty)


def _engine_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.engine"
    path.write_bytes(b"engine")
    return path


def test_tensorrt_predictor_rejects_a_missing_engine(tmp_path: Path):
    """Catches startup that attempts TensorRT deserialization without an engine artifact."""
    with pytest.raises(ModelUnavailableError, match="engine is unavailable"):
        TensorRTPredictor(tmp_path / "missing.engine")


def test_tensorrt_predictor_reports_an_unavailable_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Catches an unavailable TensorRT runtime escaping as an import error at startup."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "tensorrt", None)

    with pytest.raises(ModelUnavailableError, match="bindings are unavailable"):
        TensorRTPredictor(_engine_path(tmp_path))


def test_tensorrt_predictor_rejects_invalid_engine_bindings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Catches engines whose output binding cannot produce the four model masks."""
    input_mode, output_mode, float32 = object(), object(), object()
    engine = _FakeEngine(
        {"input": input_mode, "output": output_mode},
        {"input": (1, 3, *IMAGE_SIZE), "output": (1, 3, *IMAGE_SIZE)},
        float32,
    )
    _install_fake_runtime(monkeypatch, engine, input_mode=input_mode, output_mode=output_mode, float32=float32)

    with pytest.raises(ModelUnavailableError, match="output shape"):
        TensorRTPredictor(_engine_path(tmp_path))


def test_tensorrt_predictor_raises_when_execution_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Catches failed async execution being converted into a stale output mask."""
    input_mode, output_mode, float32 = object(), object(), object()
    engine = _FakeEngine(
        {"input": input_mode, "output": output_mode},
        {"input": (1, 3, *IMAGE_SIZE), "output": (1, len(CLASS_NAMES), *IMAGE_SIZE)},
        float32,
        execute_result=False,
    )
    _install_fake_runtime(monkeypatch, engine, input_mode=input_mode, output_mode=output_mode, float32=float32)

    with pytest.raises(RuntimeError, match="inference failed"):
        TensorRTPredictor(_engine_path(tmp_path)).predict(np.zeros((3, 5, 3), dtype=np.uint8))


def test_tensorrt_predictor_returns_source_sized_tensorrt_masks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Catches TensorRT results that expose model-sized masks or the wrong backend label."""
    input_mode, output_mode, float32 = object(), object(), object()
    engine = _FakeEngine(
        {"input": input_mode, "output": output_mode},
        {"input": (1, 3, *IMAGE_SIZE), "output": (1, len(CLASS_NAMES), *IMAGE_SIZE)},
        float32,
    )
    _install_fake_runtime(monkeypatch, engine, input_mode=input_mode, output_mode=output_mode, float32=float32)

    result = TensorRTPredictor(_engine_path(tmp_path)).predict(np.zeros((3, 5, 3), dtype=np.uint8))

    assert result.backend == "tensorrt"
    assert result.mask.shape == (4, 3, 5)
    assert result.mask.tolist() == [
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    ]
    assert [defect.class_name for defect in result.defects] == ["class_2"]
