"""TensorRT model adapter with fixed-geometry image preprocessing."""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from steel_inspection.config import CLASS_NAMES, IMAGE_SIZE
from steel_inspection.inference.pytorch import ModelUnavailableError
from steel_inspection.inference.types import Defect, PredictionResult


class TensorRTPredictor:
    """Run a fixed-shape TensorRT segmentation engine with persistent CUDA buffers."""

    backend = "tensorrt"

    def __init__(self, engine_path: Path) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise ModelUnavailableError(f"TensorRT engine is unavailable: {self.engine_path}")
        if not torch.cuda.is_available():
            raise ModelUnavailableError("TensorRT inference requires CUDA")

        try:
            import tensorrt as trt
        except ImportError as error:
            raise ModelUnavailableError("TensorRT Python bindings are unavailable") from error

        try:
            self._runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            self._engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
            if self._engine is None:
                raise ValueError("engine deserialization returned no engine")
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise ValueError("engine could not create an execution context")
            self._input_name, self._output_name = self._binding_names(trt)
            self._input = torch.empty(
                tuple(self._engine.get_tensor_shape(self._input_name)), device="cuda", dtype=torch.float32
            )
            self._output = torch.empty(
                tuple(self._engine.get_tensor_shape(self._output_name)), device="cuda", dtype=torch.float32
            )
            self._stream = torch.cuda.Stream()
            if not self._context.set_tensor_address(self._input_name, self._input.data_ptr()):
                raise ValueError("unable to bind TensorRT input buffer")
            if not self._context.set_tensor_address(self._output_name, self._output.data_ptr()):
                raise ValueError("unable to bind TensorRT output buffer")
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ModelUnavailableError(f"Unable to load TensorRT engine {self.engine_path}: {error}") from error

    def _binding_names(self, trt: object) -> tuple[str, str]:
        """Validate fixed float32 engine bindings and return their names."""
        names = [self._engine.get_tensor_name(index) for index in range(self._engine.num_io_tensors)]
        input_names = [
            name for name in names if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        output_names = [
            name for name in names if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if len(input_names) != 1 or len(output_names) != 1:
            raise ValueError("TensorRT engine must have one input and one output")

        input_name, output_name = input_names[0], output_names[0]
        expected_input = (1, 3, *IMAGE_SIZE)
        expected_output = (1, len(CLASS_NAMES), *IMAGE_SIZE)
        if tuple(self._engine.get_tensor_shape(input_name)) != expected_input:
            raise ValueError(f"TensorRT input shape must be {expected_input}")
        if tuple(self._engine.get_tensor_shape(output_name)) != expected_output:
            raise ValueError(f"TensorRT output shape must be {expected_output}")
        if self._engine.get_tensor_dtype(input_name) != trt.DataType.FLOAT:
            raise ValueError("TensorRT input must use float32")
        if self._engine.get_tensor_dtype(output_name) != trt.DataType.FLOAT:
            raise ValueError("TensorRT output must use float32")
        return input_name, output_name

    def predict(self, image_bgr: np.ndarray) -> PredictionResult:
        """Segment a BGR source image and return masks at its original dimensions."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image")
        source_height, source_width = image_bgr.shape[:2]
        image = self._preprocess(image_bgr)

        started_at = perf_counter()
        with torch.cuda.stream(self._stream):
            self._input.copy_(image, non_blocking=True)
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT inference failed")
        self._stream.synchronize()
        probabilities = torch.sigmoid(self._output)[0].detach().cpu().numpy()
        latency_ms = (perf_counter() - started_at) * 1000

        masks = probabilities >= 0.5
        source_masks = np.stack(
            [
                cv2.resize(mask.astype(np.uint8), (source_width, source_height), interpolation=cv2.INTER_NEAREST)
                for mask in masks
            ]
        )
        defects = [
            Defect(
                class_name=class_name,
                confidence=float(probabilities[index].max()),
                coverage=float(source_masks[index].mean()),
            )
            for index, class_name in enumerate(CLASS_NAMES)
            if source_masks[index].any()
        ]
        return PredictionResult(
            has_defect=bool(defects),
            defects=defects,
            latency_ms=latency_ms,
            backend=self.backend,
            mask=source_masks,
        )

    @staticmethod
    def _preprocess(image_bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR input to a normalized CPU tensor at the fixed model size."""
        target_height, target_width = IMAGE_SIZE
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        array = np.ascontiguousarray(resized.transpose(2, 0, 1))
        return torch.from_numpy(array).float().div(255.0).unsqueeze(0)
