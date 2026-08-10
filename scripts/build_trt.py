"""Build an explicit-batch TensorRT FP16 engine from ONNX."""
import argparse
from pathlib import Path


def build_engine(onnx_path: Path, output_path: Path) -> None:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT Python bindings are required to build an engine") from error
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        details = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Unable to parse ONNX: {details}")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the FP16 engine")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(serialized))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_engine(args.onnx, args.output)
    print(args.output)
