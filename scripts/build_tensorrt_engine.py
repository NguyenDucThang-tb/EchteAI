#!/usr/bin/env python3
"""Build TensorRT engine tu ONNX artifact cua backbone/FPN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    """Doc duong dan ONNX va che do precision can build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--engine")
    parser.add_argument("--precision", choices=["fp32", "int8"], required=True)
    parser.add_argument("--workspace-mb", type=int, default=4096)
    return parser.parse_args()


def load_tensorrt():
    """Import TensorRT tai runtime de local khong co TensorRT van doc duoc repo."""
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT Python package is required for this step.") from error
    return trt


def _network_creation_flags(trt):
    flag_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    explicit_batch = getattr(flag_enum, "EXPLICIT_BATCH", None) if flag_enum is not None else None
    return 0 if explicit_batch is None else 1 << int(explicit_batch)


def _set_workspace(trt, config, workspace_mb):
    bytes_count = int(workspace_mb) * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit"):
        pool_type = getattr(trt, "MemoryPoolType", None)
        if pool_type is not None and hasattr(pool_type, "WORKSPACE"):
            config.set_memory_pool_limit(pool_type.WORKSPACE, bytes_count)
            return
    if hasattr(config, "max_workspace_size"):
        config.max_workspace_size = bytes_count


def _enable_int8_if_supported(trt, config):
    builder_flag = getattr(trt, "BuilderFlag", None)
    int8_flag = getattr(builder_flag, "INT8", None) if builder_flag is not None else None
    if int8_flag is None and builder_flag is not None:
        int8_flag = getattr(builder_flag, "kINT8", None)
    if int8_flag is None:
        return False
    if hasattr(config, "set_flag"):
        config.set_flag(int8_flag)
    else:
        config.flags |= 1 << int(int8_flag)
    return True


def main():
    """Parse ONNX, tao optimization profile co dinh va luu serialized engine."""
    args = parse_args()
    trt = load_tensorrt()
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    metadata_path = Path(args.metadata) if args.metadata else onnx_path.with_name(f"{onnx_path.stem}_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    example_shape = tuple(int(value) for value in metadata.get("example_shape", [1, 3, 960, 1600]))
    input_name = metadata.get("input_name", "input0")

    engine_path = Path(args.engine) if args.engine else onnx_path.with_suffix(f".{args.precision}.engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(_network_creation_flags(trt))
    parser = trt.OnnxParser(network, logger)
    parsed = bool(parser.parse_from_file(str(onnx_path))) if hasattr(parser, "parse_from_file") else bool(parser.parse(onnx_path.read_bytes()))
    if not parsed:
        errors = [parser.get_error(index).desc() for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    _set_workspace(trt, config, args.workspace_mb)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, example_shape, example_shape, example_shape)
    config.add_optimization_profile(profile)
    if args.precision == "int8" and not _enable_int8_if_supported(trt, config):
        print("INT8 builder flag unavailable; relying on explicit Q/DQ ONNX if present.", flush=True)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine.")
    engine_path.write_bytes(bytes(serialized))

    summary = {
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": args.precision,
        "input_name": input_name,
        "example_shape": list(example_shape),
        "tensorrt_version": trt.__version__,
    }
    summary_path = engine_path.with_suffix(engine_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved engine summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

