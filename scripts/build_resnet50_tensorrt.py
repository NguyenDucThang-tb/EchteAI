#!/usr/bin/env python3
"""Build TensorRT engines from exported ONNX backbone artifacts.

Supports two INT8 paths:

- explicit_qdq: consume a Q/DQ ONNX graph directly
- native_ptq: build INT8 from an FP32 ONNX graph using TensorRT calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.fasterrcnn_qat.config import load_config
from pipelines.fasterrcnn_qat.data import build_coco_loader
from pipelines.fasterrcnn_qat.models import build_fasterrcnn_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--engine")
    parser.add_argument("--precision", choices=["fp32", "int8"], required=True)
    parser.add_argument(
        "--int8-mode",
        choices=["explicit_qdq", "native_ptq"],
        default="explicit_qdq",
        help="INT8 build path. explicit_qdq expects a Q/DQ ONNX graph. native_ptq calibrates an FP32 ONNX graph.",
    )
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--calib-cache")
    parser.add_argument("--min-shape", help="Dynamic profile min shape, e.g. 1x3x800x800")
    parser.add_argument("--opt-shape", help="Dynamic profile opt shape, e.g. 1x3x1088x800")
    parser.add_argument("--max-shape", help="Dynamic profile max shape, e.g. 1x3x1333x1333")
    parser.add_argument("--config", help="Runtime config required for --int8-mode native_ptq")
    parser.add_argument("--calib-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--calib-limit", type=int, default=128)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument(
        "--calib-shape",
        help="Calibration tensor shape CxHxW or NxCxHxW. Defaults to ONNX example_shape / opt_shape.",
    )
    return parser.parse_args()


def load_tensorrt():
    try:
        import tensorrt as trt
    except ImportError:
        try:
            from tensorrt_bindings import tensorrt as trt
        except ImportError as error:
            raise RuntimeError("TensorRT Python package is required for this step.") from error
    return trt


def get_tensorrt_version(trt):
    version = getattr(trt, "__version__", None)
    if version is not None:
        return str(version)
    try:
        from importlib.metadata import version as pkg_version

        for package_name in ("tensorrt", "tensorrt-cu13-bindings", "tensorrt-cu13-libs"):
            try:
                return str(pkg_version(package_name))
            except Exception:
                continue
    except Exception:
        pass
    return "unknown"


def _logger(trt):
    return trt.Logger(trt.Logger.WARNING)


def _network_creation_flags(trt):
    flag_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    if flag_enum is None:
        return 0
    explicit_batch = getattr(flag_enum, "EXPLICIT_BATCH", None)
    if explicit_batch is None:
        return 0
    return 1 << int(explicit_batch)


def _set_workspace(trt, config, workspace_mb):
    if hasattr(config, "set_memory_pool_limit"):
        pool_type = getattr(trt, "MemoryPoolType", None)
        if pool_type is not None and hasattr(pool_type, "WORKSPACE"):
            config.set_memory_pool_limit(pool_type.WORKSPACE, int(workspace_mb) * 1024 * 1024)
            return
        if hasattr(config, "max_workspace_size"):
            config.max_workspace_size = int(workspace_mb) * 1024 * 1024
            return
    else:
        config.max_workspace_size = int(workspace_mb) * 1024 * 1024


def _shape_from_text(text):
    return tuple(int(value) for value in str(text).split("x"))


def _resolve_profile_shapes(args, example_shape):
    min_shape = _shape_from_text(args.min_shape) if args.min_shape else tuple(example_shape)
    opt_shape = _shape_from_text(args.opt_shape) if args.opt_shape else tuple(example_shape)
    max_shape = _shape_from_text(args.max_shape) if args.max_shape else tuple(example_shape)
    if not (len(min_shape) == len(opt_shape) == len(max_shape) == len(example_shape)):
        raise ValueError("min/opt/max shapes must have the same rank as example_shape")
    for axis, (mn, op, mx) in enumerate(zip(min_shape, opt_shape, max_shape)):
        if not (mn <= op <= mx):
            raise ValueError(
                f"Invalid TensorRT profile axis {axis}: expected min <= opt <= max, got {mn}, {op}, {mx}"
            )
    return min_shape, opt_shape, max_shape


def _write_int8_calibration_cache(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scales = metadata.get("checkpoint_extra", {}).get("calibration_cache")
    if isinstance(scales, str) and scales.strip():
        path.write_text(scales, encoding="utf-8")
        return path
    path.write_text("", encoding="utf-8")
    return path


def _enable_int8_if_supported(trt, config):
    builder_flag = getattr(trt, "BuilderFlag", None)
    if builder_flag is None:
        return False
    int8_flag = getattr(builder_flag, "INT8", None)
    if int8_flag is None:
        int8_flag = getattr(builder_flag, "kINT8", None)
    if int8_flag is None:
        return False
    if hasattr(config, "set_flag"):
        config.set_flag(int8_flag)
    else:
        config.flags |= 1 << int(int8_flag)
    return True


def _ensure_pycuda():
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except ImportError as error:
        raise RuntimeError("pycuda is required for TensorRT native PTQ calibration.") from error
    return cuda


def _parse_calib_shape(text, fallback_nchw):
    if not text:
        return tuple(int(v) for v in fallback_nchw)
    values = _shape_from_text(text)
    if len(values) == 3:
        return (int(fallback_nchw[0]),) + tuple(int(v) for v in values)
    if len(values) == 4:
        return tuple(int(v) for v in values)
    raise ValueError("--calib-shape must be CxHxW or NxCxHxW")


def _preprocess_for_calibration(model, images, target_nchw):
    device = next(model.parameters()).device
    images = [image.to(device) for image in images]
    image_list, _ = model.transform(images, None)
    tensors = image_list.tensors.detach().cpu().numpy().astype(np.float32, copy=False)
    target_nchw = tuple(int(v) for v in target_nchw)
    if tuple(int(v) for v in tensors.shape) == target_nchw:
        return np.ascontiguousarray(tensors)

    import torch
    import torch.nn.functional as F

    resized = F.interpolate(
        torch.from_numpy(tensors),
        size=target_nchw[-2:],
        mode="bilinear",
        align_corners=False,
    )
    if int(resized.shape[0]) != int(target_nchw[0]):
        raise ValueError(
            f"Calibration batch mismatch: transformed batch={int(resized.shape[0])} target batch={int(target_nchw[0])}"
        )
    return np.ascontiguousarray(resized.numpy().astype(np.float32, copy=False))


def make_trt_calibrator(trt, cuda, batches, cache_path):
    base_cls = getattr(trt, "IInt8EntropyCalibrator2", None)
    if base_cls is None:
        base_cls = getattr(trt, "IInt8MinMaxCalibrator", None)
    if base_cls is None:
        raise RuntimeError("This TensorRT runtime does not expose an INT8 calibrator base class.")

    class TensorRTImageCalibrator(base_cls):
        def __init__(self):
            super().__init__()
            self.cuda = cuda
            self.batches = list(batches)
            self.cache_path = Path(cache_path)
            self.index = 0
            if not self.batches:
                raise ValueError("Calibration batches are empty")
            self.batch_size = int(self.batches[0].shape[0])
            self.device_input = cuda.mem_alloc(self.batches[0].nbytes)

        def get_batch_size(self):
            return int(self.batch_size)

        def get_batch(self, names):
            if self.index >= len(self.batches):
                return None
            batch = self.batches[self.index]
            self.index += 1
            self.cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_path.is_file():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(cache)

    return TensorRTImageCalibrator()


def _build_native_ptq_calibrator(args, example_shape, opt_shape):
    if not args.config:
        raise ValueError("--config is required for --int8-mode native_ptq")

    config = load_config(args.config, require_dataset=True)
    model = build_fasterrcnn_model(config).cpu().eval()
    loader = build_coco_loader(
        config,
        args.calib_split,
        shuffle=False,
        limit=int(args.calib_limit),
        batch_size=int(args.calib_batch_size),
    )
    target_nchw = _parse_calib_shape(args.calib_shape, opt_shape or example_shape)
    if int(target_nchw[1]) != 3:
        raise ValueError(f"Calibration expects RGB input with 3 channels, got shape {target_nchw}")

    batches = []
    for images, _ in loader:
        batch = _preprocess_for_calibration(model, images, target_nchw)
        batches.append(batch)
    if not batches:
        raise RuntimeError("No calibration batches were produced from the requested split/limit.")

    cuda = _ensure_pycuda()
    cache_path = args.calib_cache or Path(args.engine).with_suffix(".cache")
    return cuda, batches, cache_path, target_nchw


def main():
    args = parse_args()
    trt = load_tensorrt()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    metadata_path = Path(args.metadata) if args.metadata else onnx_path.with_name(f"{onnx_path.stem}_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    example_shape = tuple(int(v) for v in metadata.get("example_shape", [1, 3, 256, 320]))
    input_name = metadata.get("input_name", "input0")
    min_shape, opt_shape, max_shape = _resolve_profile_shapes(args, example_shape)

    engine_path = Path(args.engine) if args.engine else onnx_path.with_suffix(f".{args.precision}.engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = _logger(trt)
    builder = trt.Builder(logger)
    flags = _network_creation_flags(trt)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    parsed = False
    if hasattr(parser, "parse_from_file"):
        parsed = bool(parser.parse_from_file(str(onnx_path)))
    else:
        parsed = bool(parser.parse(onnx_path.read_bytes()))
    if not parsed:
        errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    _set_workspace(trt, config, args.workspace_mb)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if args.precision == "int8":
        enabled = _enable_int8_if_supported(trt, config)
        if args.int8_mode == "explicit_qdq":
            if not enabled:
                print(
                    "TensorRT INT8 builder flag is unavailable in this runtime; relying on explicit Q/DQ ONNX graph.",
                    flush=True,
                )
            calib_cache = args.calib_cache or engine_path.with_suffix(".cache")
            _write_int8_calibration_cache(calib_cache, metadata)
        else:
            if not enabled:
                raise RuntimeError("TensorRT INT8 builder flag is unavailable; native PTQ cannot run.")
            cuda, batches, cache_path, calib_shape = _build_native_ptq_calibrator(
                args, example_shape, opt_shape
            )
            calibrator = make_trt_calibrator(trt, cuda, batches, cache_path)
            if hasattr(config, "int8_calibrator"):
                config.int8_calibrator = calibrator
            else:
                raise RuntimeError("This TensorRT runtime does not expose builder_config.int8_calibrator.")
            if hasattr(config, "set_calibration_profile"):
                calib_profile = builder.create_optimization_profile()
                calib_profile.set_shape(input_name, calib_shape, calib_shape, calib_shape)
                config.set_calibration_profile(calib_profile)
            print(
                f"TensorRT native PTQ calibration batches={len(batches)} calib_shape={list(calib_shape)} cache={cache_path}",
                flush=True,
            )

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine.")
    engine_path.write_bytes(bytes(serialized))

    summary = {
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": args.precision,
        "int8_mode": args.int8_mode if args.precision == "int8" else None,
        "input_name": input_name,
        "example_shape": list(example_shape),
        "profile_min_shape": list(min_shape),
        "profile_opt_shape": list(opt_shape),
        "profile_max_shape": list(max_shape),
        "tensorrt_version": get_tensorrt_version(trt),
    }
    summary_path = engine_path.with_suffix(engine_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved engine summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
