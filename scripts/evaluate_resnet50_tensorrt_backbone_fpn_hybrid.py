#!/usr/bin/env python3
"""Evaluate a TensorRT backbone-FPN engine + PyTorch RPN/ROI heads hybrid."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.fasterrcnn_qat.checkpoint import load_checkpoint, load_partial_checkpoint
from pipelines.fasterrcnn_qat.config import choose_device, load_config, quantized_modules_for_variant, resolve_qat_profile
from pipelines.fasterrcnn_qat.data import build_coco_loader, unwrap_coco_dataset
from pipelines.fasterrcnn_qat.metrics import _coco_metrics, native_detection_metrics, save_metrics
from pipelines.fasterrcnn_qat.models import build_fasterrcnn_model
from pipelines.fasterrcnn_qat.quantization import (
    mixed_precision_policy_from_config,
    module_qconfig_map_from_policy,
    policy_scope_to_quantized_modules,
    prepare_selective_qat,
)


ENGINE_OUTPUT_NAMES = ("p2", "p3", "p4", "p5", "p6_pool")
MODEL_FEATURE_NAMES = ("0", "1", "2", "3", "pool")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["fp32", "qat_graph"], default="qat_graph")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--qat-checkpoint")
    parser.add_argument("--partial-fp32-checkpoint", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--progress-frequency", type=int, default=10)
    return parser.parse_args()


def load_tensorrt_and_cuda():
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT Python package is required for this step.") from error
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except ImportError as error:
        raise RuntimeError("pycuda is required for TensorRT hybrid evaluation.") from error
    return trt, cuda


def _tensor_io_mode_enum(trt, engine):
    enum_obj = getattr(trt, "TensorIOMode", None)
    if enum_obj is not None:
        return enum_obj
    return getattr(engine, "TensorIOMode", None)


def _trt_dtype_to_numpy(trt, dtype):
    mapping = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int8: np.int8,
        trt.int32: np.int32,
        trt.bool: np.bool_,
    }
    if hasattr(trt, "uint8"):
        mapping[trt.uint8] = np.uint8
    if dtype not in mapping:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


class TensorRTBackboneFPNRunner:
    def __init__(self, engine_path: str | Path):
        trt, cuda = load_tensorrt_and_cuda()
        self.trt = trt
        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.tensor_mode_enum = _tensor_io_mode_enum(trt, self.engine)
        self.input_name, self.output_names = self._discover_tensors()
        self.current_shape = None
        self.output_tensors = {}
        self.shapes_seen = set()

    def _discover_tensors(self):
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        input_mode = getattr(self.tensor_mode_enum, "INPUT", None)
        output_mode = getattr(self.tensor_mode_enum, "OUTPUT", None)
        input_names = [name for name in tensor_names if self.engine.get_tensor_mode(name) == input_mode]
        output_names = [name for name in tensor_names if self.engine.get_tensor_mode(name) == output_mode]
        if len(input_names) != 1:
            raise RuntimeError(f"Expected exactly one TensorRT input, found {input_names}")
        if tuple(output_names) != ENGINE_OUTPUT_NAMES:
            raise RuntimeError(
                f"Expected TensorRT outputs {ENGINE_OUTPUT_NAMES}, got {tuple(output_names)}"
            )
        return input_names[0], output_names

    def _allocate(self, shape):
        shape = tuple(int(v) for v in shape)
        if self.current_shape == shape:
            return
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(self.input_name, shape)
        else:
            index = self.engine.get_binding_index(self.input_name)
            self.context.set_binding_shape(index, shape)
        self.output_tensors = {}
        for name in self.output_names:
            out_shape = tuple(int(v) for v in self.context.get_tensor_shape(name))
            out_dtype = _trt_dtype_to_numpy(self.trt, self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.empty((), dtype=out_dtype)).dtype
            self.output_tensors[name] = torch.empty(out_shape, device="cuda", dtype=torch_dtype)
        self.current_shape = shape

    def __call__(self, batch_tensor: torch.Tensor):
        if batch_tensor.device.type != "cuda":
            raise ValueError("TensorRT backbone expects a CUDA tensor input")
        shape = tuple(int(v) for v in batch_tensor.shape)
        self._allocate(shape)
        self.shapes_seen.add(shape)
        self.context.set_tensor_address(self.input_name, int(batch_tensor.data_ptr()))
        for name, tensor in self.output_tensors.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT hybrid backbone execution failed.")
        self.stream.synchronize()
        return OrderedDict(
            (model_name, self.output_tensors[engine_name])
            for model_name, engine_name in zip(MODEL_FEATURE_NAMES, self.output_names)
        )


def load_hybrid_model(config, args, device):
    model = build_fasterrcnn_model(config)
    if args.model == "fp32":
        checkpoint = args.fp32_checkpoint or config["output"].get("fp32_best")
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError("FP32 hybrid mode requires --fp32-checkpoint or output.fp32_best")
        if args.partial_fp32_checkpoint:
            load_partial_checkpoint(checkpoint, model, map_location="cpu")
        else:
            load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
        return model.to(device).eval()

    checkpoint = args.qat_checkpoint or config["output"].get("qat_best") or config["output"].get("qat_last")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("QAT graph hybrid mode requires --qat-checkpoint or output.qat_best/output.qat_last")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("extra", {}) if isinstance(payload, dict) else {}
    variant = str(metadata.get("variant", config["quantization"].get("variant", "M3"))).upper()
    backend = metadata.get("backend", config["quantization"].get("backend", "x86"))
    qat_profile = resolve_qat_profile(config, metadata)
    quantized_modules = metadata.get(
        "quantized_modules", quantized_modules_for_variant(config, variant)
    )
    mixed_precision_policy = metadata.get("mixed_precision_policy") or mixed_precision_policy_from_config(config)
    module_qconfig_map = None
    if mixed_precision_policy is not None:
        quantized_modules = policy_scope_to_quantized_modules(mixed_precision_policy)
        module_qconfig_map = module_qconfig_map_from_policy(mixed_precision_policy)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="must run observer before calling calculate_qparams")
        model = prepare_selective_qat(
            model,
            variant,
            backend,
            quantized_modules=quantized_modules,
            module_qconfig_map=module_qconfig_map,
            qconfig_profile=qat_profile,
        )
    load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    return model.to(device).eval()


def preprocess_batch(model, images, device):
    images = [image.to(device) for image in images]
    original_sizes = [tuple(int(v) for v in image.shape[-2:]) for image in images]
    image_list, _ = model.transform(images, None)
    return image_list, original_sizes


@torch.inference_mode()
def evaluate_hybrid_model(model, backbone_runner, loader, device, progress_frequency=10):
    model.eval()
    predictions, targets = [], []
    total_images = len(loader.dataset)
    processed = 0
    timings = []
    print(
        f"hybrid backbone-fpn evaluation started: target={total_images} images device={device} transform=model.transform(...)",
        flush=True,
    )

    for images, batch_targets in loader:
        image_list, original_sizes = preprocess_batch(model, images, device)

        t0 = time.perf_counter()
        features = backbone_runner(image_list.tensors)
        proposals, _ = model.rpn(image_list, features, None)
        outputs, _ = model.roi_heads(features, proposals, image_list.image_sizes, None)
        outputs = model.transform.postprocess(outputs, image_list.image_sizes, original_sizes)
        timings.append((time.perf_counter() - t0) * 1000.0 / max(len(images), 1))

        predictions.extend(
            [{key: value.detach().cpu() for key, value in output.items()} for output in outputs]
        )
        targets.extend(
            [
                {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
                for target in batch_targets
            ]
        )
        processed += len(images)
        if (
            progress_frequency
            and (processed % progress_frequency < len(images) or processed >= total_images)
        ):
            print(f"hybrid backbone-fpn progress: {processed}/{total_images} images", flush=True)

    print("hybrid backbone-fpn inference completed; calculating detection metrics", flush=True)
    metrics = native_detection_metrics(predictions, targets)
    dataset = unwrap_coco_dataset(loader.dataset)
    canonical = None
    if not bool(getattr(dataset, "binary_collapse_foreground", False)):
        canonical = _coco_metrics(predictions, targets, dataset)
    if canonical:
        metrics.update(canonical)
    metrics["avg_inference_ms_per_image"] = sum(timings) / max(len(timings), 1)
    metrics["fps"] = 1000.0 / metrics["avg_inference_ms_per_image"] if metrics["avg_inference_ms_per_image"] > 0 else float("nan")
    metrics["engine_input_shapes_seen"] = [
        list(shape) for shape in sorted(backbone_runner.shapes_seen)
    ]
    print("hybrid backbone-fpn evaluation completed", flush=True)
    return metrics


def main():
    args = parse_args()
    if int(args.batch_size) != 1:
        raise ValueError("Hybrid TensorRT benchmark currently supports batch_size=1 only")
    config = load_config(args.config, require_dataset=True)
    device = choose_device(config.get("device", "auto"))
    if device.type != "cuda":
        raise RuntimeError("Hybrid TensorRT benchmark requires CUDA")

    loader = build_coco_loader(
        config, args.split, shuffle=False, limit=args.limit, batch_size=1,
    )
    model = load_hybrid_model(config, args, device)
    backbone_runner = TensorRTBackboneFPNRunner(args.engine)
    metrics = evaluate_hybrid_model(
        model,
        backbone_runner,
        loader,
        device,
        progress_frequency=int(args.progress_frequency),
    )

    output = args.output or str(Path(config["output"]["directory"]) / "tensorrt_backbone_fpn_hybrid_eval.json")
    save_metrics(output, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved metrics to {output}", flush=True)


if __name__ == "__main__":
    main()
