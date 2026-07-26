#!/usr/bin/env python3
"""Benchmark ConvNeXt TensorRT-backbone hybrid FP32 vs INT8/QAT."""

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
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import concat_box_prediction_layers

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint
from pipelines.convnext_qat.compiler import resolve_compiler_scope
from pipelines.convnext_qat.config import choose_device, load_config, quantized_modules_for_variant
from pipelines.convnext_qat.data import build_coco_loader, unwrap_coco_dataset
from pipelines.convnext_qat.metrics import _coco_metrics, native_detection_metrics, save_metrics
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import prepare_selective_qat, set_qat_phase


M3_SCOPE = "backbone_fpn_rpn_m3"


def parse_args():
    """Doc engine TensorRT, checkpoint va so anh benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--fp32-engine", required=True)
    parser.add_argument("--int8-engine", required=True)
    parser.add_argument("--fp32-checkpoint", required=True)
    parser.add_argument("--qat-checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--progress-frequency", type=int, default=10)
    return parser.parse_args()


def load_tensorrt():
    """Import TensorRT tai runtime de script chi phu thuoc khi thuc su chay."""
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT Python package is required for this step.") from error
    return trt


def _tensor_io_mode_enum(trt, engine):
    enum_obj = getattr(trt, "TensorIOMode", None)
    return enum_obj if enum_obj is not None else getattr(engine, "TensorIOMode", None)


def _trt_dtype_to_torch(trt, dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "uint8"):
        mapping[trt.uint8] = torch.uint8
    if dtype not in mapping:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


class TensorRTBackboneRunner:
    """Chay TensorRT engine va tra tuple CUDA tensor cho PyTorch phan con lai."""

    def __init__(self, engine_path: str | Path):
        trt = load_tensorrt()
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.tensor_mode_enum = _tensor_io_mode_enum(trt, self.engine)
        self.input_name, self.output_names = self._discover_tensors()
        self.current_shape = None
        self.output_tensors = {}

    def _discover_tensors(self):
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        input_mode = getattr(self.tensor_mode_enum, "INPUT", None)
        output_mode = getattr(self.tensor_mode_enum, "OUTPUT", None)
        input_names = [name for name in tensor_names if self.engine.get_tensor_mode(name) == input_mode]
        output_names = [name for name in tensor_names if self.engine.get_tensor_mode(name) == output_mode]
        if len(input_names) != 1:
            raise RuntimeError(f"Expected exactly one TensorRT input, found {input_names}")
        return input_names[0], output_names

    def _allocate(self, shape):
        shape = tuple(int(value) for value in shape)
        if self.current_shape == shape:
            return
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(self.input_name, shape)
        else:
            index = self.engine.get_binding_index(self.input_name)
            self.context.set_binding_shape(index, shape)
        self.output_tensors = {}
        for name in self.output_names:
            out_shape = tuple(int(value) for value in self.context.get_tensor_shape(name))
            out_dtype = _trt_dtype_to_torch(self.trt, self.engine.get_tensor_dtype(name))
            self.output_tensors[name] = torch.empty(out_shape, device="cuda", dtype=out_dtype)
        self.current_shape = shape

    def __call__(self, batch_tensor: torch.Tensor):
        if batch_tensor.device.type != "cuda":
            raise ValueError("TensorRT backbone expects a CUDA tensor input")
        stream = torch.cuda.current_stream(batch_tensor.device)
        self._allocate(batch_tensor.shape)
        self.context.set_tensor_address(self.input_name, int(batch_tensor.data_ptr()))
        for name, tensor in self.output_tensors.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        ok = self.context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT backbone execution failed.")
        stream.synchronize()
        return tuple(self.output_tensors[name] for name in self.output_names)


def load_fp32_model(config, checkpoint, device):
    """Nap detector FP32; body se duoc chay bang TensorRT trong evaluator."""
    model = build_fasterrcnn_convnext(config)
    load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    return model.to(device).eval()


def load_qat_model(config, checkpoint, device):
    """Nap topology selective-QAT frozen de FPN/RPN con lai khop checkpoint."""
    model = build_fasterrcnn_convnext(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("extra", {}) if isinstance(payload, dict) else {}
    variant = str(metadata.get("variant", config["quantization"].get("variant", "M3"))).upper()
    backend = metadata.get("backend", config["quantization"].get("backend", "auto"))
    quantized_modules = metadata.get("quantized_modules", quantized_modules_for_variant(config, variant))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="must run observer before calling calculate_qparams")
        model = prepare_selective_qat(
            model, variant, backend, quantized_modules=quantized_modules,
        )
    load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    set_qat_phase(model, "frozen")
    return model.to(device).eval()


def preprocess_batch(model, images, targets, fixed_height, fixed_width, device):
    """Resize co dinh theo engine va giu target goc cho metrics."""
    image_mean = torch.tensor(model.transform.image_mean, device=device).view(-1, 1, 1)
    image_std = torch.tensor(model.transform.image_std, device=device).view(-1, 1, 1)
    batched, original_sizes = [], []
    for image in images:
        image = image.to(device)
        original_sizes.append(tuple(int(value) for value in image.shape[-2:]))
        normalized = (image - image_mean) / image_std
        resized = F.interpolate(
            normalized.unsqueeze(0),
            size=(fixed_height, fixed_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        batched.append(resized)
    image_list = ImageList(torch.stack(batched, dim=0), [(fixed_height, fixed_width)] * len(batched))
    cpu_targets = [
        {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]
    return image_list, original_sizes, cpu_targets


def _features_from_fpn_tuple(tensors):
    """Chuyen tuple feature P2-P6 thanh OrderedDict dung format torchvision."""
    return OrderedDict((str(index), tensor) for index, tensor in enumerate(tensors))


def _split_m3_outputs(trt_outputs):
    """Tach output cua engine M3: FPN features, RPN shared features, objectness."""
    if len(trt_outputs) % 3 != 0:
        raise RuntimeError(
            "M3 TensorRT engine must return FPN features + RPN shared features "
            f"+ RPN objectness logits; got {len(trt_outputs)} tensors"
        )
    levels = len(trt_outputs) // 3
    if levels <= 0:
        raise RuntimeError("M3 TensorRT engine returned no feature levels")
    fpn_features = trt_outputs[:levels]
    shared_features = trt_outputs[levels: 2 * levels]
    objectness = trt_outputs[2 * levels:]
    return fpn_features, shared_features, objectness


def _rpn_proposals_from_precomputed_head(model, image_list, features, objectness, pred_bbox_deltas):
    """Chay proposal decode/NMS cua RPN tu logits/deltas da co.

    Scope M3 da dua backbone+FPN+RPN shared conv+RPN classification vao
    TensorRT. RPN bbox regression khong thuoc M3, nen bbox deltas van duoc
    tinh bang PyTorch tu shared features roi dua vao decode/filter goc.
    """
    feature_list = list(features.values())
    anchors = model.rpn.anchor_generator(image_list, feature_list)
    num_images = len(anchors)
    num_anchors_per_level = [
        int(tensor.shape[1] * tensor.shape[2] * tensor.shape[3])
        for tensor in objectness
    ]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(
        list(objectness),
        list(pred_bbox_deltas),
    )
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, _ = model.rpn.filter_proposals(
        proposals,
        objectness,
        image_list.image_sizes,
        num_anchors_per_level,
    )
    return boxes


@torch.inference_mode()
def evaluate_hybrid_model(model, runner, loader, device, fixed_height, fixed_width, scope, progress_frequency=10):
    """Chay TensorRT theo scope da chon roi tiep tuc phan con lai bang PyTorch."""
    predictions, targets, timings = [], [], []
    total_images = len(loader.dataset)
    processed = 0
    print(
        f"hybrid evaluation started: target={total_images} images device={device} shape={fixed_height}x{fixed_width} scope={scope}",
        flush=True,
    )
    for images, batch_targets in loader:
        image_list, original_sizes, cpu_targets = preprocess_batch(
            model, images, batch_targets, fixed_height, fixed_width, device,
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        trt_outputs = runner(image_list.tensors)
        if scope == "backbone":
            feature_dict = _features_from_fpn_tuple(trt_outputs)
            features = model.backbone.fpn(feature_dict)
            proposals, _ = model.rpn(image_list, features, None)
        elif scope == M3_SCOPE:
            fpn_features, shared_features, objectness = _split_m3_outputs(trt_outputs)
            features = _features_from_fpn_tuple(fpn_features)
            pred_bbox_deltas = [
                model.rpn.head.bbox_pred(feature)
                for feature in shared_features
            ]
            proposals = _rpn_proposals_from_precomputed_head(
                model,
                image_list,
                features,
                objectness,
                pred_bbox_deltas,
            )
        else:
            features = _features_from_fpn_tuple(trt_outputs)
            proposals, _ = model.rpn(image_list, features, None)
        outputs, _ = model.roi_heads(features, proposals, image_list.image_sizes, None)
        outputs = model.transform.postprocess(outputs, image_list.image_sizes, original_sizes)
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0 / max(len(images), 1))

        predictions.extend(
            [{key: value.detach().cpu() for key, value in output.items()} for output in outputs]
        )
        targets.extend(cpu_targets)
        processed += len(images)
        if progress_frequency and (processed % progress_frequency < len(images) or processed >= total_images):
            print(f"hybrid evaluation progress: {processed}/{total_images} images", flush=True)

    metrics = native_detection_metrics(predictions, targets)
    dataset = unwrap_coco_dataset(loader.dataset)
    if not bool(getattr(dataset, "binary_collapse_foreground", False)):
        metrics.update(_coco_metrics(predictions, targets, dataset))
    metrics["avg_inference_ms_per_image"] = float(np.mean(timings)) if timings else float("nan")
    metrics["fps"] = 1000.0 / metrics["avg_inference_ms_per_image"] if metrics["avg_inference_ms_per_image"] > 0 else float("nan")
    metrics["engine_shape"] = [int(fixed_height), int(fixed_width)]
    metrics["scope"] = scope
    return metrics


def _line(label, value, suffix=""):
    print(f"  {label}: {value:.4f}{suffix}")


def main():
    """Benchmark FP32 TensorRT hybrid va INT8 TensorRT hybrid tren cung anh."""
    args = parse_args()
    if args.images <= 0:
        raise ValueError("images must be positive")
    config = load_config(args.config, require_dataset=True)
    compiler_cfg = config.get("quantization", {}).get("compiler", {})
    scope = resolve_compiler_scope(config)
    fixed_height = int(args.height or compiler_cfg.get("example_height", config["model"].get("min_size", 960)))
    fixed_width = int(args.width or compiler_cfg.get("example_width", config["model"].get("max_size", 1600)))
    device = choose_device(config.get("device", "auto"))
    if device.type != "cuda":
        raise RuntimeError("TensorRT hybrid benchmark requires CUDA")
    loader = build_coco_loader(config, args.split, shuffle=False, limit=args.images, batch_size=1)

    print(f"Loading FP32 model: {args.fp32_checkpoint}", flush=True)
    fp32_model = load_fp32_model(config, args.fp32_checkpoint, device)
    print(f"Loading FP32 TensorRT engine: {args.fp32_engine}", flush=True)
    fp32_runner = TensorRTBackboneRunner(args.fp32_engine)
    fp32 = evaluate_hybrid_model(
        fp32_model, fp32_runner, loader, device, fixed_height, fixed_width, scope, args.progress_frequency,
    )

    print(f"Loading QAT model: {args.qat_checkpoint}", flush=True)
    qat_model = load_qat_model(config, args.qat_checkpoint, device)
    print(f"Loading INT8 TensorRT engine: {args.int8_engine}", flush=True)
    int8_runner = TensorRTBackboneRunner(args.int8_engine)
    int8 = evaluate_hybrid_model(
        qat_model, int8_runner, loader, device, fixed_height, fixed_width, scope, args.progress_frequency,
    )

    delta = {
        "accuracy": int8["accuracy"] - fp32["accuracy"],
        "precision": int8["precision"] - fp32["precision"],
        "mean_iou": int8["mean_iou"] - fp32["mean_iou"],
        "map_50_95": int8["map_50_95"] - fp32["map_50_95"],
        "speedup": fp32["avg_inference_ms_per_image"] / int8["avg_inference_ms_per_image"],
    }
    results = {"images": args.images, "fp32_hybrid": fp32, "int8_hybrid": int8, "delta": delta}
    output = Path(args.output or Path(config["output"]["directory"]) / "convnext_tensorrt_hybrid_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(output, results)

    print("\nFP32 TensorRT hybrid:")
    _line("mAP@50:95", fp32["map_50_95"])
    _line("Avg inference", fp32["avg_inference_ms_per_image"], " ms/image")
    print("\nINT8 TensorRT hybrid:")
    _line("mAP@50:95", int8["map_50_95"])
    _line("Avg inference", int8["avg_inference_ms_per_image"], " ms/image")
    print("\nDelta:")
    _line("mAP@50:95 delta", delta["map_50_95"])
    _line("Inference speedup", delta["speedup"], "x")
    print(f"\nSaved benchmark: {output}", flush=True)


if __name__ == "__main__":
    main()
