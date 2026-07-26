#!/usr/bin/env python3
"""Run ResNet50 TensorRT-backbone hybrid FP32 and INT8 detectors on a video side by side."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.fasterrcnn_qat.checkpoint import load_checkpoint
from pipelines.fasterrcnn_qat.config import choose_device, load_config, quantized_modules_for_variant, resolve_qat_profile
from pipelines.fasterrcnn_qat.models import build_fasterrcnn_model
from pipelines.fasterrcnn_qat.quantization import (
    mixed_precision_policy_from_config,
    module_qconfig_map_from_policy,
    policy_scope_to_quantized_modules,
    prepare_selective_qat,
)


COLORS = [(0, 220, 0), (0, 180, 255), (255, 120, 0), (255, 0, 180), (180, 255, 0)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_resnet50_hawq_compiler.yaml")
    parser.add_argument("--fp32-engine", required=True)
    parser.add_argument("--fp32-checkpoint", required=True)
    parser.add_argument("--int8-engine", required=True)
    parser.add_argument("--qat-checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs/video_resnet50_hybrid_fp32_vs_int8.mp4")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-frequency", type=int, default=10)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--labels", nargs="*", default=None)
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
        raise RuntimeError("pycuda is required for TensorRT video inference.") from error
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


class TensorRTBackboneRunner:
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
        self.context.set_tensor_address(self.input_name, int(batch_tensor.data_ptr()))
        for name, tensor in self.output_tensors.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT backbone execution failed.")
        self.stream.synchronize()
        return tuple(self.output_tensors[name] for name in self.output_names)


def load_hybrid_fp32_model(config, checkpoint, device):
    model = build_fasterrcnn_model(config)
    load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    return model.to(device).eval()


def load_hybrid_qat_model(config, checkpoint, device):
    model = build_fasterrcnn_model(config)
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


def preprocess_frame(model, frame_bgr, fixed_height, fixed_width, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = to_tensor(rgb).to(device)
    image_mean = torch.tensor(model.transform.image_mean, device=device).view(-1, 1, 1)
    image_std = torch.tensor(model.transform.image_std, device=device).view(-1, 1, 1)
    original_h, original_w = image.shape[-2:]
    normalized = (image - image_mean) / image_std
    resized = F.interpolate(
        normalized.unsqueeze(0),
        size=(fixed_height, fixed_width),
        mode="bilinear",
        align_corners=False,
    )
    image_list = ImageList(resized, [(fixed_height, fixed_width)])
    return image_list, (original_h, original_w)


@torch.inference_mode()
def hybrid_predict(model, backbone_runner, frame_bgr, fixed_height, fixed_width, device):
    image_list, original_size = preprocess_frame(model, frame_bgr, fixed_height, fixed_width, device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    c_features = backbone_runner(image_list.tensors)
    feature_dict = OrderedDict((str(index), tensor) for index, tensor in enumerate(c_features))
    features = model.backbone.fpn(feature_dict)
    proposals, _ = model.rpn(image_list, features, None)
    outputs, _ = model.roi_heads(features, proposals, image_list.image_sizes, None)
    outputs = model.transform.postprocess(outputs, image_list.image_sizes, [original_size])
    torch.cuda.synchronize(device)
    return outputs[0], (time.perf_counter() - started) * 1000.0


def annotate(frame, prediction, title, latency_ms, threshold, labels):
    canvas = frame.copy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    class_ids = prediction["labels"].detach().cpu().numpy()
    kept = 0
    for box, score, class_id in zip(boxes, scores, class_ids):
        if float(score) < threshold:
            continue
        kept += 1
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = COLORS[(int(class_id) - 1) % len(COLORS)]
        name = labels[int(class_id) - 1] if labels and 0 < int(class_id) <= len(labels) else f"class {class_id}"
        text = f"{name} {score:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 5, y1), color, -1)
        cv2.putText(canvas, text, (x1 + 2, max(th + 1, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    header = f"{title} | {latency_ms:.1f} ms | {1000.0 / latency_ms:.2f} FPS | detections={kept}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(canvas, header, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas, kept


def main():
    args = parse_args()
    if args.threads <= 0 or args.frame_stride <= 0 or args.progress_frequency <= 0:
        raise ValueError("threads, frame-stride and progress-frequency must be positive")
    torch.set_num_threads(args.threads)
    config = load_config(args.config, require_dataset=False)
    device = choose_device(config.get("device", "auto"))
    if device.type != "cuda":
        raise RuntimeError("This hybrid TensorRT video script requires CUDA")

    print(f"Loading FP32 hybrid model on {device}: {args.fp32_checkpoint}", flush=True)
    fp32_model = load_hybrid_fp32_model(config, args.fp32_checkpoint, device)
    print(f"Loading INT8/QAT hybrid model on {device}: {args.qat_checkpoint}", flush=True)
    int8_model = load_hybrid_qat_model(config, args.qat_checkpoint, device)
    print(f"Loading FP32 TensorRT engine: {args.fp32_engine}", flush=True)
    fp32_runner = TensorRTBackboneRunner(args.fp32_engine)
    print(f"Loading INT8 TensorRT engine: {args.int8_engine}", flush=True)
    int8_runner = TensorRTBackboneRunner(args.int8_engine)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    input_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_input = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = (total_input + args.frame_stride - 1) // args.frame_stride
    if args.max_frames is not None:
        expected = min(expected, args.max_frames)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        input_fps / args.frame_stride,
        (width * 2, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output}")

    timings = {"fp32": [], "int8": []}
    detections = {"fp32": 0, "int8": 0}
    input_index = processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (args.max_frames is not None and processed >= args.max_frames):
                break
            if input_index % args.frame_stride:
                input_index += 1
                continue
            input_index += 1
            fp32_prediction, fp32_ms = hybrid_predict(
                fp32_model, fp32_runner, frame, args.height, args.width, device
            )
            int8_prediction, int8_ms = hybrid_predict(
                int8_model, int8_runner, frame, args.height, args.width, device
            )
            fp32_frame, fp32_count = annotate(
                frame, fp32_prediction, "FP32 hybrid", fp32_ms, args.score_threshold, args.labels
            )
            int8_frame, int8_count = annotate(
                frame, int8_prediction, "INT8 hybrid", int8_ms, args.score_threshold, args.labels
            )
            writer.write(np.concatenate((fp32_frame, int8_frame), axis=1))
            timings["fp32"].append(fp32_ms)
            timings["int8"].append(int8_ms)
            detections["fp32"] += fp32_count
            detections["int8"] += int8_count
            processed += 1
            if processed == 1 or processed % args.progress_frequency == 0 or processed == expected:
                print(
                    f"progress={processed}/{expected} fp32={fp32_ms:.1f}ms int8={int8_ms:.1f}ms",
                    flush=True,
                )
    finally:
        capture.release()
        writer.release()

    if not processed:
        raise RuntimeError("No video frames were processed")
    result = {
        "video": str(Path(args.video).resolve()),
        "output": str(output.resolve()),
        "processed_frames": processed,
        "frame_stride": args.frame_stride,
        "score_threshold": args.score_threshold,
        "engine_shape": [int(args.height), int(args.width)],
        "fp32": {
            "device": str(device),
            "mean_latency_ms": float(np.mean(timings["fp32"])),
            "fps": 1000.0 / float(np.mean(timings["fp32"])),
            "detections": detections["fp32"],
        },
        "int8": {
            "device": str(device),
            "mean_latency_ms": float(np.mean(timings["int8"])),
            "fps": 1000.0 / float(np.mean(timings["int8"])),
            "detections": detections["int8"],
        },
    }
    metrics_path = Path(args.metrics_output) if args.metrics_output else output.with_suffix(".json")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved video: {output}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
