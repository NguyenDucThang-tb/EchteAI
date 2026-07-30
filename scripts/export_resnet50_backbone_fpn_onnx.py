#!/usr/bin/env python3
"""Export a ResNet50 Faster R-CNN backbone in full FPN interface form.

This exports the same tensor interface returned by ``model.backbone(...)`` for
the ResNet50 + FPN detector:

- P2 / key "0"
- P3 / key "1"
- P4 / key "2"
- P5 / key "3"
- P6 / key "pool"

All outputs are 256-channel FPN-ready tensors, which allows downstream hybrid
execution to connect them directly into RPN + ROI heads without rebuilding FPN
inside the PyTorch half.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.fasterrcnn_qat.checkpoint import load_checkpoint, load_partial_checkpoint
from pipelines.fasterrcnn_qat.compiler import build_compiler_target_module, resolve_compiler_scope
from pipelines.fasterrcnn_qat.config import load_config, quantized_modules_for_variant, resolve_qat_profile
from pipelines.fasterrcnn_qat.models import build_fasterrcnn_model
from pipelines.fasterrcnn_qat.quantization import (
    mixed_precision_policy_from_config,
    module_qconfig_map_from_policy,
    policy_scope_to_quantized_modules,
    prepare_selective_qat,
    set_qat_phase,
)


FEATURE_NAMES = ["p2", "p3", "p4", "p5", "p6_pool"]


def _resolve_quantized_modules(config, metadata, variant):
    quantized_modules = metadata.get("quantized_modules")
    if quantized_modules == []:
        quantized_modules = None
    if quantized_modules is None:
        quantized_modules = quantized_modules_for_variant(config, variant)
    return quantized_modules


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["fp32", "qat_graph"], default="qat_graph")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--qat-checkpoint")
    parser.add_argument("--partial-fp32-checkpoint", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--force-w8a8", action="store_true")
    parser.add_argument("--tensorrt-friendly-int8", action="store_true")
    parser.add_argument("--dynamic-hw", action="store_true", help="Export ONNX with dynamic height/width axes")
    return parser.parse_args()


def normalize_qdq_zero_points_for_tensorrt(onnx_path: Path):
    import onnx
    from onnx import numpy_helper
    import numpy as np

    model = onnx.load(str(onnx_path), load_external_data=True)
    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    touched = 0

    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        if len(node.input) < 3:
            continue
        zero_point_name = node.input[2]
        initializer = initializer_by_name.get(zero_point_name)
        if initializer is None:
            continue
        array = numpy_helper.to_array(initializer)
        if array.size == 0:
            continue
        zeroed = np.zeros_like(array)
        replacement = numpy_helper.from_array(zeroed, name=initializer.name)
        initializer.CopyFrom(replacement)
        touched += 1

    if touched:
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{onnx_path.name}.data",
            size_threshold=1024,
            convert_attribute=False,
        )
    return touched


def load_source_model(
    config,
    model_kind,
    fp32_checkpoint=None,
    qat_checkpoint=None,
    force_w8a8=False,
    partial_fp32_checkpoint=False,
):
    model = build_fasterrcnn_model(config).cpu().eval()
    if model_kind == "fp32":
        checkpoint = fp32_checkpoint or config["output"].get("fp32_best")
        if checkpoint and Path(checkpoint).is_file():
            print(f"Loading FP32 checkpoint: {checkpoint}", flush=True)
            if partial_fp32_checkpoint:
                payload = load_partial_checkpoint(checkpoint, model, map_location="cpu")
                print(
                    "Partial FP32 load summary: "
                    f"matched={payload.get('extra', {}).get('matched_key_count', 0)} "
                    f"missing={payload.get('extra', {}).get('missing_key_count', 0)} "
                    f"unexpected={payload.get('extra', {}).get('unexpected_key_count', 0)} "
                    f"shape_mismatches={payload.get('extra', {}).get('shape_mismatch_count', 0)}",
                    flush=True,
                )
            else:
                payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
        else:
            print("No FP32 checkpoint loaded; exporting current model weights.", flush=True)
            payload = {}
        return model, payload

    checkpoint = qat_checkpoint or config["output"].get("qat_best") or config["output"].get("qat_last")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("QAT graph export requires --qat-checkpoint or output.qat_best/output.qat_last")
    print(f"Loading QAT graph checkpoint: {checkpoint}", flush=True)
    raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = raw_payload.get("extra", {}) if isinstance(raw_payload, dict) else {}
    variant = str(metadata.get("variant", config["quantization"].get("variant", "M3"))).upper()
    backend = metadata.get("backend", config["quantization"].get("backend", "x86"))
    qat_profile = resolve_qat_profile(config, metadata)
    quantized_modules = _resolve_quantized_modules(config, metadata, variant)
    mixed_precision_policy = None if force_w8a8 else (
        metadata.get("mixed_precision_policy") or mixed_precision_policy_from_config(config)
    )
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
    payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    set_qat_phase(model, "frozen")
    return model.cpu().eval(), payload


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=False)
    config.setdefault("quantization", {}).setdefault("compiler", {})
    config["quantization"]["compiler"]["scope"] = "backbone_fpn"
    compiler_cfg = config["quantization"]["compiler"]
    scope = resolve_compiler_scope(config)

    batch_size = int(args.batch_size or compiler_cfg.get("example_batch_size", 1))
    height = int(args.height or compiler_cfg.get("example_height", config["model"].get("min_size", 1080)))
    width = int(args.width or compiler_cfg.get("example_width", config["model"].get("max_size", 1920)))

    model, payload = load_source_model(
        config,
        args.model,
        fp32_checkpoint=args.fp32_checkpoint,
        qat_checkpoint=args.qat_checkpoint,
        force_w8a8=args.force_w8a8,
        partial_fp32_checkpoint=args.partial_fp32_checkpoint,
    )

    target_module = build_compiler_target_module(model, config).cpu().eval()
    sample = torch.randn(batch_size, 3, height, width)

    with torch.inference_mode():
        outputs = target_module(sample)

    if len(outputs) != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Expected {len(FEATURE_NAMES)} backbone FPN outputs, got {len(outputs)}"
        )

    onnx_path = Path(args.output)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = onnx_path.with_name(f"{onnx_path.stem}_metadata.json")

    export_kwargs = dict(
        input_names=["input0"],
        output_names=FEATURE_NAMES,
        opset_version=int(args.opset),
        do_constant_folding=True,
    )
    if args.dynamic_hw:
        export_kwargs["dynamic_axes"] = {
            "input0": {2: "height", 3: "width"},
            "p2": {2: "p2_height", 3: "p2_width"},
            "p3": {2: "p3_height", 3: "p3_width"},
            "p4": {2: "p4_height", 3: "p4_width"},
            "p5": {2: "p5_height", 3: "p5_width"},
            "p6_pool": {2: "p6_height", 3: "p6_width"},
        }
    torch.onnx.export(
        target_module,
        (sample,),
        str(onnx_path),
        dynamo=False,
        **export_kwargs,
    )

    normalized_zero_points = 0
    if args.model == "qat_graph" and args.tensorrt_friendly_int8:
        normalized_zero_points = normalize_qdq_zero_points_for_tensorrt(onnx_path)
        print(
            f"Normalized {normalized_zero_points} Q/DQ zero-point initializer(s) to zero for TensorRT compatibility.",
            flush=True,
        )

    metadata = {
        "model_kind": args.model,
        "backbone": config["model"].get("backbone", "unknown"),
        "scope": scope,
        "interface": "model.backbone(...)",
        "onnx_path": str(onnx_path),
        "input_name": "input0",
        "output_names": FEATURE_NAMES,
        "example_shape": list(sample.shape),
        "output_shapes": [list(tensor.shape) for tensor in outputs],
        "checkpoint_extra": payload.get("extra", {}) if isinstance(payload, dict) else {},
        "qat_profile": payload.get("extra", {}).get("qat_profile", resolve_qat_profile(config)) if isinstance(payload, dict) else resolve_qat_profile(config),
        "force_w8a8": bool(args.force_w8a8),
        "tensorrt_friendly_int8": bool(args.tensorrt_friendly_int8),
        "dynamic_hw": bool(args.dynamic_hw),
        "normalized_zero_points": int(normalized_zero_points),
        "opset": int(args.opset),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved ONNX model: {onnx_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
