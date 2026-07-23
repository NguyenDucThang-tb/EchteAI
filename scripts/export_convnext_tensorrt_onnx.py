#!/usr/bin/env python3
"""Export ConvNeXt selective-QAT backbone/FPN sang ONNX cho TensorRT.

Voi ``--model fp32`` script export backbone FP32. Voi ``--model qat_graph``
script dung checkpoint QAT frozen, giu fake-quant trong graph de ONNX co Q/DQ
ma TensorRT co the build engine INT8 tren GPU.
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

from pipelines.convnext_qat.checkpoint import load_checkpoint
from pipelines.convnext_qat.compiler import build_compiler_target_module, resolve_compiler_scope
from pipelines.convnext_qat.config import load_config, quantized_modules_for_variant
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import prepare_selective_qat, set_qat_phase


def parse_args():
    """Doc checkpoint nguon va kich thuoc engine co dinh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--model", choices=["fp32", "qat_graph"], default="fp32")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--qat-checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tensorrt-friendly-int8", action="store_true")
    return parser.parse_args()


def normalize_qdq_zero_points_for_tensorrt(onnx_path: Path):
    """Dua zero-point cua Q/DQ ve 0 de giam loi parse tren TensorRT INT8."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path), load_external_data=True)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    touched = 0
    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"} or len(node.input) < 3:
            continue
        initializer = initializers.get(node.input[2])
        if initializer is None:
            continue
        array = numpy_helper.to_array(initializer)
        replacement = numpy_helper.from_array(np.zeros_like(array), name=initializer.name)
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


def load_source_model(config, model_kind, fp32_checkpoint=None, qat_checkpoint=None):
    """Dung model FP32 hoac dung topology selective-QAT de nap QAT checkpoint."""
    model = build_fasterrcnn_convnext(config).cpu().eval()
    if model_kind == "fp32":
        checkpoint = fp32_checkpoint or config["output"].get("fp32_best")
        if checkpoint and Path(checkpoint).is_file():
            print(f"Loading FP32 checkpoint: {checkpoint}", flush=True)
            payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
        else:
            print("No FP32 checkpoint loaded; exporting current model weights.", flush=True)
            payload = {}
        return model, payload

    checkpoint = qat_checkpoint or config["output"].get("qat_best") or config["output"].get("qat_last")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("QAT graph export requires --qat-checkpoint or output.qat_best/output.qat_last")
    print(f"Loading selective QAT checkpoint: {checkpoint}", flush=True)
    raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = raw_payload.get("extra", {}) if isinstance(raw_payload, dict) else {}
    variant = str(metadata.get("variant", config["quantization"].get("variant", "M3"))).upper()
    backend = metadata.get("backend", config["quantization"].get("backend", "auto"))
    quantized_modules = metadata.get("quantized_modules", quantized_modules_for_variant(config, variant))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="must run observer before calling calculate_qparams")
        model = prepare_selective_qat(
            model, variant, backend, quantized_modules=quantized_modules,
        )
    payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    set_qat_phase(model, "frozen")
    return model.cpu().eval(), payload


def main():
    """Export target module va luu metadata dung cho buoc build TensorRT."""
    args = parse_args()
    config = load_config(args.config, require_dataset=False)
    compiler_cfg = config.get("quantization", {}).get("compiler", {})
    scope = resolve_compiler_scope(config)
    batch_size = int(args.batch_size or compiler_cfg.get("example_batch_size", 1))
    height = int(args.height or compiler_cfg.get("example_height", config["model"].get("min_size", 960)))
    width = int(args.width or compiler_cfg.get("example_width", config["model"].get("max_size", 1600)))

    artifact_dir = Path(
        args.artifact_dir
        or compiler_cfg.get("artifact_dir")
        or Path(config["output"]["directory"]) / "tensorrt_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_source_model(
        config,
        args.model,
        fp32_checkpoint=args.fp32_checkpoint,
        qat_checkpoint=args.qat_checkpoint,
    )
    target_module = build_compiler_target_module(model, config).cpu().eval()
    sample = torch.randn(batch_size, 3, height, width)
    with torch.inference_mode():
        outputs = target_module(sample)
    output_names = [f"output_{index}" for index in range(len(outputs))]

    onnx_path = Path(args.output) if args.output else artifact_dir / f"convnext_{scope}_{args.model}.onnx"
    metadata_path = onnx_path.with_name(f"{onnx_path.stem}_metadata.json")
    # Với checkpoint QAT eager, các module FakeQuantize vẫn chứa nhánh:
    # ``if self.observer_enabled[0] == 1``. Exporter mới dựa trên
    # ``torch.export`` coi điều kiện này là data-dependent guard và có thể lỗi
    # ``GuardOnDataDependentSymNode`` dù observer đã bị freeze. Legacy tracer
    # sẽ hằng-hoá trạng thái frozen hiện tại và xuất các fake-quant op sang
    # ONNX ổn định hơn cho TensorRT Q/DQ workflow.
    torch.onnx.export(
        target_module,
        (sample,),
        str(onnx_path),
        input_names=["input0"],
        output_names=output_names,
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamo=False,
    )

    normalized_zero_points = 0
    if args.model == "qat_graph" and args.tensorrt_friendly_int8:
        normalized_zero_points = normalize_qdq_zero_points_for_tensorrt(onnx_path)
        print(f"Normalized {normalized_zero_points} Q/DQ zero-point initializer(s).", flush=True)

    metadata = {
        "model_kind": args.model,
        "backbone": config["model"].get("backbone", "unknown"),
        "scope": scope,
        "onnx_path": str(onnx_path),
        "input_name": "input0",
        "output_names": output_names,
        "example_shape": list(sample.shape),
        "checkpoint_extra": payload.get("extra", {}) if isinstance(payload, dict) else {},
        "tensorrt_friendly_int8": bool(args.tensorrt_friendly_int8),
        "normalized_zero_points": int(normalized_zero_points),
        "opset": int(args.opset),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved ONNX model: {onnx_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
