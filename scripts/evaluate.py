#!/usr/bin/env python3
"""Đánh giá checkpoint FP32 hoặc Selective-INT8 trên tập COCO val/test."""
import argparse
import sys
import warnings
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint, model_state_size_mb
from pipelines.convnext_qat.config import choose_device, load_config, quantized_modules_for_variant, validate_dataset_paths
from pipelines.convnext_qat.data import build_coco_loader
from pipelines.convnext_qat.metrics import evaluate_model, save_metrics
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import convert_selective_qat, prepare_selective_qat


def load_model(config, kind, checkpoint, device):
    """Dựng đúng topology FP32/INT8 rồi nạp checkpoint vào model."""
    model = build_fasterrcnn_convnext(config)
    if kind == "fp32":
        load_checkpoint(checkpoint, model)
        return model.to(device).eval()
    # Phải tái tạo topology quantized trước khi nạp packed parameters INT8.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("extra", {}) if isinstance(payload, dict) else {}
    variant = str(metadata.get("variant", config["quantization"].get("variant", "M3"))).upper()
    backend = metadata.get("backend", config["quantization"].get("backend", "x86"))
    quantized_modules = metadata.get(
        "quantized_modules", quantized_modules_for_variant(config, variant)
    )
    with warnings.catch_warnings():
        # Observer trống ở bước dựng topology là chủ ý; packed weight/scale được
        # nạp ngay sau đó từ checkpoint.
        warnings.filterwarnings("ignore", message="must run observer before calling calculate_qparams")
        model = convert_selective_qat(
            prepare_selective_qat(
                model, variant, backend, quantized_modules=quantized_modules
            )
        )
    load_checkpoint(checkpoint, model)
    return model.cpu().eval()


def parse_args():
    """Đọc model, split và giới hạn ảnh cần đánh giá."""
    parser = argparse.ArgumentParser(description="Evaluate FP32 or selective-INT8 detector")
    parser.add_argument("--config", default="configs/fasterrcnn_convnext_qat.yaml")
    parser.add_argument("--model", choices=["fp32", "int8"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    parser.add_argument("--skip-rpn-metrics", action="store_true")
    return parser.parse_args()


def main():
    """Chạy COCO evaluation, bổ sung kích thước model và lưu metrics."""
    args = parse_args()
    config = load_config(args.config)
    validate_dataset_paths(config, (args.split,))
    device = choose_device(config.get("device", "auto")) if args.model == "fp32" else choose_device("cpu")
    checkpoint = args.checkpoint or config["output"]["fp32_best" if args.model == "fp32" else "int8_model"]
    loader = build_coco_loader(config, args.split, shuffle=False, limit=args.limit)
    model = load_model(config, args.model, checkpoint, device)
    base_model = model
    metrics = evaluate_model(
        model, loader, device,
        include_rpn=not args.skip_rpn_metrics,
    )
    metrics["model_size_mb"] = model_state_size_mb(base_model)
    metrics["parameters"] = int(base_model.logical_parameter_count)
    metrics["model"] = args.model
    if args.output:
        output = args.output
    else:
        base_output = Path(config["output"]["evaluation_json"])
        output = str(base_output.with_name(f"{base_output.stem}_{args.model}{base_output.suffix}"))
    save_metrics(output, metrics)
    print(metrics)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
