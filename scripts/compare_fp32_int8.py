#!/usr/bin/env python3
"""Fair CPU comparison of FP32 and selective-INT8 detection checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_model
from EchteAI.pipelines.convnext_qat.checkpoint import model_state_size_mb
from EchteAI.pipelines.convnext_qat.config import load_config
from EchteAI.pipelines.convnext_qat.data import build_coco_loader
from EchteAI.pipelines.convnext_qat.engine import benchmark_inference
from EchteAI.pipelines.convnext_qat.metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--fp32-checkpoint", required=True)
    parser.add_argument("--int8-checkpoint", required=True)
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def evaluate_one(config, kind, checkpoint, loader, device, images):
    print(f"Loading {kind.upper()} checkpoint: {checkpoint}", flush=True)
    model = load_model(config, kind, checkpoint, device)
    metrics = evaluate_model(model, loader, device, include_rpn=False)
    timing = benchmark_inference(model, loader, device, images)
    return {
        "device": str(device),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mean_iou": metrics["mean_iou"],
        "map_50_95": metrics["map_50_95"],
        "map_50": metrics["map_50"],
        "avg_inference_ms_per_image": timing["latency_ms_per_image"],
        "fps": timing["fps"],
        "backbone_size_mb": model_state_size_mb(model.backbone),
        "full_model_size_mb": model_state_size_mb(model),
    }


def line(label, value, suffix=""):
    print(f"  {label}: {value:.4f}{suffix}")


def main():
    args = parse_args()
    if args.images <= 0 or args.threads <= 0:
        raise ValueError("images and threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    config = load_config(args.config, require_dataset=True)
    # A batch of one gives per-image CPU latency and avoids batch-dependent bias.
    loader = build_coco_loader(
        config, "val", shuffle=False, limit=args.images, batch_size=1,
    )
    fp32 = evaluate_one(
        config, "fp32", args.fp32_checkpoint, loader, device, args.images,
    )
    int8 = evaluate_one(
        config, "int8", args.int8_checkpoint, loader, device, args.images,
    )
    delta = {
        "accuracy": int8["accuracy"] - fp32["accuracy"],
        "precision": int8["precision"] - fp32["precision"],
        "mean_iou": int8["mean_iou"] - fp32["mean_iou"],
        "map_50_95": int8["map_50_95"] - fp32["map_50_95"],
        "speedup": fp32["avg_inference_ms_per_image"] / int8["avg_inference_ms_per_image"],
        "backbone_reduction_percent": 100.0 * (1.0 - int8["backbone_size_mb"] / fp32["backbone_size_mb"]),
        "full_model_reduction_percent": 100.0 * (1.0 - int8["full_model_size_mb"] / fp32["full_model_size_mb"]),
    }
    results = {"images": args.images, "threads": args.threads, "fp32": fp32, "int8": int8, "delta": delta}
    output = Path(args.output or Path(config["output"]["directory"]) / "fp32_int8_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nModel size:")
    line("FP32 backbone", fp32["backbone_size_mb"], " MB")
    line("INT8 backbone", int8["backbone_size_mb"], " MB")
    line("Backbone reduction", delta["backbone_reduction_percent"], "%")
    line("FP32 full model", fp32["full_model_size_mb"], " MB")
    line("INT8 full model", int8["full_model_size_mb"], " MB")
    line("Full model reduction", delta["full_model_reduction_percent"], "%")
    for name, values in (("FP32", fp32), ("INT8", int8)):
        print(f"\n{name}:")
        print(f"  Device: {values['device']}")
        line("Accuracy", values["accuracy"])
        line("Precision", values["precision"])
        line("Mean IoU", values["mean_iou"])
        line("mAP@50:95", values["map_50_95"])
        line("Avg inference", values["avg_inference_ms_per_image"], " ms/image")
    print("\nDelta:")
    line("Accuracy delta", delta["accuracy"])
    line("Precision delta", delta["precision"])
    line("Mean IoU delta", delta["mean_iou"])
    line("mAP@50:95 delta", delta["map_50_95"])
    line("Inference speedup", delta["speedup"], "x")
    print(f"\nSaved comparison: {output}", flush=True)


if __name__ == "__main__":
    main()
