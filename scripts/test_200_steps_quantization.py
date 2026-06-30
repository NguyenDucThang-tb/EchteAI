#!/usr/bin/env python3
"""Train/benchmark a 200-step FP32 checkpoint, then smoke-test QAT and INT8."""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from EchteAI.pipelines.convnext_qat.checkpoint import load_checkpoint, save_checkpoint
from EchteAI.pipelines.convnext_qat.config import (
    choose_device, load_config, quantized_modules_for_variant,
)
from EchteAI.pipelines.convnext_qat.data import build_coco_loader
from EchteAI.pipelines.convnext_qat.engine import (
    benchmark_inference, make_optimizer, train_one_epoch,
)
from EchteAI.pipelines.convnext_qat.models import build_fasterrcnn_convnext
from EchteAI.pipelines.convnext_qat.quantization import (
    convert_selective_qat, prepare_selective_qat, set_qat_phase,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--benchmark-images", type=int, default=100)
    parser.add_argument("--calibration-images", type=int, default=100)
    parser.add_argument("--variant", choices=["M0", "M1", "M2", "M3", "M4"])
    return parser.parse_args()


@torch.inference_mode()
def calibrate(model, loader, device, image_count):
    model.eval()
    set_qat_phase(model, "calibration")
    observed = 0
    for images, _ in loader:
        remaining = image_count - observed
        if remaining <= 0:
            break
        images = [image.to(device) for image in images[:remaining]]
        model(images)
        observed += len(images)
        if observed % 25 < len(images) or observed >= image_count:
            print(f"calibration progress: {observed}/{image_count}", flush=True)
    return observed


def main():
    args = parse_args()
    if min(args.steps, args.benchmark_images, args.calibration_images) <= 0:
        raise ValueError("steps and image counts must be positive")

    config = load_config(args.config, require_dataset=True)
    device = choose_device(config.get("device", "auto"))
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    fp32_path = output / f"fp32_{args.steps}_steps.pt"
    qat_path = output / f"qat_{args.steps}_steps.pt"
    int8_path = output / f"int8_{args.steps}_steps.pt"
    results_path = output / f"test_{args.steps}_steps_results.json"

    train_loader = build_coco_loader(config, "train")
    val_loader = build_coco_loader(config, "val", shuffle=False)
    print(
        f"Smoke test device={device} steps={args.steps} "
        f"benchmark_images={args.benchmark_images} output={output}",
        flush=True,
    )

    fp32_model = build_fasterrcnn_convnext(config).to(device)
    fp32_optimizer = make_optimizer(fp32_model, config)
    fp32_train = train_one_epoch(
        fp32_model, train_loader, fp32_optimizer, device,
        float(config["training"].get("grad_clip_norm", 0)),
        int(config["training"].get("print_frequency", 50)),
        max_steps=args.steps,
    )
    fp32_benchmark = benchmark_inference(
        fp32_model, val_loader, device, args.benchmark_images,
    )
    save_checkpoint(
        fp32_path, fp32_model, fp32_optimizer, metrics={
            "train": fp32_train, "benchmark": fp32_benchmark,
        }, extra={"format": "fp32", "steps": args.steps},
    )
    print(f"saved {args.steps}-step FP32 checkpoint: {fp32_path}", flush=True)

    del fp32_model, fp32_optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    variant = args.variant or str(config["quantization"].get("variant", "M3")).upper()
    backend = config["quantization"].get("backend", "x86")
    quantized_modules = quantized_modules_for_variant(config, variant)
    base_model = build_fasterrcnn_convnext(config)
    load_checkpoint(fp32_path, base_model)
    qat_model = prepare_selective_qat(
        base_model, variant, backend, quantized_modules=quantized_modules,
    ).to(device)
    del base_model
    observed = calibrate(qat_model, train_loader, device, args.calibration_images)
    set_qat_phase(qat_model, "full")
    qat_optimizer = make_optimizer(qat_model, config, qat=True)
    qat_train = train_one_epoch(
        qat_model, train_loader, qat_optimizer, device,
        float(config["training"].get("grad_clip_norm", 0)),
        int(config["training"].get("print_frequency", 50)),
        max_steps=args.steps,
    )
    set_qat_phase(qat_model, "frozen")
    qat_benchmark = benchmark_inference(
        qat_model, val_loader, device, args.benchmark_images,
    )
    save_checkpoint(
        qat_path, qat_model, qat_optimizer, metrics={
            "train": qat_train, "benchmark": qat_benchmark,
        }, extra={
            "format": "prepared_qat", "steps": args.steps, "variant": variant,
            "backend": backend, "quantized_modules": quantized_modules or [],
        },
    )
    print(f"saved {args.steps}-step QAT checkpoint: {qat_path}", flush=True)

    int8_model = convert_selective_qat(qat_model.to("cpu"))
    save_checkpoint(
        int8_path, int8_model,
        extra={
            "format": "selective_int8", "variant": variant, "backend": backend,
            "quantized_modules": quantized_modules or [],
        },
    )
    int8_benchmark = benchmark_inference(
        int8_model, val_loader, torch.device("cpu"), args.benchmark_images,
    )
    results = {
        "steps": args.steps,
        "calibration_images": observed,
        "fp32": fp32_benchmark,
        "qat_fake_quant": qat_benchmark,
        "int8": int8_benchmark,
        "checkpoints": {
            "fp32": str(fp32_path), "qat": str(qat_path), "int8": str(int8_path),
        },
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"saved results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
