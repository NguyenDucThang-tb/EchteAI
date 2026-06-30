#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from EchteAI.pipelines.convnext_qat.checkpoint import load_checkpoint, save_checkpoint
from EchteAI.pipelines.convnext_qat.config import choose_device, load_config, quantized_modules_for_variant
from EchteAI.pipelines.convnext_qat.data import build_coco_loader
from EchteAI.pipelines.convnext_qat.engine import make_optimizer, set_optimizer_lr, train_one_epoch
from EchteAI.pipelines.convnext_qat.metrics import evaluate_model
from EchteAI.pipelines.convnext_qat.models import build_fasterrcnn_convnext
from EchteAI.pipelines.convnext_qat.quantization import convert_selective_qat, prepare_selective_qat, set_qat_phase


def parse_args():
    parser = argparse.ArgumentParser(description="Selective QAT for Faster R-CNN ConvNeXt-FPN")
    parser.add_argument("--config", default="configs/fasterrcnn_convnext_qat.yaml")
    parser.add_argument("--fp32-checkpoint", help="override output.fp32_best")
    parser.add_argument("--variant", choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", help="resume a prepared-QAT checkpoint")
    return parser.parse_args()


@torch.no_grad()
def observer_warmup(model, loader, device, image_count):
    model.eval()
    set_qat_phase(model, "calibration")
    observed = 0
    for images, _ in loader:
        model([image.to(device) for image in images])
        observed += len(images)
        if observed >= image_count:
            break
    model.train()
    return observed


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=True)
    variant = args.variant or str(config["quantization"].get("variant", "M3")).upper()
    backend = config["quantization"].get("backend", "x86")
    device = choose_device(config.get("device", "auto"))
    train_loader = build_coco_loader(config, "train", limit=args.limit)
    val_loader = build_coco_loader(config, "val", shuffle=False, limit=args.limit)
    quantized_modules = quantized_modules_for_variant(config, variant)

    fp32_model = build_fasterrcnn_convnext(config)
    load_checkpoint(args.fp32_checkpoint or config["output"]["fp32_best"], fp32_model)
    qat_model = prepare_selective_qat(
        fp32_model, variant, backend, quantized_modules=quantized_modules
    ).to(device)
    optimizer = make_optimizer(qat_model, config, qat=True)
    total_epochs = int(config["training"]["qat_epochs"])
    weight_only_epochs = int(config["quantization"].get("weight_only_warmup_epochs", 1))
    freeze_epochs = int(config["quantization"].get("observer_freeze_epochs", 2))
    if weight_only_epochs + freeze_epochs > total_epochs:
        raise ValueError("weight-only plus observer-freeze epochs cannot exceed QAT epochs")

    best_map = -1.0
    best_saved = Path(config["output"]["qat_best"]).is_file()
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(args.resume, qat_model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", 0))
        best_map = float(payload.get("extra", {}).get("best_map", -1.0))
        print(f"resumed QAT checkpoint={args.resume} epoch={start_epoch}")
    else:
        observed_images = observer_warmup(
            qat_model, train_loader, device,
            int(config["quantization"].get("calibration_images", 256)),
        )
        print(f"observer calibration images={observed_images}")
        if observed_images < int(config["quantization"].get("calibration_images", 256)):
            print("warning: calibration dataset ended before the requested image count")
    phase_lrs = config["quantization"].get("phase_learning_rates", {})
    for epoch in range(start_epoch, total_epochs):
        if epoch < weight_only_epochs:
            phase = "weight_only"
        elif epoch >= total_epochs - freeze_epochs:
            phase = "frozen"
        else:
            phase = "full"
        set_qat_phase(qat_model, phase)
        set_optimizer_lr(optimizer, phase_lrs.get(phase, config["training"]["qat_lr"]))
        train_metrics = train_one_epoch(
            qat_model, train_loader, optimizer, device,
            float(config["training"].get("grad_clip_norm", 0)),
            int(config["training"].get("print_frequency", 20)),
        )
        val_metrics = evaluate_model(qat_model, val_loader, device, include_rpn=False)
        print(
            f"qat_epoch={epoch + 1}/{total_epochs} phase={phase} "
            f"train={train_metrics} validation={val_metrics}"
        )
        # Convert only a checkpoint trained with fixed observer ranges. Earlier
        # phases are useful diagnostics but are not eligible for final INT8.
        if phase == "frozen" and (not best_saved or val_metrics["map_50_95"] > best_map):
            best_map = val_metrics["map_50_95"]
            best_saved = True
            save_checkpoint(
                config["output"]["qat_best"], qat_model, optimizer, epoch + 1,
                val_metrics,
                {"variant": variant, "backend": backend, "format": "prepared_qat", "quantized_modules": quantized_modules or [], "best_map": best_map},
            )
        save_checkpoint(
            config["output"]["qat_last"], qat_model, optimizer, epoch + 1, val_metrics,
            {"variant": variant, "backend": backend, "format": "prepared_qat", "quantized_modules": quantized_modules or [], "best_map": best_map},
        )
    if not best_saved:
        raise ValueError("QAT schedule has no frozen-observer epoch eligible for best checkpoint selection")
    load_checkpoint(config["output"]["qat_best"], qat_model)
    int8_model = convert_selective_qat(qat_model.to("cpu"))
    save_checkpoint(
        config["output"]["int8_model"], int8_model,
        metrics={"best_qat_map_50_95": best_map},
        extra={"variant": variant, "backend": backend, "format": "selective_int8", "quantized_modules": quantized_modules or []},
    )
    print(f"Converted selective INT8 checkpoint: {config['output']['int8_model']}")


if __name__ == "__main__":
    main()
