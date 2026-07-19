#!/usr/bin/env python3
"""Train bù sai số Selective Eager QAT trên một GPU và convert INT8."""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint, save_checkpoint
from pipelines.convnext_qat.config import choose_device, load_config, quantized_modules_for_variant
from pipelines.convnext_qat.data import build_coco_loader
from pipelines.convnext_qat.engine import (
    append_epoch_benchmark, benchmark_inference, make_optimizer, set_optimizer_lr,
    train_one_epoch,
)
from pipelines.convnext_qat.metrics import evaluate_model
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import convert_selective_qat, prepare_selective_qat, set_qat_phase


def parse_args():
    """Đọc FP32 source, QAT resume, variant và số epoch cần chạy."""
    parser = argparse.ArgumentParser(description="Selective QAT for Faster R-CNN ConvNeXt-FPN")
    parser.add_argument("--config", default="configs/fasterrcnn_convnext_qat.yaml")
    parser.add_argument("--fp32-checkpoint", help="override output.fp32_best")
    parser.add_argument("--variant", choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", help="resume a prepared-QAT checkpoint")
    parser.add_argument(
        "--epochs-this-run", type=int,
        help="stop after this many epochs; useful for short Colab sessions",
    )
    return parser.parse_args()


@torch.no_grad()
def observer_warmup(model, loader, device, image_count):
    """Thu range activation trên một số ảnh trước phase fake-quant."""
    print(f"observer calibration started target={image_count} device={device}", flush=True)
    model.eval()
    set_qat_phase(model, "calibration")
    observed = 0
    for images, _ in loader:
        model([image.to(device) for image in images])
        observed += len(images)
        if observed % 25 < len(images) or observed >= image_count:
            print(f"observer calibration {observed}/{image_count} images", flush=True)
        if observed >= image_count:
            break
    model.train()
    return observed


def main():
    """Prepare selective QAT, chạy lịch phase, validate và lưu INT8 cuối."""
    args = parse_args()
    config = load_config(args.config, require_dataset=True)
    variant = args.variant or str(config["quantization"].get("variant", "M3")).upper()
    backend = config["quantization"].get("backend", "x86")
    device = choose_device(config.get("device", "auto"))
    qat_batch_size = config["training"].get("qat_batch_size", config["training"]["batch_size"])
    train_loader = build_coco_loader(
        config, "train", limit=args.limit, batch_size=qat_batch_size,
    )
    val_loader = build_coco_loader(
        config, "val", shuffle=False, limit=args.limit, batch_size=qat_batch_size,
    )
    quantized_modules = quantized_modules_for_variant(config, variant)
    print(
        f"QAT setup device={device} variant={variant} "
        f"train_images={len(train_loader.dataset)} val_images={len(val_loader.dataset)}",
        flush=True,
    )
    print(f"quantized_modules={quantized_modules}", flush=True)

    source_checkpoint = args.fp32_checkpoint or config["output"]["fp32_best"]
    print("building FP32 model topology...", flush=True)
    fp32_model = build_fasterrcnn_convnext(config)
    print(f"loading FP32 checkpoint={source_checkpoint}", flush=True)
    load_checkpoint(source_checkpoint, fp32_model)
    print("FP32 checkpoint loaded; preparing selective QAT model...", flush=True)
    qat_model = prepare_selective_qat(
        fp32_model, variant, backend, quantized_modules=quantized_modules
    ).to(device)
    backend = qat_model.quantized_backend
    print("selective QAT model prepared and moved to device", flush=True)
    optimizer = make_optimizer(qat_model, config, qat=True)
    print(f"QAT optimizer ready lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
    total_epochs = int(config["training"]["qat_epochs"])
    if args.epochs_this_run is not None and args.epochs_this_run <= 0:
        raise ValueError("--epochs-this-run must be positive")
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
    end_epoch = total_epochs
    if args.epochs_this_run is not None:
        end_epoch = min(start_epoch + args.epochs_this_run, total_epochs)
    for epoch in range(start_epoch, end_epoch):
        if epoch < weight_only_epochs:
            phase = "weight_only"
        elif epoch >= total_epochs - freeze_epochs:
            phase = "frozen"
        else:
            phase = "full"
        set_qat_phase(qat_model, phase)
        set_optimizer_lr(optimizer, phase_lrs.get(phase, config["training"]["qat_lr"]))
        print(
            f"QAT epoch={epoch + 1}/{total_epochs} phase={phase} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}",
            flush=True,
        )
        train_metrics = train_one_epoch(
            qat_model, train_loader, optimizer, device,
            float(config["training"].get("grad_clip_norm", 0)),
            int(config["training"].get("print_frequency", 20)),
        )
        # QAT has no epoch scheduler, so this prepared-model checkpoint can be
        # resumed directly even when validation or benchmarking later fails.
        save_checkpoint(
            config["output"]["qat_last"], qat_model, optimizer, epoch + 1,
            {"train": train_metrics, "validation_pending": True},
            {"variant": variant, "backend": backend, "format": "prepared_qat", "quantized_modules": quantized_modules or [], "best_map": best_map},
        )
        print(f"saved pre-validation QAT checkpoint: {config['output']['qat_last']}", flush=True)
        print("QAT validation started", flush=True)
        val_metrics = evaluate_model(
            qat_model, val_loader, device, include_rpn=True,
        )
        print("QAT validation completed", flush=True)
        benchmark_metrics = benchmark_inference(
            qat_model, val_loader, device,
            int(config["training"].get("epoch_benchmark_images", 100)),
        )
        benchmark_record = {
            "stage": "qat", "epoch": epoch + 1, "phase": phase, **benchmark_metrics,
        }
        benchmark_history = config["output"].get(
            "epoch_benchmarks",
            str(Path(config["output"]["directory"]) / "epoch_benchmarks.json"),
        )
        append_epoch_benchmark(benchmark_history, benchmark_record)
        print(f"QAT epoch benchmark={benchmark_record}", flush=True)
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
                {**val_metrics, "benchmark": benchmark_metrics},
                {"variant": variant, "backend": backend, "format": "prepared_qat", "quantized_modules": quantized_modules or [], "best_map": best_map},
            )
            print(f"saved new QAT best: {config['output']['qat_best']}", flush=True)
        save_checkpoint(
            config["output"]["qat_last"], qat_model, optimizer, epoch + 1,
            {**val_metrics, "benchmark": benchmark_metrics},
            {"variant": variant, "backend": backend, "format": "prepared_qat", "quantized_modules": quantized_modules or [], "best_map": best_map},
        )
        print(f"saved QAT resume checkpoint: {config['output']['qat_last']}", flush=True)
    print(
        f"QAT run completed at epoch {end_epoch}/{total_epochs}. "
        f"Resume checkpoint: {config['output']['qat_last']}",
        flush=True,
    )
    if end_epoch < total_epochs:
        return
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
