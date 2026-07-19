#!/usr/bin/env python3
"""Train Faster R-CNN FP32 trên một GPU/CPU và lưu checkpoint mỗi epoch."""
import argparse
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint, save_checkpoint
from pipelines.convnext_qat.config import choose_device, load_config
from pipelines.convnext_qat.data import build_coco_loader
from pipelines.convnext_qat.engine import (
    append_epoch_benchmark, benchmark_inference, make_optimizer, train_one_epoch,
)
from pipelines.convnext_qat.metrics import evaluate_model
from pipelines.convnext_qat.models import build_fasterrcnn_convnext


def parse_args():
    """Đọc config, resume checkpoint và giới hạn dữ liệu."""
    parser = argparse.ArgumentParser(description="Train FP32 Faster R-CNN ConvNeXt-FPN")
    parser.add_argument("--config", default="configs/fasterrcnn_convnext_qat.yaml")
    parser.add_argument("--limit", type=int, help="limit each split for a quick experiment")
    parser.add_argument("--resume", help="resume an FP32 training checkpoint")
    parser.add_argument(
        "--epochs-this-run", type=int,
        help="stop after this many epochs; useful for short Colab sessions",
    )
    return parser.parse_args()


def main():
    """Dựng data/model/optimizer, train–validate và quản lý best/last."""
    args = parse_args()
    config = load_config(args.config, require_dataset=True)
    random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))
    device = choose_device(config.get("device", "auto"))
    train_loader = build_coco_loader(
        config, "train", limit=args.limit,
        batch_size=config["training"].get("fp32_batch_size", config["training"]["batch_size"]),
    )
    val_loader = build_coco_loader(config, "val", shuffle=False, limit=args.limit)
    print(
        f"FP32 setup device={device} train_images={len(train_loader.dataset)} "
        f"val_images={len(val_loader.dataset)} epochs={config['training']['fp32_epochs']}",
        flush=True,
    )
    print(f"FP32 best={config['output']['fp32_best']}", flush=True)
    print(f"FP32 last={config['output']['fp32_last']}", flush=True)
    model = build_fasterrcnn_convnext(config).to(device)
    print(
        f"model={config['model']['backbone']} parameters={model.logical_parameter_count:,}",
        flush=True,
    )
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["training"].get("lr_step_size", 8)),
        gamma=float(config["training"].get("lr_gamma", 0.1)),
    )
    best_map = -1.0
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(
            args.resume, model, optimizer, map_location=device, scheduler=scheduler
        )
        start_epoch = int(payload.get("epoch", 0))
        best_map = float(payload.get("extra", {}).get("best_map", -1.0))
        print(f"resumed FP32 checkpoint={args.resume} epoch={start_epoch}")
    total_epochs = int(config["training"]["fp32_epochs"])
    if args.epochs_this_run is not None and args.epochs_this_run <= 0:
        raise ValueError("--epochs-this-run must be positive")
    end_epoch = total_epochs
    if args.epochs_this_run is not None:
        end_epoch = min(start_epoch + args.epochs_this_run, total_epochs)
    for epoch in range(start_epoch, end_epoch):
        print(
            f"FP32 epoch={epoch + 1}/{config['training']['fp32_epochs']} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}",
            flush=True,
        )
        warmup_scheduler = None
        if epoch == 0:
            warmup_iterations = min(
                int(config["training"].get("warmup_iterations", 0)),
                max(len(train_loader) - 1, 0),
            )
            if warmup_iterations:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.001, total_iters=warmup_iterations
                )
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            float(config["training"].get("grad_clip_norm", 0)),
            int(config["training"].get("print_frequency", 20)),
            warmup_scheduler,
        )
        # Persist the completed training epoch before validation so a metric
        # or Drive interruption cannot force the whole epoch to be repeated.
        scheduler.step()
        save_checkpoint(
            config["output"]["fp32_last"], model, optimizer, epoch + 1,
            {"train": train_metrics, "validation_pending": True},
            {"backbone": config["model"]["backbone"], "format": "fp32", "best_map": best_map},
            scheduler,
        )
        print(f"saved pre-validation FP32 checkpoint: {config['output']['fp32_last']}", flush=True)
        print("FP32 validation started", flush=True)
        val_metrics = evaluate_model(
            model, val_loader, device,
            include_rpn=True,
        )
        print("FP32 validation completed", flush=True)
        benchmark_metrics = benchmark_inference(
            model, val_loader, device,
            int(config["training"].get("epoch_benchmark_images", 100)),
        )
        benchmark_record = {"stage": "fp32", "epoch": epoch + 1, **benchmark_metrics}
        benchmark_history = config["output"].get(
            "epoch_benchmarks",
            str(Path(config["output"]["directory"]) / "epoch_benchmarks.json"),
        )
        append_epoch_benchmark(benchmark_history, benchmark_record)
        print(f"FP32 epoch benchmark={benchmark_record}", flush=True)
        print(f"epoch={epoch + 1} train={train_metrics} validation={val_metrics}")
        if val_metrics["map_50_95"] > best_map:
            best_map = val_metrics["map_50_95"]
            save_checkpoint(
                config["output"]["fp32_best"], model, optimizer, epoch + 1,
                {**val_metrics, "benchmark": benchmark_metrics},
                {"backbone": config["model"]["backbone"], "format": "fp32", "best_map": best_map},
                scheduler,
            )
            print(f"saved new FP32 best: {config['output']['fp32_best']}", flush=True)
        save_checkpoint(
            config["output"]["fp32_last"], model, optimizer, epoch + 1,
            {**val_metrics, "benchmark": benchmark_metrics},
            {"backbone": config["model"]["backbone"], "format": "fp32", "best_map": best_map},
            scheduler,
        )
        print(f"saved FP32 resume checkpoint: {config['output']['fp32_last']}", flush=True)
    print(
        f"FP32 run completed at epoch {end_epoch}/{total_epochs}. "
        f"Resume checkpoint: {config['output']['fp32_last']}",
        flush=True,
    )
    if end_epoch == total_epochs:
        print(f"Best FP32 checkpoint: {config['output']['fp32_best']} (mAP={best_map:.4f})")


if __name__ == "__main__":
    main()
