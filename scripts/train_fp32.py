#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from EchteAI.pipelines.convnext_qat.checkpoint import load_checkpoint, save_checkpoint
from EchteAI.pipelines.convnext_qat.config import choose_device, load_config
from EchteAI.pipelines.convnext_qat.data import build_coco_loader
from EchteAI.pipelines.convnext_qat.engine import make_optimizer, train_one_epoch
from EchteAI.pipelines.convnext_qat.metrics import evaluate_model
from EchteAI.pipelines.convnext_qat.models import build_fasterrcnn_convnext


def parse_args():
    parser = argparse.ArgumentParser(description="Train FP32 Faster R-CNN ConvNeXt-FPN")
    parser.add_argument("--config", default="configs/fasterrcnn_convnext_qat.yaml")
    parser.add_argument("--limit", type=int, help="limit each split for a quick experiment")
    parser.add_argument("--resume", help="resume an FP32 training checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=True)
    random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))
    device = choose_device(config.get("device", "auto"))
    train_loader = build_coco_loader(config, "train", limit=args.limit)
    val_loader = build_coco_loader(config, "val", shuffle=False, limit=args.limit)
    model = build_fasterrcnn_convnext(config).to(device)
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
    for epoch in range(start_epoch, int(config["training"]["fp32_epochs"])):
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
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step()
        print(f"epoch={epoch + 1} train={train_metrics} validation={val_metrics}")
        if val_metrics["map_50_95"] > best_map:
            best_map = val_metrics["map_50_95"]
            save_checkpoint(
                config["output"]["fp32_best"], model, optimizer, epoch + 1, val_metrics,
                {"backbone": config["model"]["backbone"], "format": "fp32", "best_map": best_map},
                scheduler,
            )
        save_checkpoint(
            config["output"]["fp32_last"], model, optimizer, epoch + 1, val_metrics,
            {"backbone": config["model"]["backbone"], "format": "fp32", "best_map": best_map},
            scheduler,
        )
    print(f"Best FP32 checkpoint: {config['output']['fp32_best']} (mAP={best_map:.4f})")


if __name__ == "__main__":
    main()
