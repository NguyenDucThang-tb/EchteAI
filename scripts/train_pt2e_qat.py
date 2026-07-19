#!/usr/bin/env python3
"""Train bù sai số PT2E graph-mode cho backbone, giữ detector heads FP32."""

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
from pipelines.convnext_qat.engine import benchmark_inference, make_optimizer, train_one_epoch
from pipelines.convnext_qat.metrics import evaluate_model, save_metrics
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import (
    convert_pt2e_backbone, prepare_pt2e_backbone_qat, save_pt2e_int8_artifact,
    pt2e_observers_disabled, pt2e_qat_phase, set_pt2e_qat_phase,
    validate_pt2e_schedule,
)


def parse_args():
    """Đọc config, resume checkpoint và số epoch PT2E cần chạy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--epochs-this-run", type=int)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def checkpoint_extra(config, best_map):
    """Tạo metadata mô tả scope, lịch phase và best mAP."""
    return {
        "format": "pt2e_prepared_qat",
        "region": config.get("quantization", {}).get("pt2e", {}).get("region", "backbone"),
        "backend": "x86_inductor",
        "best_map": best_map,
        "anchor_sizes": config["model"].get("anchor_sizes"),
    }


def main():
    """Prepare graph, chạy lịch observer/full/frozen, convert và đánh giá INT8."""
    args = parse_args()
    config = load_config(args.config, require_dataset=True)
    random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))
    device = choose_device(config.get("device", "auto"))
    total_epochs = int(config["training"].get("pt2e_qat_epochs", 3))
    pt2e_config = config["quantization"].get("pt2e", {})
    observer_warmup_epochs = int(pt2e_config.get("observer_warmup_epochs", 1))
    observer_freeze_epochs = int(pt2e_config.get("observer_freeze_epochs", 1))
    validate_pt2e_schedule(total_epochs, observer_warmup_epochs, observer_freeze_epochs)
    validate_after_epoch = bool(
        config["training"].get("pt2e_validate_after_epoch", True)
    )
    batch_size = int(config["training"].get("qat_batch_size", 1))
    train_loader = build_coco_loader(config, "train", limit=args.limit, batch_size=batch_size)
    val_loader = None
    if validate_after_epoch:
        val_loader = build_coco_loader(
            config, "val", shuffle=False, limit=args.limit, batch_size=batch_size,
        )

    model = build_fasterrcnn_convnext(config)
    fp32_checkpoint = args.fp32_checkpoint or config["output"]["fp32_best"]
    print(f"Loading FP32 checkpoint: {fp32_checkpoint}", flush=True)
    load_checkpoint(fp32_checkpoint, model)
    print("Exporting backbone body and preparing PT2E x86 QAT graph...", flush=True)
    model = prepare_pt2e_backbone_qat(model, config).to(device)
    optimizer = make_optimizer(model, config, qat=True)

    start_epoch, best_map = 0, -1.0
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", 0))
        best_map = float(payload.get("extra", {}).get("best_map", -1.0))
        print(f"Resumed PT2E QAT checkpoint={args.resume} epoch={start_epoch}", flush=True)
    if args.epochs_this_run is not None and args.epochs_this_run <= 0:
        raise ValueError("--epochs-this-run must be positive")
    end_epoch = total_epochs
    if args.epochs_this_run is not None:
        end_epoch = min(total_epochs, start_epoch + args.epochs_this_run)

    last_path = config["output"].get(
        "pt2e_qat_last", str(Path(config["output"]["directory"]) / "pt2e_qat_last.pt"),
    )
    best_path = config["output"].get(
        "pt2e_qat_best", str(Path(config["output"]["directory"]) / "pt2e_qat_best.pt"),
    )
    int8_path = config["output"].get(
        "pt2e_int8_model", str(Path(config["output"]["directory"]) / "pt2e_int8.pt"),
    )
    int8_metrics_path = config["output"].get(
        "pt2e_int8_evaluation",
        str(Path(config["output"]["directory"]) / "pt2e_int8_evaluation.json"),
    )
    for epoch in range(start_epoch, end_epoch):
        phase = pt2e_qat_phase(
            epoch, total_epochs, observer_warmup_epochs, observer_freeze_epochs,
        )
        fake_quantizers = set_pt2e_qat_phase(model, phase)
        print(
            f"PT2E QAT epoch={epoch + 1}/{total_epochs} phase={phase} "
            f"fake_quantizers={fake_quantizers}", flush=True,
        )
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            float(config["training"].get("grad_clip_norm", 0)),
            int(config["training"].get("print_frequency", 20)),
        )
        if not validate_after_epoch:
            metrics = {"train": train_metrics, "validation_skipped": True}
            save_checkpoint(
                last_path, model, optimizer, epoch + 1, metrics,
                checkpoint_extra(config, best_map),
            )
            print(
                f"Saved PT2E checkpoint without full validation: {last_path}",
                flush=True,
            )
            continue

        save_checkpoint(
            last_path, model, optimizer, epoch + 1,
            {"train": train_metrics, "validation_pending": True},
            checkpoint_extra(config, best_map),
        )
        print(f"Saved pre-validation PT2E checkpoint: {last_path}", flush=True)
        with pt2e_observers_disabled(model):
            validation = evaluate_model(
                model, val_loader, device, include_rpn=True,
            )
            timing = benchmark_inference(
                model, val_loader, device,
                int(config["training"].get("epoch_benchmark_images", 100)),
            )
        metrics = {**validation, "benchmark": timing}
        # Only frozen-range checkpoints are eligible for final conversion.
        if phase == "frozen" and validation["map_50_95"] > best_map:
            best_map = validation["map_50_95"]
            save_checkpoint(
                best_path, model, optimizer, epoch + 1, metrics,
                checkpoint_extra(config, best_map),
            )
            print(f"Saved new PT2E QAT best: {best_path}", flush=True)
            print("Converting best PT2E checkpoint and evaluating real INT8 on CPU...", flush=True)
            int8_model = convert_pt2e_backbone(model, inplace=False, compile_region=False)
            int8_metrics = evaluate_model(
                int8_model, val_loader, torch.device("cpu"),
                include_rpn=True,
            )
            save_pt2e_int8_artifact(
                int8_path, int8_model, int8_metrics,
                {
                    "source_epoch": epoch + 1,
                    "source_qat": str(best_path),
                    "region": model.pt2e_quantized_region,
                },
            )
            save_metrics(int8_metrics_path, int8_metrics)
            print(
                f"Saved PT2E INT8 artifact={int8_path} "
                f"mAP@50:95={int8_metrics['map_50_95']:.4f}", flush=True,
            )
            del int8_model
        save_checkpoint(
            last_path, model, optimizer, epoch + 1, metrics,
            checkpoint_extra(config, best_map),
        )
        print(f"PT2E epoch={epoch + 1} train={train_metrics} validation={validation}", flush=True)
    print(f"PT2E QAT run completed at epoch {end_epoch}/{total_epochs}", flush=True)


if __name__ == "__main__":
    main()
