#!/usr/bin/env python3
"""Distributed PT2E QAT training for multi-GPU Kaggle sessions.

Launch example:
    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/train_pt2e_qat_ddp.py --config runtime.yaml

Important: training.qat_batch_size is interpreted as per-GPU batch size.
"""

import argparse
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from pipelines.convnext_qat.config import load_config  # noqa: E402
from pipelines.convnext_qat.data import (  # noqa: E402
    CocoDetectionDataset,
    build_coco_loader,
    detection_collate,
)
from pipelines.convnext_qat.engine import (  # noqa: E402
    append_epoch_benchmark,
    benchmark_inference,
    make_optimizer,
)
from pipelines.convnext_qat.metrics import evaluate_model, save_metrics  # noqa: E402
from pipelines.convnext_qat.models import build_fasterrcnn_convnext  # noqa: E402
from pipelines.convnext_qat.quantization import (  # noqa: E402
    convert_pt2e_backbone,
    prepare_pt2e_backbone_qat,
    pt2e_observers_disabled,
    pt2e_qat_phase,
    save_pt2e_int8_artifact,
    set_pt2e_qat_phase,
    synchronize_pt2e_observers,
    validate_pt2e_schedule,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--epochs-this-run", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--no-find-unused-parameters",
        action="store_true",
        help="disable DDP unused-parameter detection after the graph is stable",
    )
    return parser.parse_args()


def checkpoint_extra(config, best_map, world_size):
    return {
        "format": "pt2e_prepared_qat",
        "region": config.get("quantization", {}).get("pt2e", {}).get("region", "backbone"),
        "backend": "x86_inductor",
        "best_map": best_map,
        "anchor_sizes": config["model"].get("anchor_sizes"),
        "ddp_world_size": world_size,
    }


def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("train_pt2e_qat_ddp.py must be launched with torch.distributed.run/torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PT2E DDP training")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=3),
    )
    return local_rank, rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def rank0_print(rank, *values):
    if rank == 0:
        print(*values, flush=True)


def build_distributed_loader(config, split, rank, world_size, limit=None, batch_size=None):
    dataset_cfg = config["dataset"]
    dataset = CocoDetectionDataset(
        dataset_cfg[f"{split}_images"],
        dataset_cfg[f"{split}_annotations"],
        training=split == "train",
        augmentation={
            **(config.get("augmentation", {}) if split == "train" else {}),
            "ignore_category_ids": dataset_cfg.get("ignore_category_ids", []),
        },
    )
    if int(dataset_cfg["num_classes"]) != len(dataset.category_id_to_label) + 1:
        raise ValueError(
            f"dataset.num_classes={dataset_cfg['num_classes']} but {split} annotations "
            f"contain {len(dataset.category_id_to_label)} foreground categories"
        )
    if limit is not None:
        dataset = Subset(dataset, range(min(int(limit), len(dataset))))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=split == "train",
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or config["training"]["batch_size"]),
        sampler=sampler,
        num_workers=int(dataset_cfg.get("workers", 2)),
        collate_fn=detection_collate,
        pin_memory=True,
    )
    return loader, sampler


def move_targets(targets, device):
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def reduce_train_metrics(loss_sum, steps, seconds, device):
    values = torch.tensor([loss_sum, float(steps), seconds], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_loss, total_steps, summed_seconds = values.tolist()
    return {
        "loss": total_loss / max(total_steps, 1.0),
        "global_steps": int(total_steps),
        "steps_per_rank": int(steps),
        "seconds_avg_rank": summed_seconds / max(dist.get_world_size(), 1),
    }


def train_one_epoch_ddp(
    model,
    loader,
    sampler,
    optimizer,
    device,
    epoch,
    rank,
    grad_clip_norm=0.0,
    print_frequency=20,
):
    sampler.set_epoch(epoch)
    model.train()
    total_loss = 0.0
    started = time.perf_counter()
    for step, (images, targets) in enumerate(loader, 1):
        if rank == 0 and (step == 1 or (print_frequency and step % print_frequency == 0)):
            elapsed = time.perf_counter() - started
            print(
                f"rank0_step_start={step}/{len(loader)} "
                f"batch={len(images)} elapsed={elapsed:.1f}s",
                flush=True,
            )
        step_started = time.perf_counter()
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets(targets, device)
        losses = model(images, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"rank={rank} non-finite loss at step={step}: {losses}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += float(loss.detach())
        if rank == 0 and (step == 1 or (print_frequency and step % print_frequency == 0)):
            elapsed = time.perf_counter() - started
            learning_rate = optimizer.param_groups[0]["lr"]
            memory_gb = torch.cuda.max_memory_allocated(device) / 2**30
            print(
                f"rank0_step={step}/{len(loader)} loss={total_loss / step:.4f} "
                f"lr={learning_rate:.3e} step_seconds={time.perf_counter() - step_started:.1f} "
                f"elapsed={elapsed:.1f}s max_mem={memory_gb:.2f}GB",
                flush=True,
            )
    return reduce_train_metrics(total_loss, len(loader), time.perf_counter() - started, device)


def main():
    args = parse_args()
    local_rank, rank, world_size, device = setup_distributed()
    try:
        config = load_config(args.config, require_dataset=True)
        random.seed(config.get("seed", 42) + rank)
        torch.manual_seed(config.get("seed", 42) + rank)
        total_epochs = int(config["training"].get("pt2e_qat_epochs", 3))
        pt2e_config = config["quantization"].get("pt2e", {})
        observer_warmup_epochs = int(pt2e_config.get("observer_warmup_epochs", 1))
        observer_freeze_epochs = int(pt2e_config.get("observer_freeze_epochs", 1))
        validate_pt2e_schedule(total_epochs, observer_warmup_epochs, observer_freeze_epochs)
        validate_after_epoch = bool(
            config["training"].get("pt2e_validate_after_epoch", True)
        )
        batch_size = int(config["training"].get("qat_batch_size", 1))
        train_loader, train_sampler = build_distributed_loader(
            config, "train", rank, world_size, limit=args.limit, batch_size=batch_size,
        )
        val_loader = None
        if rank == 0 and validate_after_epoch:
            val_loader = build_coco_loader(
                config, "val", shuffle=False, limit=args.limit, batch_size=batch_size,
            )

        rank0_print(
            rank,
            f"PT2E DDP setup world_size={world_size} batch_per_gpu={batch_size} "
            f"global_batch={batch_size * world_size}",
        )
        rank0_print(rank, f"rank0_train_batches={len(train_loader)}")

        model = build_fasterrcnn_convnext(config)
        fp32_checkpoint = args.fp32_checkpoint or config["output"]["fp32_best"]
        rank0_print(rank, f"Loading FP32 checkpoint: {fp32_checkpoint}")
        load_checkpoint(fp32_checkpoint, model)
        rank0_print(rank, "Exporting backbone body and preparing PT2E x86 QAT graph...")
        model = prepare_pt2e_backbone_qat(model, config).to(device)
        optimizer = make_optimizer(model, config, qat=True)

        start_epoch, best_map = 0, -1.0
        if args.resume:
            payload = load_checkpoint(args.resume, model, optimizer, map_location=device)
            start_epoch = int(payload.get("epoch", 0))
            best_map = float(payload.get("extra", {}).get("best_map", -1.0))
            rank0_print(rank, f"Resumed PT2E QAT checkpoint={args.resume} epoch={start_epoch}")

        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # Preserve ranges collected independently by each GPU; merge them
            # explicitly once per epoch instead of overwriting rank 1 each step.
            broadcast_buffers=False,
            find_unused_parameters=not args.no_find_unused_parameters,
        )

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
            rank0_print(
                rank,
                f"PT2E QAT DDP epoch={epoch + 1}/{total_epochs} phase={phase} "
                f"fake_quantizers={fake_quantizers}",
            )
            train_metrics = train_one_epoch_ddp(
                ddp_model,
                train_loader,
                train_sampler,
                optimizer,
                device,
                epoch,
                rank,
                float(config["training"].get("grad_clip_norm", 0)),
                int(config["training"].get("print_frequency", 20)),
            )
            synchronized_observers = synchronize_pt2e_observers(model)
            rank0_print(rank, f"PT2E synchronized_observers={synchronized_observers}")
            dist.barrier()

            if not validate_after_epoch:
                if rank == 0:
                    metrics = {"train": train_metrics, "validation_skipped": True}
                    save_checkpoint(
                        last_path, model, optimizer, epoch + 1, metrics,
                        checkpoint_extra(config, best_map, world_size),
                    )
                    print(
                        f"Saved PT2E checkpoint without full validation: {last_path}",
                        flush=True,
                    )
                dist.barrier()
                continue

            if rank == 0:
                save_checkpoint(
                    last_path, model, optimizer, epoch + 1,
                    {"train": train_metrics, "validation_pending": True},
                    checkpoint_extra(config, best_map, world_size),
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
                benchmark_record = {
                    "stage": "pt2e",
                    "epoch": epoch + 1,
                    "phase": phase,
                    "ddp_world_size": world_size,
                    **timing,
                }
                append_epoch_benchmark(
                    config["output"].get(
                        "epoch_benchmarks",
                        str(Path(config["output"]["directory"]) / "epoch_benchmarks.json"),
                    ),
                    benchmark_record,
                )
                if phase == "frozen" and validation["map_50_95"] > best_map:
                    best_map = validation["map_50_95"]
                    save_checkpoint(
                        best_path, model, optimizer, epoch + 1, metrics,
                        checkpoint_extra(config, best_map, world_size),
                    )
                    print(f"Saved new PT2E QAT best: {best_path}", flush=True)
                    print("Converting best PT2E checkpoint and evaluating real INT8 on CPU...", flush=True)
                    int8_model = convert_pt2e_backbone(model, inplace=False, compile_region=False)
                    int8_metrics = evaluate_model(
                        int8_model, val_loader, torch.device("cpu"), include_rpn=True,
                    )
                    save_pt2e_int8_artifact(
                        int8_path, int8_model, int8_metrics,
                        {
                            "source_epoch": epoch + 1,
                            "source_qat": str(best_path),
                            "region": model.pt2e_quantized_region,
                            "ddp_world_size": world_size,
                        },
                    )
                    save_metrics(int8_metrics_path, int8_metrics)
                    print(
                        f"Saved PT2E INT8 artifact={int8_path} "
                        f"mAP@50:95={int8_metrics['map_50_95']:.4f}",
                        flush=True,
                    )
                    del int8_model
                save_checkpoint(
                    last_path, model, optimizer, epoch + 1, metrics,
                    checkpoint_extra(config, best_map, world_size),
                )
                print(
                    f"PT2E epoch={epoch + 1} train={train_metrics} validation={validation}",
                    flush=True,
                )
            dist.barrier()

        if rank == 0:
            print(f"PT2E DDP run completed at epoch {end_epoch}/{total_epochs}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
