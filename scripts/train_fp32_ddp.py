#!/usr/bin/env python3
"""Train FP32 phân tán bằng DDP cho phiên Kaggle nhiều GPU.

Launch example:
    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/train_fp32_ddp.py --config runtime.yaml

The FP32 batch size is interpreted per GPU.
"""

import argparse
import os
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
from pipelines.convnext_qat.metrics import evaluate_model  # noqa: E402
from pipelines.convnext_qat.models import build_fasterrcnn_convnext  # noqa: E402


def parse_args():
    """Đọc config, checkpoint resume và số epoch của lần chạy hiện tại."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", help="resume an FP32 training checkpoint")
    parser.add_argument(
        "--epochs-this-run", type=int,
        help="stop after this many epochs; useful for short Kaggle sessions",
    )
    parser.add_argument(
        "--no-find-unused-parameters",
        action="store_true",
        help="disable DDP unused-parameter detection after the graph is stable",
    )
    return parser.parse_args()


def setup_distributed():
    """Khởi tạo NCCL process group và gán mỗi rank vào một GPU."""
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("train_fp32_ddp.py must be launched with torch.distributed.run/torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FP32 DDP training")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=3),
    )
    return local_rank, rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_distributed():
    """Đóng process group sau khi mọi rank hoàn tất."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def rank0_print(rank, *values):
    """Chỉ in log ở rank 0 để tránh trùng dòng."""
    if rank == 0:
        print(*values, flush=True)


def build_distributed_loader(config, split, rank, world_size, limit=None, batch_size=None):
    """Tạo DataLoader cùng DistributedSampler cho từng rank."""
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
    """Chuyển toàn bộ tensor trong target sang GPU của rank hiện tại."""
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def reduce_train_metrics(loss_sum, steps, seconds, device):
    """All-reduce loss/step/time để tạo thống kê chung cho các GPU."""
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
    iteration_scheduler=None,
):
    """Train một epoch DDP, warmup LR và ghi log tiến độ ở rank 0."""
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
        if iteration_scheduler is not None:
            iteration_scheduler.step()
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


def broadcast_buffers_from_rank0(module):
    """Đồng bộ buffer model từ rank 0 trước validation/checkpoint."""
    for buffer in module.buffers():
        dist.broadcast(buffer, src=0)


def main():
    """Điều phối toàn bộ vòng train FP32 DDP và validation trên rank 0."""
    args = parse_args()
    local_rank, rank, world_size, device = setup_distributed()
    try:
        config = load_config(args.config, require_dataset=True)
        batch_size = int(config["training"].get("fp32_batch_size", config["training"]["batch_size"]))
        train_loader, train_sampler = build_distributed_loader(
            config, "train", rank, world_size, limit=args.limit, batch_size=batch_size,
        )
        val_loader = None
        if rank == 0:
            val_loader = build_coco_loader(
                config, "val", shuffle=False, limit=args.limit, batch_size=batch_size,
            )

        rank0_print(
            rank,
            f"FP32 DDP setup world_size={world_size} batch_per_gpu={batch_size} "
            f"global_batch={batch_size * world_size}",
        )
        rank0_print(rank, f"rank0_train_batches={len(train_loader)}")

        model = build_fasterrcnn_convnext(config)
        source_checkpoint = config["output"]["fp32_last"]
        if args.resume:
            source_checkpoint = args.resume
        if Path(source_checkpoint).is_file():
            rank0_print(rank, f"loading checkpoint={source_checkpoint}")
            load_checkpoint(source_checkpoint, model)
        model = model.to(device)
        optimizer = make_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config["training"].get("lr_step_size", 8)),
            gamma=float(config["training"].get("lr_gamma", 0.1)),
        )

        start_epoch = 0
        best_map = -1.0
        if args.resume:
            payload = load_checkpoint(args.resume, model, optimizer, map_location=device, scheduler=scheduler)
            start_epoch = int(payload.get("epoch", 0))
            best_map = float(payload.get("extra", {}).get("best_map", -1.0))
            rank0_print(rank, f"resumed FP32 checkpoint={args.resume} epoch={start_epoch}")

        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=not args.no_find_unused_parameters,
        )

        total_epochs = int(config["training"]["fp32_epochs"])
        if args.epochs_this_run is not None and args.epochs_this_run <= 0:
            raise ValueError("--epochs-this-run must be positive")
        end_epoch = total_epochs
        if args.epochs_this_run is not None:
            end_epoch = min(start_epoch + args.epochs_this_run, total_epochs)

        for epoch in range(start_epoch, end_epoch):
            rank0_print(
                rank,
                f"FP32 DDP epoch={epoch + 1}/{total_epochs} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}",
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
                warmup_scheduler,
            )
            scheduler.step()
            dist.barrier()

            if rank == 0:
                save_checkpoint(
                    config["output"]["fp32_last"],
                    model,
                    optimizer,
                    epoch + 1,
                    {"train": train_metrics, "validation_pending": True},
                    {"backbone": config["model"]["backbone"], "format": "fp32_ddp", "best_map": best_map},
                    scheduler,
                )
                print(f"saved pre-validation FP32 checkpoint: {config['output']['fp32_last']}", flush=True)
                print("FP32 validation started on rank0", flush=True)
                val_metrics = evaluate_model(
                    model,
                    val_loader,
                    device,
                    include_rpn=True,
                )
                print("FP32 validation completed", flush=True)
                benchmark_metrics = benchmark_inference(
                    model,
                    val_loader,
                    device,
                    int(config["training"].get("epoch_benchmark_images", 100)),
                )
                benchmark_record = {"stage": "fp32", "epoch": epoch + 1, "ddp_world_size": world_size, **benchmark_metrics}
                benchmark_history = config["output"].get(
                    "epoch_benchmarks",
                    str(Path(config["output"]["directory"]) / "epoch_benchmarks.json"),
                )
                append_epoch_benchmark(benchmark_history, benchmark_record)
                print(f"FP32 epoch benchmark={benchmark_record}", flush=True)
                print(f"fp32_epoch={epoch + 1}/{total_epochs} train={train_metrics} validation={val_metrics}", flush=True)
                if val_metrics["map_50_95"] > best_map:
                    best_map = val_metrics["map_50_95"]
                    save_checkpoint(
                        config["output"]["fp32_best"],
                        model,
                        optimizer,
                        epoch + 1,
                        {**val_metrics, "benchmark": benchmark_metrics},
                        {"backbone": config["model"]["backbone"], "format": "fp32_ddp", "best_map": best_map},
                        scheduler,
                    )
                    print(f"saved new FP32 best: {config['output']['fp32_best']}", flush=True)
                save_checkpoint(
                    config["output"]["fp32_last"],
                    model,
                    optimizer,
                    epoch + 1,
                    {**val_metrics, "benchmark": benchmark_metrics},
                    {"backbone": config["model"]["backbone"], "format": "fp32_ddp", "best_map": best_map},
                    scheduler,
                )
                print(f"saved FP32 resume checkpoint: {config['output']['fp32_last']}", flush=True)
            dist.barrier()

        if rank == 0:
            print(
                f"FP32 DDP run completed at epoch {end_epoch}/{total_epochs}. "
                f"Resume checkpoint: {config['output']['fp32_last']}",
                flush=True,
            )
            if end_epoch >= total_epochs:
                print(
                    f"Best FP32 checkpoint: {config['output']['fp32_best']} "
                    f"(mAP={best_map:.4f})",
                    flush=True,
                )
        dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
