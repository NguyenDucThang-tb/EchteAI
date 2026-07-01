#!/usr/bin/env python3
"""Distributed selective QAT training for multi-GPU Kaggle sessions.

Launch example:
    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/train_qat_ddp.py --config runtime.yaml --fp32-checkpoint fp32_best.pt

Important: training.qat_batch_size is interpreted as per-GPU batch size.
On 2 GPUs, qat_batch_size=1 gives global batch size 2.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from EchteAI.pipelines.convnext_qat.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from EchteAI.pipelines.convnext_qat.config import load_config, quantized_modules_for_variant  # noqa: E402
from EchteAI.pipelines.convnext_qat.data import (  # noqa: E402
    CocoDetectionDataset,
    build_coco_loader,
    detection_collate,
)
from EchteAI.pipelines.convnext_qat.engine import (  # noqa: E402
    append_epoch_benchmark,
    benchmark_inference,
    make_optimizer,
    set_optimizer_lr,
)
from EchteAI.pipelines.convnext_qat.metrics import evaluate_model  # noqa: E402
from EchteAI.pipelines.convnext_qat.models import build_fasterrcnn_convnext  # noqa: E402
from EchteAI.pipelines.convnext_qat.quantization import (  # noqa: E402
    convert_selective_qat,
    prepare_selective_qat,
    set_qat_phase,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fasterrcnn_convnext_qat.yaml")
    parser.add_argument("--fp32-checkpoint", help="override output.fp32_best")
    parser.add_argument("--variant", choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", help="resume a prepared-QAT checkpoint")
    parser.add_argument("--epochs-this-run", type=int, help="stop after this many epochs")
    parser.add_argument(
        "--no-find-unused-parameters",
        action="store_true",
        help="disable DDP unused-parameter detection after you know the graph is stable",
    )
    return parser.parse_args()


def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("train_qat_ddp.py must be launched with torch.distributed.run/torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP QAT training")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
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
        if rank == 0 and print_frequency and step % print_frequency == 0:
            elapsed = time.perf_counter() - started
            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"rank0_step={step}/{len(loader)} loss={total_loss / step:.4f} "
                f"lr={learning_rate:.3e} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return reduce_train_metrics(total_loss, len(loader), time.perf_counter() - started, device)


def broadcast_buffers_from_rank0(module):
    for buffer in module.buffers():
        dist.broadcast(buffer, src=0)


@torch.no_grad()
def observer_warmup_ddp(model, loader, device, image_count, rank):
    rank0_print(rank, f"observer calibration started target={image_count} device={device}")
    model.eval()
    set_qat_phase(model.module, "calibration")
    observed = 0
    for images, _ in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        model(images)
        observed += len(images)
        if rank == 0 and (observed % 25 < len(images) or observed >= image_count):
            print(f"observer calibration rank0={observed}/{image_count} images", flush=True)
        if observed >= image_count:
            break
    # Keep all ranks numerically aligned before train. Rank 0's observer stats
    # are enough for this one-epoch Kaggle workflow and avoid divergent buffers.
    broadcast_buffers_from_rank0(model.module)
    model.train()
    return observed


def main():
    args = parse_args()
    local_rank, rank, world_size, device = setup_distributed()
    try:
        config = load_config(args.config, require_dataset=True)
        variant = args.variant or str(config["quantization"].get("variant", "M3")).upper()
        backend = config["quantization"].get("backend", "x86")
        qat_batch_size = int(config["training"].get("qat_batch_size", config["training"]["batch_size"]))
        train_loader, train_sampler = build_distributed_loader(
            config, "train", rank, world_size, limit=args.limit, batch_size=qat_batch_size,
        )
        val_loader = None
        if rank == 0:
            val_loader = build_coco_loader(
                config, "val", shuffle=False, limit=args.limit, batch_size=qat_batch_size,
            )
        quantized_modules = quantized_modules_for_variant(config, variant)
        rank0_print(
            rank,
            f"QAT DDP setup world_size={world_size} variant={variant} "
            f"batch_per_gpu={qat_batch_size} global_batch={qat_batch_size * world_size}",
        )
        rank0_print(rank, f"rank0_train_batches={len(train_loader)}")
        rank0_print(rank, f"quantized_modules={quantized_modules}")

        source_checkpoint = args.fp32_checkpoint or config["output"]["fp32_best"]
        rank0_print(rank, "building FP32 model topology...")
        fp32_model = build_fasterrcnn_convnext(config)
        rank0_print(rank, f"loading FP32 checkpoint={source_checkpoint}")
        load_checkpoint(source_checkpoint, fp32_model)
        rank0_print(rank, "FP32 checkpoint loaded; preparing selective QAT model...")
        qat_model = prepare_selective_qat(
            fp32_model, variant, backend, quantized_modules=quantized_modules,
        ).to(device)
        optimizer = make_optimizer(qat_model, config, qat=True)
        rank0_print(rank, f"QAT optimizer ready lr={optimizer.param_groups[0]['lr']:.3e}")

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
            rank0_print(rank, f"resumed QAT checkpoint={args.resume} epoch={start_epoch}")

        ddp_model = DistributedDataParallel(
            qat_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=not args.no_find_unused_parameters,
        )

        if not args.resume:
            observed_images = observer_warmup_ddp(
                ddp_model,
                train_loader,
                device,
                int(config["quantization"].get("calibration_images", 256)),
                rank,
            )
            rank0_print(rank, f"observer calibration rank{rank} images={observed_images}")
        dist.barrier()

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
            rank0_print(
                rank,
                f"QAT DDP epoch={epoch + 1}/{total_epochs} phase={phase} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}",
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
            dist.barrier()

            if rank == 0:
                save_checkpoint(
                    config["output"]["qat_last"],
                    qat_model,
                    optimizer,
                    epoch + 1,
                    {"train": train_metrics, "validation_pending": True},
                    {
                        "variant": variant,
                        "backend": backend,
                        "format": "prepared_qat",
                        "quantized_modules": quantized_modules or [],
                        "best_map": best_map,
                        "ddp_world_size": world_size,
                    },
                )
                print(f"saved pre-validation QAT checkpoint: {config['output']['qat_last']}", flush=True)
                print("QAT validation started on rank0", flush=True)
                val_metrics = evaluate_model(qat_model, val_loader, device, include_rpn=False)
                print("QAT validation completed", flush=True)
                benchmark_metrics = benchmark_inference(
                    qat_model,
                    val_loader,
                    device,
                    int(config["training"].get("epoch_benchmark_images", 100)),
                )
                benchmark_record = {
                    "stage": "qat",
                    "epoch": epoch + 1,
                    "phase": phase,
                    "ddp_world_size": world_size,
                    **benchmark_metrics,
                }
                benchmark_history = config["output"].get(
                    "epoch_benchmarks",
                    str(Path(config["output"]["directory"]) / "epoch_benchmarks.json"),
                )
                append_epoch_benchmark(benchmark_history, benchmark_record)
                print(f"QAT epoch benchmark={benchmark_record}", flush=True)
                print(
                    f"qat_epoch={epoch + 1}/{total_epochs} phase={phase} "
                    f"train={train_metrics} validation={val_metrics}",
                    flush=True,
                )
                if phase == "frozen" and (not best_saved or val_metrics["map_50_95"] > best_map):
                    best_map = val_metrics["map_50_95"]
                    best_saved = True
                    save_checkpoint(
                        config["output"]["qat_best"],
                        qat_model,
                        optimizer,
                        epoch + 1,
                        {**val_metrics, "benchmark": benchmark_metrics},
                        {
                            "variant": variant,
                            "backend": backend,
                            "format": "prepared_qat",
                            "quantized_modules": quantized_modules or [],
                            "best_map": best_map,
                            "ddp_world_size": world_size,
                        },
                    )
                    print(f"saved new QAT best: {config['output']['qat_best']}", flush=True)
                save_checkpoint(
                    config["output"]["qat_last"],
                    qat_model,
                    optimizer,
                    epoch + 1,
                    {**val_metrics, "benchmark": benchmark_metrics},
                    {
                        "variant": variant,
                        "backend": backend,
                        "format": "prepared_qat",
                        "quantized_modules": quantized_modules or [],
                        "best_map": best_map,
                        "ddp_world_size": world_size,
                    },
                )
                print(f"saved QAT resume checkpoint: {config['output']['qat_last']}", flush=True)
            dist.barrier()

        if rank == 0:
            print(
                f"QAT DDP run completed at epoch {end_epoch}/{total_epochs}. "
                f"Resume checkpoint: {config['output']['qat_last']}",
                flush=True,
            )
            if end_epoch >= total_epochs:
                if not best_saved:
                    raise ValueError("QAT schedule has no frozen-observer epoch eligible for best checkpoint selection")
                load_checkpoint(config["output"]["qat_best"], qat_model)
                int8_model = convert_selective_qat(qat_model.to("cpu"))
                save_checkpoint(
                    config["output"]["int8_model"],
                    int8_model,
                    metrics={"best_qat_map_50_95": best_map},
                    extra={
                        "variant": variant,
                        "backend": backend,
                        "format": "selective_int8",
                        "quantized_modules": quantized_modules or [],
                        "ddp_world_size": world_size,
                    },
                )
                print(f"Converted selective INT8 checkpoint: {config['output']['int8_model']}", flush=True)
        dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
