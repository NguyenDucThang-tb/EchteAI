import json
import time
from pathlib import Path

import torch


def move_targets(targets, device):
    return [{key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]


def train_one_epoch(
    model, loader, optimizer, device, grad_clip_norm=0.0, print_frequency=20,
    iteration_scheduler=None, max_steps=None,
):
    model.train()
    total_loss = 0.0
    started = time.perf_counter()
    for step, (images, targets) in enumerate(loader, 1):
        images = [image.to(device) for image in images]
        targets = move_targets(targets, device)
        losses = model(images, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {losses}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if iteration_scheduler is not None:
            iteration_scheduler.step()
        total_loss += float(loss.detach())
        if print_frequency and step % print_frequency == 0:
            elapsed = time.perf_counter() - started
            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"step={step}/{len(loader)} loss={total_loss / step:.4f} "
                f"lr={learning_rate:.3e} elapsed={elapsed:.1f}s",
                flush=True,
            )
        if max_steps is not None and step >= int(max_steps):
            break
    completed_steps = min(len(loader), int(max_steps)) if max_steps is not None else len(loader)
    return {
        "loss": total_loss / max(completed_steps, 1),
        "seconds": time.perf_counter() - started,
        "steps": completed_steps,
    }


@torch.inference_mode()
def benchmark_inference(model, loader, device, max_images=100):
    """Run one inference pass over at most max_images and report throughput."""
    was_training = model.training
    model.eval()
    processed = 0
    print(f"epoch benchmark started: target={max_images} images device={device}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for images, _ in loader:
        remaining = max_images - processed
        if remaining <= 0:
            break
        images = [image.to(device) for image in images[:remaining]]
        model(images)
        processed += len(images)
        if processed % 25 < len(images) or processed >= max_images:
            print(f"epoch benchmark progress: {processed}/{max_images} images", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if was_training:
        model.train()
    if processed == 0:
        raise ValueError("cannot benchmark an empty data loader")
    return {
        "images": processed,
        "seconds": seconds,
        "latency_ms_per_image": 1000.0 * seconds / processed,
        "fps": processed / seconds,
        "device": str(device),
    }


def append_epoch_benchmark(path, record):
    """Persist benchmark history, replacing duplicate stage/epoch entries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if path.exists():
        history = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            raise ValueError(f"epoch benchmark history must be a list: {path}")
    history = [item for item in history if not (
        item.get("stage") == record.get("stage")
        and item.get("epoch") == record.get("epoch")
    )]
    history.append(record)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_optimizer(model, config, qat=False):
    training = config["training"]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    lr = float(training["qat_lr"] if qat else training["fp32_lr"])
    weight_decay = float(training.get("weight_decay", 0.0))
    name = training.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError("training.optimizer must be adamw or sgd")


def set_optimizer_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
