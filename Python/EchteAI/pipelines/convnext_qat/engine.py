import time

import torch


def move_targets(targets, device):
    return [{key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]


def train_one_epoch(
    model, loader, optimizer, device, grad_clip_norm=0.0, print_frequency=20,
    iteration_scheduler=None,
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
            print(f"step={step}/{len(loader)} loss={total_loss / step:.4f}")
    return {"loss": total_loss / max(len(loader), 1), "seconds": time.perf_counter() - started}


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
