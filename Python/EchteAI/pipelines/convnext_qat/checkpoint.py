import io
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, epoch=0, metrics=None, extra=None, scheduler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path, model, optimizer=None, map_location="cpu", strict=True, scheduler=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = payload.get("model", payload)
    model.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def checkpoint_size_mb(path):
    return Path(path).stat().st_size / (1024.0 * 1024.0)


def model_state_size_mb(model):
    """Serialized model-only size, excluding optimizer and training metadata."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024.0 * 1024.0)
