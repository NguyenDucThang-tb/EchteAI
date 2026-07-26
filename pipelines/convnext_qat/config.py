from pathlib import Path

import yaml


def load_config(path, require_dataset=False):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("dataset", "model", "training", "quantization", "output"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Relative dataset/output paths are resolved from the repository root, not cwd.
    root = path.parent.parent
    for key in ("train_images", "train_annotations", "val_images", "val_annotations", "test_images", "test_annotations"):
        value = config["dataset"].get(key)
        if value and not Path(value).is_absolute():
            config["dataset"][key] = str((root / value).resolve())
    for key, value in config["output"].items():
        if isinstance(value, str) and not Path(value).is_absolute():
            config["output"][key] = str((root / value).resolve())
    mixed_precision = config.get("quantization", {}).get("mixed_precision", {})
    policy_path = mixed_precision.get("policy_path")
    if policy_path and not Path(policy_path).is_absolute():
        mixed_precision["policy_path"] = str((root / policy_path).resolve())
    policy_output = mixed_precision.get("policy_output")
    if policy_output and not Path(policy_output).is_absolute():
        mixed_precision["policy_output"] = str((root / policy_output).resolve())
    compiler = config.get("quantization", {}).get("compiler", {})
    for key in ("artifact_dir", "benchmark_json", "int8_reference_checkpoint"):
        value = compiler.get(key)
        if value and not Path(value).is_absolute():
            compiler[key] = str((root / value).resolve())
    if require_dataset:
        validate_dataset_paths(config)
    return config


def validate_dataset_paths(config, splits=("train", "val")):
    missing = []
    for split in splits:
        for suffix in ("images", "annotations"):
            path = Path(config["dataset"][f"{split}_{suffix}"])
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing COCO dataset paths:\n  " + "\n  ".join(missing))


def choose_device(value):
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def quantized_modules_for_variant(config, variant):
    variant = str(variant).upper()
    quantization = config["quantization"]
    configured_variant = str(quantization.get("variant", "M3")).upper()
    if variant == configured_variant and "quantized_modules" in quantization:
        return list(quantization["quantized_modules"])
    variants = quantization.get("variant_modules", {})
    if variant in variants:
        return list(variants[variant])
    return None


def qat_profile_from_config(config):
    quantization = config.get("quantization", {})
    profile = str(quantization.get("qat_profile", "eager")).strip().lower()
    if profile not in {"eager", "tensorrt"}:
        raise ValueError("quantization.qat_profile must be 'eager' or 'tensorrt'")
    return profile


def resolve_qat_profile(config, checkpoint_extra=None):
    if isinstance(checkpoint_extra, dict):
        profile = checkpoint_extra.get("qat_profile")
        if profile is not None:
            profile = str(profile).strip().lower()
            if profile not in {"eager", "tensorrt"}:
                raise ValueError("checkpoint extra qat_profile must be 'eager' or 'tensorrt'")
            return profile
    return qat_profile_from_config(config)
