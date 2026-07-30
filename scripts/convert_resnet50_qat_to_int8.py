#!/usr/bin/env python3
"""Convert a prepared ResNet50 QAT checkpoint into a true eager W8A8 INT8 checkpoint."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from pipelines.fasterrcnn_qat.checkpoint import load_checkpoint, save_checkpoint
from pipelines.fasterrcnn_qat.config import load_config, quantized_modules_for_variant, resolve_qat_profile
from pipelines.fasterrcnn_qat.models import build_fasterrcnn_model
from pipelines.fasterrcnn_qat.quantization import (
    convert_selective_qat,
    mixed_precision_policy_from_config,
    module_qconfig_map_from_policy,
    policy_has_non_int8_weights,
    policy_scope_to_quantized_modules,
    prepare_selective_qat,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_resnet50_hawq_compiler.yaml")
    parser.add_argument("--qat-checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--variant")
    parser.add_argument("--backend")
    parser.add_argument("--force-w8a8", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, require_dataset=False)
    output_path = Path(args.output or config["output"]["int8_model"])

    qat_payload = torch.load(args.qat_checkpoint, map_location="cpu", weights_only=False)
    metadata = qat_payload.get("extra", {}) if isinstance(qat_payload, dict) else {}

    variant = str(args.variant or metadata.get("variant") or config["quantization"].get("variant", "M3")).upper()
    backend = str(args.backend or metadata.get("backend") or config["quantization"].get("backend", "x86"))
    qat_profile = resolve_qat_profile(config, metadata)
    quantized_modules = metadata.get("quantized_modules", quantized_modules_for_variant(config, variant))
    if metadata.get("format") == "prepared_qat" and quantized_modules == []:
        quantized_modules = None
    mixed_precision_policy = None if args.force_w8a8 else (metadata.get("mixed_precision_policy") or mixed_precision_policy_from_config(config))
    module_qconfig_map = None
    if mixed_precision_policy is not None:
        if policy_has_non_int8_weights(mixed_precision_policy):
            raise ValueError(
                "This QAT checkpoint carries a mixed-precision policy with sub-8-bit weights. "
                "Pass --force-w8a8 if you want a pure eager W8A8 export for compiler benchmarking."
            )
        quantized_modules = policy_scope_to_quantized_modules(mixed_precision_policy)
        module_qconfig_map = module_qconfig_map_from_policy(mixed_precision_policy)

    model = build_fasterrcnn_model(config).cpu().eval()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="must run observer before calling calculate_qparams")
        prepared_model = prepare_selective_qat(
            model,
            variant,
            backend,
            quantized_modules=quantized_modules,
            module_qconfig_map=module_qconfig_map,
            qconfig_profile=qat_profile,
        ).cpu().eval()
    load_checkpoint(args.qat_checkpoint, prepared_model, map_location="cpu", strict=True)
    int8_model = convert_selective_qat(prepared_model, inplace=False).cpu().eval()

    metrics = qat_payload.get("metrics", {}) if isinstance(qat_payload, dict) else {}
    save_checkpoint(
        output_path,
        int8_model,
        epoch=int(qat_payload.get("epoch", 0)) if isinstance(qat_payload, dict) else 0,
        metrics=metrics,
        extra={
            "variant": variant,
            "backend": backend,
            "format": "selective_int8",
            "quantized_modules": quantized_modules_for_variant(config, variant) if quantized_modules is None else quantized_modules,
            "mixed_precision_policy": None if args.force_w8a8 else mixed_precision_policy,
            "qat_profile": qat_profile,
            "force_w8a8": bool(args.force_w8a8),
            "source_qat_checkpoint": str(args.qat_checkpoint),
        },
    )
    print(f"Saved eager INT8 checkpoint: {output_path}", flush=True)


if __name__ == "__main__":
    main()
