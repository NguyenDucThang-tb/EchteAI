#!/usr/bin/env python3
"""Benchmark riêng backbone FP32 và PT2E INT8 bằng tensor đầu vào giả."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.config import load_config
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization.pt2e_qat import (
    build_backbone_body_region,
    convert_pt2e_backbone,
    prepare_pt2e_backbone_qat,
    set_pt2e_qat_phase,
)


def parse_args():
    """Đọc kích thước tensor và số vòng benchmark từ CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--observer-iters", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scope", choices=["backbone", "backbone_fpn"], default="backbone")
    parser.add_argument("--output")
    return parser.parse_args()


def benchmark_region(region, sample, warmup_iters, iters):
    """Đo latency trung bình của một vùng graph trên CPU."""
    region = region.cpu().eval()
    timings = []
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = region(sample)
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = region(sample)
            timings.append((time.perf_counter() - t0) * 1000.0)
    avg_ms = sum(timings) / len(timings)
    return {
        "avg_ms": avg_ms,
        "fps": 1000.0 / avg_ms if avg_ms > 0 else float("nan"),
        "iters": int(iters),
    }


def main():
    """Prepare/convert PT2E rồi so tốc độ cùng một vùng và input FP32."""
    args = parse_args()
    config = load_config(args.config, require_dataset=False)
    random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))
    torch.set_num_threads(int(args.threads))

    pt2e_cfg = config.get("quantization", {}).get("pt2e", {})
    batch_size = int(args.batch_size or pt2e_cfg.get("example_batch_size", 2))
    height = int(args.height or pt2e_cfg.get("example_height", 256))
    width = int(args.width or pt2e_cfg.get("example_width", 320))

    sample = torch.randn(batch_size, 3, height, width)

    print("=== Build FP32 detector and backbone region ===", flush=True)
    fp32_model = build_fasterrcnn_convnext(config).cpu().eval()
    fp32_region = build_backbone_body_region(fp32_model.backbone, scope=args.scope).cpu().eval()
    fp32_result = benchmark_region(fp32_region, sample, args.warmup_iters, args.iters)
    print(f"FP32 region result: {fp32_result}", flush=True)

    print("=== Prepare PT2E detector ===", flush=True)
    pt2e_model = build_fasterrcnn_convnext(config).cpu()
    pt2e_model = prepare_pt2e_backbone_qat(pt2e_model, config)

    # Warmup observer bằng dữ liệu giả để qparam có giá trị trước khi convert.
    print("=== Observer warmup ===", flush=True)
    fake_quantizers = set_pt2e_qat_phase(pt2e_model, "observer_warmup")
    print(f"fake_quantizers observer_warmup={fake_quantizers}", flush=True)
    pt2e_model.train()
    with torch.no_grad():
        for _ in range(int(args.observer_iters)):
            _ = pt2e_model.backbone.body_region(sample)

    print("=== Freeze and convert PT2E ===", flush=True)
    fake_quantizers = set_pt2e_qat_phase(pt2e_model, "frozen")
    print(f"fake_quantizers frozen={fake_quantizers}", flush=True)
    int8_model = convert_pt2e_backbone(pt2e_model, inplace=False, compile_region=False)
    int8_region = int8_model.backbone.body_region.cpu().eval()
    int8_result = benchmark_region(int8_region, sample, args.warmup_iters, args.iters)
    print(f"PT2E INT8 region result: {int8_result}", flush=True)

    summary = {
        "backbone": config["model"].get("backbone", "convnext_tiny"),
        "scope": args.scope,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "threads": int(args.threads),
        "warmup_iters": int(args.warmup_iters),
        "iters": int(args.iters),
        "observer_iters": int(args.observer_iters),
        "fp32": fp32_result,
        "pt2e_int8": int8_result,
        "speedup_vs_fp32": fp32_result["avg_ms"] / int8_result["avg_ms"],
    }
    print(json.dumps(summary, indent=2), flush=True)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary -> {output}", flush=True)


if __name__ == "__main__":
    main()
