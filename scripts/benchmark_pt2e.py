#!/usr/bin/env python3
"""Compare FP32, eager-island INT8, and PT2E backbone INT8 on x86 CPU."""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import benchmark
from evaluate import load_model
from pipelines.convnext_qat.checkpoint import checkpoint_size_mb, model_state_size_mb
from pipelines.convnext_qat.config import load_config, validate_dataset_paths
from pipelines.convnext_qat.data import build_coco_loader
from pipelines.convnext_qat.metrics import evaluate_model
from pipelines.convnext_qat.quantization import (
    compile_pt2e_region, load_pt2e_int8_artifact,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--fp32-checkpoint", required=True)
    parser.add_argument("--pt2e-int8-checkpoint", required=True)
    parser.add_argument("--eager-int8-checkpoint")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--images", type=int, default=100, help="images used for accuracy evaluation")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.images <= 0:
        parser.error("--images must be positive")

    config = load_config(args.config)
    validate_dataset_paths(config, ("test",))
    torch.set_num_threads(int(config.get("benchmark", {}).get("num_threads", 1)))
    images, _ = next(iter(build_coco_loader(config, "test", shuffle=False, batch_size=1)))
    images = [image.cpu() for image in images]
    warmup = int(config.get("benchmark", {}).get("warmup_iterations", 10))
    iterations = int(config.get("benchmark", {}).get("iterations", 50))
    results = {}

    fp32 = load_model(config, "fp32", args.fp32_checkpoint, torch.device("cpu"))
    results["fp32"] = benchmark(fp32, images, warmup, iterations)
    results["fp32"]["model_size_mb"] = model_state_size_mb(fp32)
    results["fp32"]["accuracy"] = evaluate_model(
        fp32,
        build_coco_loader(config, "test", shuffle=False, limit=args.images, batch_size=1),
        torch.device("cpu"), include_rpn=False,
    )
    results["fp32"]["accuracy_images"] = args.images
    del fp32
    gc.collect()

    if args.eager_int8_checkpoint:
        eager = load_model(config, "int8", args.eager_int8_checkpoint, torch.device("cpu"))
        results["eager_m3"] = benchmark(eager, images, warmup, iterations)
        results["eager_m3"]["model_size_mb"] = model_state_size_mb(eager)
        results["eager_m3"]["accuracy"] = evaluate_model(
            eager,
            build_coco_loader(config, "test", shuffle=False, limit=args.images, batch_size=1),
            torch.device("cpu"), include_rpn=False,
        )
        results["eager_m3"]["accuracy_images"] = args.images
        del eager
        gc.collect()

    pt2e, artifact = load_pt2e_int8_artifact(args.pt2e_int8_checkpoint, config)
    print("Evaluating converted PT2E INT8 artifact before latency benchmark...", flush=True)
    pt2e_accuracy = evaluate_model(
        pt2e, build_coco_loader(
            config, "test", shuffle=False, limit=args.images, batch_size=1,
        ),
        torch.device("cpu"), include_rpn=False,
    )
    pt2e_size = model_state_size_mb(pt2e)
    if args.compile:
        compile_pt2e_region(pt2e)
    results["pt2e_backbone"] = benchmark(pt2e, images, warmup, iterations)
    results["pt2e_backbone"]["model_size_mb"] = pt2e_size
    results["pt2e_backbone"]["compiled"] = args.compile
    results["pt2e_backbone"]["accuracy"] = pt2e_accuracy
    results["pt2e_backbone"]["artifact_metadata"] = artifact.get("extra", {})
    results["pt2e_backbone"]["artifact_file_size_mb"] = checkpoint_size_mb(
        args.pt2e_int8_checkpoint,
    )
    results["pt2e_backbone"]["accuracy_images"] = args.images
    results["pt2e_speedup_vs_fp32"] = (
        results["fp32"]["latency_ms"] / results["pt2e_backbone"]["latency_ms"]
    )
    output = Path(args.output or Path(config["output"]["directory"]) / "pt2e_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
