#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from evaluate import load_model
from EchteAI.pipelines.convnext_qat.checkpoint import model_state_size_mb
from EchteAI.pipelines.convnext_qat.config import load_config, validate_dataset_paths
from EchteAI.pipelines.convnext_qat.data import build_coco_loader


@torch.inference_mode()
def benchmark(model, images, warmup, iterations):
    def rss_mb():
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        except ImportError:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    rss_before = rss_mb()
    for _ in range(warmup):
        model(images)
    timings = []
    peak_rss = rss_mb()
    for _ in range(iterations):
        started = time.perf_counter()
        model(images)
        timings.append(time.perf_counter() - started)
        peak_rss = max(peak_rss, rss_mb())
    latency = 1000.0 * sum(timings) / len(timings)
    ordered = sorted(timings)
    return {
        "latency_ms": latency,
        "latency_p50_ms": 1000.0 * ordered[len(ordered) // 2],
        "latency_p95_ms": 1000.0 * ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)],
        "fps": 1000.0 * len(images) / latency,
        "peak_rss_mb": peak_rss,
        "inference_memory_delta_mb": max(peak_rss - rss_before, 0.0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="CPU benchmark FP32 versus selective INT8")
    parser.add_argument("--config", default="configs/fasterrcnn_convnext_qat.yaml")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--int8-checkpoint")
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    validate_dataset_paths(config, ("test",))
    torch.set_num_threads(int(config.get("benchmark", {}).get("num_threads", 1)))
    images, _ = next(iter(build_coco_loader(config, "test", shuffle=False)))
    images = [image.cpu() for image in images]
    warmup = int(config.get("benchmark", {}).get("warmup_iterations", 10))
    iterations = int(config.get("benchmark", {}).get("iterations", 50))
    checkpoints = {
        "fp32": args.fp32_checkpoint or config["output"]["fp32_best"],
        "int8": args.int8_checkpoint or config["output"]["int8_model"],
    }
    results = {}
    for kind, checkpoint in checkpoints.items():
        model = load_model(config, kind, checkpoint, torch.device("cpu"))
        results[kind] = benchmark(model, images, warmup, iterations)
        results[kind]["model_size_mb"] = model_state_size_mb(model)
        results[kind]["parameters"] = int(model.logical_parameter_count)
        del model
        gc.collect()
    results["speedup"] = results["fp32"]["latency_ms"] / results["int8"]["latency_ms"]
    output = Path(args.output or config["output"]["benchmark_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
