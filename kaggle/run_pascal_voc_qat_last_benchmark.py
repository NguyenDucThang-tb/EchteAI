#!/usr/bin/env python3
"""Run Pascal VOC benchmark from an existing ResNet50 QAT checkpoint dataset.

Default checkpoint dataset:
  /kaggle/input/datasets/thngvs/pascal-voc-resnet50-qat-m3-true-checkpoints

The QAT checkpoint is a prepared fake-quant checkpoint. This script converts it
to an eager selective INT8 checkpoint before running the existing evaluation and
CPU latency benchmark scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CHECKPOINT_DIR = Path(
    "/kaggle/input/datasets/thngvs/pascal-voc-resnet50-qat-m3-true-checkpoints"
)
FALLBACK_CHECKPOINT_DIR = Path("/kaggle/input/pascal-voc-resnet50-qat-m3-true-checkpoints")


def run(cmd: list[str], cwd: Path) -> None:
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def find_vocdevkit(root: Path) -> Path:
    candidates = []
    for path in root.rglob("VOCdevkit"):
        if (path / "VOC2007").is_dir() and (path / "VOC2012").is_dir():
            candidates.append(path)
    if not candidates:
        for path in root.rglob("*"):
            if path.is_dir() and (path / "VOC2007").is_dir() and (path / "VOC2012").is_dir():
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("Cannot find VOCdevkit with VOC2007 and VOC2012 under /kaggle/input")
    return sorted({candidate.resolve() for candidate in candidates})[0]


def checkpoint_dir_from_args(value: str | None) -> Path:
    if value:
        return Path(value)
    if DEFAULT_CHECKPOINT_DIR.exists():
        return DEFAULT_CHECKPOINT_DIR
    return FALLBACK_CHECKPOINT_DIR


def build_runtime_config(
    repo: Path,
    voc_root: Path,
    voc_coco_root: Path,
    work_root: Path,
    checkpoint_dir: Path,
    runtime_config: Path,
) -> None:
    base_config = repo / "configs" / "seadronessee_resnet50_fp32_2class_1080x1920.yaml"
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))

    config["dataset"]["train_images"] = str(voc_root)
    config["dataset"]["train_annotations"] = str(voc_coco_root / "instances_train.json")
    config["dataset"]["val_images"] = str(voc_root)
    config["dataset"]["val_annotations"] = str(voc_coco_root / "instances_val.json")
    config["dataset"]["test_images"] = str(voc_root)
    config["dataset"]["test_annotations"] = str(voc_coco_root / "instances_val.json")
    config["dataset"]["ignore_category_ids"] = []
    config["dataset"]["num_classes"] = 21
    config["dataset"]["binary_collapse_foreground"] = False
    config["dataset"]["workers"] = 2

    config["model"]["backbone"] = "resnet50"
    config["model"]["trainable_backbone_layers"] = 5
    config["model"]["min_size"] = 800
    config["model"]["train_min_sizes"] = [640, 672, 704, 736, 768, 800]
    config["model"]["max_size"] = 1333
    config["model"]["anchor_statistics_min_size"] = 800

    config["training"]["batch_size"] = 1
    config["training"]["fp32_batch_size"] = 1
    config["training"]["qat_batch_size"] = 1
    config["training"]["epoch_benchmark_images"] = 100

    # This checkpoint dataset is intended to be true M3/W8A8. Disable inherited
    # HAWQ mixed precision so conversion does not try to apply 4-bit policy.
    config["quantization"]["variant"] = "M3"
    config["quantization"]["quantized_modules"] = [
        "backbone.resnet50",
        "backbone.fpn",
        "rpn.shared_conv",
        "rpn.classification",
    ]
    config["quantization"]["mixed_precision"]["enabled"] = False
    config["quantization"]["mixed_precision"]["policy_path"] = None
    config["quantization"]["mixed_precision"]["policy_output"] = None
    config["quantization"]["compiler"]["example_height"] = 800
    config["quantization"]["compiler"]["example_width"] = 1333
    config["quantization"]["compiler"]["int8_reference_checkpoint"] = str(work_root / "selective_int8.pt")

    config["output"]["directory"] = str(work_root)
    config["output"]["fp32_best"] = str(checkpoint_dir / "fp32_best.pt")
    config["output"]["fp32_last"] = str(work_root / "fp32_last.pt")
    config["output"]["qat_best"] = str(work_root / "qat_best.pt")
    config["output"]["qat_last"] = str(checkpoint_dir / "qat_last.pt")
    config["output"]["int8_model"] = str(work_root / "selective_int8.pt")
    config["output"]["evaluation_json"] = str(work_root / "evaluation.json")
    config["output"]["benchmark_json"] = str(work_root / "benchmark.json")
    config["output"]["epoch_benchmarks"] = str(checkpoint_dir / "epoch_benchmarks.json")

    runtime_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/kaggle/working/EchteAI")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--work-root", default="/kaggle/working/pascal_voc_qat_last_benchmark")
    parser.add_argument("--voc-coco-root", default="/kaggle/working/pascal_voc_coco")
    parser.add_argument("--limit", type=int, help="Optional image limit for quick smoke tests")
    parser.add_argument("--force-w8a8", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo)
    work_root = Path(args.work_root)
    voc_coco_root = Path(args.voc_coco_root)
    checkpoint_dir = checkpoint_dir_from_args(args.checkpoint_dir)
    runtime_config = work_root / "runtime_pascal_voc_resnet50_qat_last.yaml"

    if not (checkpoint_dir / "qat_last.pt").is_file():
        raise FileNotFoundError(f"Missing qat_last.pt in {checkpoint_dir}")
    if not (checkpoint_dir / "fp32_best.pt").is_file():
        raise FileNotFoundError(f"Missing fp32_best.pt in {checkpoint_dir}")

    work_root.mkdir(parents=True, exist_ok=True)
    voc_coco_root.mkdir(parents=True, exist_ok=True)
    voc_root = find_vocdevkit(Path("/kaggle/input"))

    run(
        [
            sys.executable,
            "-u",
            "scripts/convert_pascal_voc_to_coco.py",
            "--voc-root",
            str(voc_root),
            "--output-dir",
            str(voc_coco_root),
            "--train-sets",
            "VOC2007:trainval",
            "VOC2012:trainval",
            "--val-sets",
            "VOC2007:test",
        ],
        repo,
    )
    build_runtime_config(repo, voc_root, voc_coco_root, work_root, checkpoint_dir, runtime_config)

    print(f"VOC root: {voc_root}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Runtime config: {runtime_config}")

    convert_cmd = [
        sys.executable,
        "-u",
        "scripts/convert_resnet50_qat_to_int8.py",
        "--config",
        str(runtime_config),
        "--qat-checkpoint",
        str(checkpoint_dir / "qat_last.pt"),
        "--output",
        str(work_root / "selective_int8.pt"),
    ]
    if args.force_w8a8:
        convert_cmd.append("--force-w8a8")
    run(convert_cmd, repo)

    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            "-u",
            "scripts/evaluate.py",
            "--config",
            str(runtime_config),
            "--model",
            "int8",
            "--checkpoint",
            str(work_root / "selective_int8.pt"),
            "--split",
            "val",
            "--output",
            str(work_root / "evaluation_int8.json"),
        ]
        if args.limit:
            eval_cmd.extend(["--limit", str(args.limit)])
        run(eval_cmd, repo)

    if not args.skip_latency:
        run(
            [
                sys.executable,
                "-u",
                "scripts/benchmark.py",
                "--config",
                str(runtime_config),
                "--fp32-checkpoint",
                str(checkpoint_dir / "fp32_best.pt"),
                "--int8-checkpoint",
                str(work_root / "selective_int8.pt"),
                "--output",
                str(work_root / "benchmark_fp32_vs_int8.json"),
            ],
            repo,
        )

    outputs = {
        "runtime_config": str(runtime_config),
        "int8_checkpoint": str(work_root / "selective_int8.pt"),
        "evaluation": str(work_root / "evaluation_int8.json"),
        "latency_benchmark": str(work_root / "benchmark_fp32_vs_int8.json"),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
