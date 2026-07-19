#!/usr/bin/env python3
"""Khoanh vùng lỗi PT2E giữa trước convert, sau convert và sau reload artifact."""

import argparse
import gc
import json
import statistics
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import checkpoint_size_mb, load_checkpoint, model_state_size_mb
from pipelines.convnext_qat.config import load_config
from pipelines.convnext_qat.data import build_coco_loader
from pipelines.convnext_qat.metrics import evaluate_model
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import (
    convert_pt2e_backbone,
    load_pt2e_int8_artifact,
    prepare_pt2e_backbone_qat,
    save_pt2e_int8_artifact,
    set_pt2e_qat_phase,
)


def parse_args():
    """Đọc checkpoint QAT, số ảnh kiểm tra và thiết bị chạy debug."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--qat-checkpoint", required=True)
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--sample-images", type=int, default=5)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--reload-atol", type=float, default=1e-5)
    parser.add_argument(
        "--allow-reload-mismatch", action="store_true",
        help="write diagnostics instead of failing when in-memory and reloaded outputs differ",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def choose_device(name):
    """Chọn CPU/CUDA và báo lỗi rõ nếu CUDA không tồn tại."""
    device = torch.device(name)
    if device.type != "cpu":
        raise ValueError(
            "PT2E X86Inductor converted inference is CPU-only; use --device cpu for a "
            "valid pre/post/reload comparison"
        )
    return device


def build_prepared_model(config, checkpoint, device):
    """Dựng lại graph prepared rồi nạp checkpoint PT2E QAT."""
    model = build_fasterrcnn_convnext(config)
    model = prepare_pt2e_backbone_qat(model, config)
    payload = load_checkpoint(checkpoint, model, map_location="cpu")
    set_pt2e_qat_phase(model, "frozen")
    model = model.to(device).eval()
    return model, payload


@torch.inference_mode()
def sample_prediction_stats(model, loader, device, sample_images):
    """Thu boxes/labels/scores vài ảnh để so sánh chi tiết từng stage."""
    records = []
    for index, (images, targets) in enumerate(loader):
        if index >= sample_images:
            break
        outputs = model([image.to(device) for image in images])
        output = outputs[0]
        scores = output["scores"].detach().cpu()
        boxes = output["boxes"].detach().cpu()
        labels = output["labels"].detach().cpu()
        records.append({
            "image_index": index + 1,
            "gt_boxes": int(len(targets[0]["boxes"])),
            "pred_boxes": int(len(boxes)),
            "top_scores": [float(value) for value in scores[:10]],
            "top_labels": [int(value) for value in labels[:10]],
            "top_boxes": [[float(value) for value in box] for box in boxes[:10]],
        })
    return records


def aggregate_score_stats(records):
    """Tổng hợp score cao nhất và số detection từ các sample."""
    top1 = [item["top_scores"][0] for item in records if item["top_scores"]]
    pred_boxes = [item["pred_boxes"] for item in records]
    if not top1:
        return {"top1_mean": 0.0, "top1_median": 0.0, "pred_boxes_mean": 0.0}
    return {
        "top1_mean": float(statistics.mean(top1)),
        "top1_median": float(statistics.median(top1)),
        "pred_boxes_mean": float(statistics.mean(pred_boxes)),
    }


def compare_sample_records(reference, candidate, atol=1e-5):
    """So prediction của hai stage và báo độ lệch boxes/scores/labels."""
    """Compare deterministic samples and expose artifact reload corruption clearly."""
    if len(reference) != len(candidate):
        return {"equivalent": False, "reason": "sample_count", "max_abs_delta": float("inf")}
    maximum = 0.0
    for expected, actual in zip(reference, candidate):
        if expected["pred_boxes"] != actual["pred_boxes"]:
            return {"equivalent": False, "reason": "prediction_count", "max_abs_delta": float("inf")}
        if expected["top_labels"] != actual["top_labels"]:
            return {"equivalent": False, "reason": "labels", "max_abs_delta": float("inf")}
        expected_values = expected["top_scores"] + [v for box in expected["top_boxes"] for v in box]
        actual_values = actual["top_scores"] + [v for box in actual["top_boxes"] for v in box]
        if len(expected_values) != len(actual_values):
            return {"equivalent": False, "reason": "tensor_length", "max_abs_delta": float("inf")}
        if expected_values:
            maximum = max(maximum, max(abs(a - b) for a, b in zip(expected_values, actual_values)))
    return {
        "equivalent": maximum <= float(atol),
        "reason": "ok" if maximum <= float(atol) else "numeric_delta",
        "max_abs_delta": maximum,
        "atol": float(atol),
    }


def evaluate_variant(name, model, config, device, limit, sample_images):
    """Chạy metrics và sample diagnostics cho một biến thể model."""
    metrics = evaluate_model(
        model,
        build_coco_loader(config, "test", shuffle=False, limit=limit, batch_size=1),
        device,
        include_rpn=False,
    )
    samples = sample_prediction_stats(
        model,
        build_coco_loader(config, "test", shuffle=False, limit=sample_images, batch_size=1),
        device,
        sample_images,
    )
    return {
        "metrics": metrics,
        "samples": samples,
        "summary": aggregate_score_stats(samples),
        "model_size_mb": model_state_size_mb(model),
        "variant": name,
    }


def main():
    """Đánh giá ba stage và lưu báo cáo xác định convert hay reload làm sai."""
    args = parse_args()
    config = load_config(args.config, require_dataset=True)
    device = choose_device(args.device)

    prepared_model, payload = build_prepared_model(config, args.qat_checkpoint, device)
    print(f"Loaded prepared PT2E checkpoint: {args.qat_checkpoint}", flush=True)
    print(f"Checkpoint epoch: {payload.get('epoch', 0)}", flush=True)

    results = {
        "config": str(Path(args.config).resolve()),
        "qat_checkpoint": str(Path(args.qat_checkpoint).resolve()),
        "device": str(device),
        "images": int(args.images),
        "sample_images": int(args.sample_images),
    }

    print("Evaluating pre-convert frozen QAT model...", flush=True)
    results["pre_convert"] = evaluate_variant(
        "pre_convert", prepared_model, config, device, args.images, args.sample_images,
    )

    print("Converting PT2E model in memory...", flush=True)
    converted_model = convert_pt2e_backbone(prepared_model, inplace=False, compile_region=False)
    converted_model = converted_model.to(device).eval()
    results["post_convert_memory"] = evaluate_variant(
        "post_convert_memory", converted_model, config, device, args.images, args.sample_images,
    )
    results["pre_convert_vs_post_convert"] = compare_sample_records(
        results["pre_convert"]["samples"],
        results["post_convert_memory"]["samples"],
        args.reload_atol,
    )

    with tempfile.TemporaryDirectory() as directory:
        artifact_path = Path(directory) / "pt2e_debug_int8.pt"
        save_pt2e_int8_artifact(
            artifact_path,
            converted_model.cpu(),
            metrics=results["post_convert_memory"]["metrics"],
            extra={
                "source_qat": str(Path(args.qat_checkpoint).resolve()),
                "source_epoch": payload.get("epoch", 0),
                "region": getattr(prepared_model, "pt2e_quantized_region", "backbone.body"),
            },
        )
        results["artifact_size_mb"] = checkpoint_size_mb(artifact_path)
        print(f"Saved temporary artifact: {artifact_path}", flush=True)
        reloaded_model, reloaded_payload = load_pt2e_int8_artifact(artifact_path, config)
        reloaded_model = reloaded_model.to(device).eval()
        results["reloaded_artifact"] = evaluate_variant(
            "reloaded_artifact", reloaded_model, config, device, args.images, args.sample_images,
        )
        results["post_convert_vs_reload"] = compare_sample_records(
            results["post_convert_memory"]["samples"],
            results["reloaded_artifact"]["samples"],
            args.reload_atol,
        )
        results["artifact_metadata"] = reloaded_payload.get("extra", {})
        del reloaded_model
        gc.collect()

    output = Path(args.output or (Path(config["output"]["directory"]) / "pt2e_debug_convert.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    reload_check = results["post_convert_vs_reload"]
    if not reload_check["equivalent"] and not args.allow_reload_mismatch:
        raise RuntimeError(
            "PT2E artifact reload changed converted predictions: "
            f"{reload_check}. Full diagnostics were saved to {output}"
        )

    print(json.dumps({
        "pre_convert_map_50_95": results["pre_convert"]["metrics"]["map_50_95"],
        "post_convert_memory_map_50_95": results["post_convert_memory"]["metrics"]["map_50_95"],
        "reloaded_artifact_map_50_95": results["reloaded_artifact"]["metrics"]["map_50_95"],
        "pre_convert_top1_mean": results["pre_convert"]["summary"]["top1_mean"],
        "post_convert_memory_top1_mean": results["post_convert_memory"]["summary"]["top1_mean"],
        "reloaded_artifact_top1_mean": results["reloaded_artifact"]["summary"]["top1_mean"],
        "artifact_size_mb": results["artifact_size_mb"],
        "post_convert_vs_reload": reload_check,
        "pre_convert_vs_post_convert": results["pre_convert_vs_post_convert"],
        "saved": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
