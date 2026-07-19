#!/usr/bin/env python3
"""Tạo ảnh trực quan GT/FP32/INT8 đặt cạnh nhau để đối chiếu detection."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_model
from pipelines.convnext_qat.config import load_config
from pipelines.convnext_qat.data import build_coco_loader, unwrap_coco_dataset


COLORS = {"gt": "#22c55e", "fp32": "#3b82f6", "int8": "#f97316"}


def parse_args():
    """Đọc checkpoint, ngưỡng score, số ảnh và thư mục đầu ra."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fp32-checkpoint", required=True)
    parser.add_argument("--int8-checkpoint", required=True)
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def label_name(label, names):
    """Đổi category id thành tên hiển thị, có fallback khi thiếu mapping."""
    return names.get(int(label), f"class_{int(label)}")


def draw_panel(image, boxes, labels, scores, names, title, color, threshold=0.0, max_detections=100):
    """Vẽ một panel bbox sau khi lọc score và giới hạn số detection."""
    panel = to_pil_image(image.cpu()).convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    kept = []
    for box, label, score in zip(boxes, labels, scores):
        confidence = float(score)
        if confidence < threshold or len(kept) >= max_detections:
            continue
        coords = [float(value) for value in box]
        name = label_name(label, names)
        caption = name if title == "Ground Truth" else f"{name} {confidence:.2f}"
        draw.rectangle(coords, outline=color, width=3)
        left, top = coords[0], max(24, coords[1])
        text_box = draw.textbbox((left, top), caption, font=font, stroke_width=2)
        draw.rectangle(text_box, fill=color)
        draw.text((left, top), caption, fill="white", font=font, stroke_width=1, stroke_fill="black")
        kept.append({"label": int(label), "name": name, "score": confidence, "box": coords})
    draw.rectangle((0, 0, panel.width, 30), fill="#111827")
    draw.text((10, 8), f"{title} | boxes={len(kept)}", fill="white", font=font)
    return panel, kept


def combine_panels(panels):
    """Ghép các ảnh PIL cùng chiều cao theo phương ngang."""
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    combined = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        combined.paste(panel, (x, 0))
        x += panel.width
    return combined


@torch.inference_mode()
def predict(model, image, device):
    """Chạy suy luận một ảnh và đưa tensor kết quả về CPU."""
    started = time.perf_counter()
    prediction = model([image.to(device)])[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {key: value.cpu() for key, value in prediction.items()}, elapsed_ms


def main():
    """Sinh 100 ảnh so sánh (hoặc số ảnh CLI) và file manifest JSON."""
    args = parse_args()
    if args.images <= 0 or args.threads <= 0:
        raise ValueError("images and threads must be positive")
    torch.set_num_threads(args.threads)
    config = load_config(args.config, require_dataset=True)
    config["model"]["pretrained_backbone"] = False
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"Loading FP32 on CPU: {args.fp32_checkpoint}", flush=True)
    fp32_model = load_model(config, "fp32", args.fp32_checkpoint, device)
    print(f"Loading INT8 on CPU: {args.int8_checkpoint}", flush=True)
    int8_model = load_model(config, "int8", args.int8_checkpoint, device)
    loader = build_coco_loader(config, "val", shuffle=False, limit=args.images, batch_size=1)
    dataset = unwrap_coco_dataset(loader.dataset)
    names = dataset.label_to_name
    records = []

    for index, (images, targets) in enumerate(loader, 1):
        image, target = images[0], targets[0]
        fp32_prediction, fp32_ms = predict(fp32_model, image, device)
        int8_prediction, int8_ms = predict(int8_model, image, device)
        gt_scores = torch.ones(len(target["labels"]), dtype=torch.float32)
        gt_panel, gt_items = draw_panel(
            image, target["boxes"], target["labels"], gt_scores, names,
            "Ground Truth", COLORS["gt"], max_detections=args.max_detections,
        )
        fp32_panel, fp32_items = draw_panel(
            image, fp32_prediction["boxes"], fp32_prediction["labels"], fp32_prediction["scores"],
            names, f"FP32 {fp32_ms:.0f} ms", COLORS["fp32"], args.score_threshold, args.max_detections,
        )
        int8_panel, int8_items = draw_panel(
            image, int8_prediction["boxes"], int8_prediction["labels"], int8_prediction["scores"],
            names, f"INT8 {int8_ms:.0f} ms", COLORS["int8"], args.score_threshold, args.max_detections,
        )
        filename = f"{index:03d}_image_{int(target['image_id'])}.jpg"
        combine_panels((gt_panel, fp32_panel, int8_panel)).save(image_dir / filename, quality=90)
        records.append({
            "index": index,
            "image_id": int(target["image_id"]),
            "file": f"images/{filename}",
            "ground_truth": gt_items,
            "fp32": {"latency_ms": fp32_ms, "detections": fp32_items},
            "int8": {"latency_ms": int8_ms, "detections": int8_items},
        })
        if index == 1 or index % 10 == 0 or index == args.images:
            print(
                f"visualization progress: {index}/{args.images} "
                f"fp32={fp32_ms:.1f}ms int8={int8_ms:.1f}ms",
                flush=True,
            )

    summary = {
        "images": len(records),
        "score_threshold": args.score_threshold,
        "threads": args.threads,
        "legend": COLORS,
        "records": records,
    }
    manifest = output_dir / "visualization_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} visualizations to {image_dir}", flush=True)
    print(f"Saved manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
