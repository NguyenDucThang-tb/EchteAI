#!/usr/bin/env python3
"""Chạy Faster R-CNN FP32 và Selective-INT8 song song trên video."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_model
from pipelines.convnext_qat.config import load_config


COLORS = [(0, 220, 0), (0, 180, 255), (255, 120, 0), (255, 0, 180), (180, 255, 0)]


def parse_args():
    """Đọc đường dẫn video/checkpoint và tùy chọn lọc frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fasterrcnn_convnext_qat.yaml")
    parser.add_argument("--fp32-checkpoint", required=True)
    parser.add_argument("--int8-checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs/video_fp32_vs_int8.mp4")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-frequency", type=int, default=10)
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Foreground label names in model-label order (labels start at 1)")
    return parser.parse_args()


def annotate(frame, prediction, title, latency_ms, threshold, labels):
    """Vẽ bbox, nhãn, score và latency lên một frame."""
    canvas = frame.copy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    class_ids = prediction["labels"].detach().cpu().numpy()
    kept = 0
    for box, score, class_id in zip(boxes, scores, class_ids):
        if float(score) < threshold:
            continue
        kept += 1
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = COLORS[(int(class_id) - 1) % len(COLORS)]
        name = labels[int(class_id) - 1] if labels and 0 < int(class_id) <= len(labels) else f"class {class_id}"
        text = f"{name} {score:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 5, y1), color, -1)
        cv2.putText(canvas, text, (x1 + 2, max(th + 1, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    header = f"{title} | {latency_ms:.1f} ms | {1000.0 / latency_ms:.2f} FPS | detections={kept}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(canvas, header, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas, kept


@torch.inference_mode()
def predict(model, image, device):
    """Tiền xử lý một frame, suy luận và trả prediction cùng latency."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model([image.to(device)])[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, (time.perf_counter() - started) * 1000.0


def main():
    """Đọc video, suy luận hai model và ghi video so sánh hai khung."""
    args = parse_args()
    if args.threads <= 0 or args.frame_stride <= 0 or args.progress_frequency <= 0:
        raise ValueError("threads, frame-stride and progress-frequency must be positive")
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    # Both checkpoints contain the complete model state. Avoid an unnecessary
    # ImageNet download while constructing the empty topology for state loading.
    config["model"]["pretrained_backbone"] = False
    fp32_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    int8_device = torch.device("cpu")

    print(f"Loading FP32 on {fp32_device}: {args.fp32_checkpoint}", flush=True)
    fp32_model = load_model(config, "fp32", args.fp32_checkpoint, fp32_device)
    print(f"Loading INT8 on CPU: {args.int8_checkpoint}", flush=True)
    int8_model = load_model(config, "int8", args.int8_checkpoint, int8_device)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    input_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_input = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = (total_input + args.frame_stride - 1) // args.frame_stride
    if args.max_frames is not None:
        expected = min(expected, args.max_frames)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             input_fps / args.frame_stride, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output}")

    timings = {"fp32": [], "int8": []}
    detections = {"fp32": 0, "int8": 0}
    input_index = processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (args.max_frames is not None and processed >= args.max_frames):
                break
            if input_index % args.frame_stride:
                input_index += 1
                continue
            input_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = to_tensor(rgb)
            fp32_prediction, fp32_ms = predict(fp32_model, image, fp32_device)
            int8_prediction, int8_ms = predict(int8_model, image, int8_device)
            fp32_frame, fp32_count = annotate(frame, fp32_prediction, "FP32", fp32_ms,
                                               args.score_threshold, args.labels)
            int8_frame, int8_count = annotate(frame, int8_prediction, "INT8", int8_ms,
                                               args.score_threshold, args.labels)
            writer.write(np.concatenate((fp32_frame, int8_frame), axis=1))
            timings["fp32"].append(fp32_ms)
            timings["int8"].append(int8_ms)
            detections["fp32"] += fp32_count
            detections["int8"] += int8_count
            processed += 1
            if processed == 1 or processed % args.progress_frequency == 0 or processed == expected:
                print(f"progress={processed}/{expected} fp32={fp32_ms:.1f}ms int8={int8_ms:.1f}ms", flush=True)
    finally:
        capture.release()
        writer.release()

    if not processed:
        raise RuntimeError("No video frames were processed")
    result = {
        "video": str(Path(args.video).resolve()), "output": str(output.resolve()),
        "processed_frames": processed, "frame_stride": args.frame_stride,
        "score_threshold": args.score_threshold,
        "fp32": {"device": str(fp32_device), "mean_latency_ms": float(np.mean(timings["fp32"])),
                 "fps": 1000.0 / float(np.mean(timings["fp32"])), "detections": detections["fp32"]},
        "int8": {"device": "cpu", "mean_latency_ms": float(np.mean(timings["int8"])),
                 "fps": 1000.0 / float(np.mean(timings["int8"])), "detections": detections["int8"]},
    }
    result["cpu_speedup_int8_vs_fp32_note"] = (
        "Not directly comparable when FP32 device is CUDA; run with CUDA hidden for fair CPU speedup."
    )
    metrics_path = Path(args.metrics_output) if args.metrics_output else output.with_suffix(".json")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved video: {output}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
