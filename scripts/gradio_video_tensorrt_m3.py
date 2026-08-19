#!/usr/bin/env python3
"""Gradio demo: SeaDronesSee FP32 PyTorch CUDA vs INT8 TensorRT M3 hybrid.

Ứng dụng nhận một video, chạy hai detector trên cùng GPU rồi trả về video ghép:

* bên trái: Faster R-CNN FP32 bằng PyTorch CUDA;
* bên phải: selective-QAT M3 được export thành TensorRT INT8 cho
  backbone + FPN + RPN shared/classification, phần còn lại chạy PyTorch CUDA.

Checkpoint mặc định được tải từ Kaggle Dataset
``nguyenducthangtb/echteai-seadronessee-m3-checkpoints``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torchvision.models.detection.image_list import ImageList
from torchvision.transforms.functional import to_tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint
from pipelines.convnext_qat.config import load_config
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from scripts.benchmark_convnext_tensorrt_hybrid import (
    M3_SCOPE,
    TensorRTBackboneRunner,
    _features_from_fpn_tuple,
    _rpn_proposals_from_precomputed_head,
    _split_m3_outputs,
    load_qat_model,
)


DEFAULT_DATASET = "nguyenducthangtb/echteai-seadronessee-m3-checkpoints"
DEFAULT_ANCHORS = [8, 14, 24, 44, 99]
DEFAULT_LABELS = ["class 1", "class 2", "class 3", "class 4", "class 5"]
COLORS = [(50, 220, 50), (0, 190, 255), (255, 140, 30), (230, 60, 220), (200, 255, 40)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/seadronessee_colab.yaml"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint-dir", help="Thư mục đã chứa fp32_best/last.pt và qat_best/last.pt")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--qat-checkpoint")
    parser.add_argument("--engine", help="TensorRT M3 INT8 engine có sẵn")
    parser.add_argument("--work-dir", default="/kaggle/working/echteai_gradio_m3")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Tạo public Gradio URL (cần trên Kaggle)")
    parser.add_argument("--rebuild-engine", action="store_true")
    return parser.parse_args()


def _find_checkpoint(root: Path, names: tuple[str, ...]) -> Path:
    # Thứ tự names thể hiện ưu tiên: best trước, last sau. Không torch.load chỉ
    # để đọc epoch vì mỗi checkpoint có thể lớn hơn 500 MB.
    for name in names:
        candidates = sorted(root.rglob(name))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"Không tìm thấy {names} trong {root}")


def resolve_checkpoint_root(dataset: str, checkpoint_dir: str | None) -> Path:
    if checkpoint_dir:
        root = Path(checkpoint_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {root}")
        return root

    # Kaggle thường mount dataset ở một trong hai layout này. Dùng mount trước
    # để tránh download lại hàng trăm MB checkpoint.
    slug = dataset.split("/", 1)[-1]
    mounted = (
        Path("/kaggle/input/datasets") / dataset,
        Path("/kaggle/input") / slug,
    )
    for root in mounted:
        if root.exists():
            print(f"Using mounted Kaggle checkpoint dataset: {root}", flush=True)
            return root

    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError("Cần cài kagglehub để tải checkpoint dataset.") from error
    print(f"Downloading Kaggle checkpoint dataset: {dataset}", flush=True)
    return Path(kagglehub.dataset_download(dataset)).resolve()


def make_runtime_config(source: str | Path, output: Path, fp32: Path, qat: Path,
                        height: int, width: int) -> tuple[dict, Path]:
    config = load_config(source, require_dataset=False)
    config["device"] = "cuda"
    config["model"]["pretrained_backbone"] = False
    # Checkpoint SeaDronesSee được train với chính bộ anchor này. Đặt cố định để
    # web demo không cần instances_train.json chỉ để suy lại anchor.
    config["model"]["anchor_sizes"] = list(DEFAULT_ANCHORS)
    config["quantization"]["variant"] = "M3"
    config["quantization"]["compiler"] = {
        **config["quantization"].get("compiler", {}),
        "scope": M3_SCOPE,
        "example_batch_size": 1,
        "example_height": int(height),
        "example_width": int(width),
        "artifact_dir": str(output / "tensorrt_artifacts"),
    }
    config["output"] = {
        **config["output"],
        "directory": str(output),
        "fp32_best": str(fp32),
        "fp32_last": str(fp32),
        "qat_best": str(qat),
        "qat_last": str(qat),
    }
    runtime_path = output / "runtime_gradio_m3.yaml"
    runtime_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, runtime_path


def _run(command: list[str]):
    print("Command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def build_int8_engine(runtime_config: Path, qat_checkpoint: Path, artifact_dir: Path,
                      height: int, width: int, workspace_mb: int) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = artifact_dir / "seadronessee_m3_qat_int8_qdq.onnx"
    engine_path = artifact_dir / "seadronessee_m3_int8.engine"
    _run([
        sys.executable, "-u", "scripts/export_convnext_tensorrt_onnx.py",
        "--config", str(runtime_config),
        "--model", "qat_graph",
        "--qat-checkpoint", str(qat_checkpoint),
        "--output", str(onnx_path),
        "--height", str(height),
        "--width", str(width),
        "--batch-size", "1",
        "--tensorrt-friendly-int8",
    ])
    _run([
        sys.executable, "-u", "scripts/build_tensorrt_engine.py",
        "--onnx", str(onnx_path),
        "--engine", str(engine_path),
        "--precision", "int8",
        "--workspace-mb", str(workspace_mb),
    ])
    return engine_path


def _validate_m3_engine(runner: TensorRTBackboneRunner, engine_path: Path):
    outputs = len(runner.output_names)
    if outputs < 3 or outputs % 3:
        raise RuntimeError(
            f"Engine {engine_path} có {outputs} output, không phải engine M3. "
            "M3 phải trả [FPN, RPN-shared, RPN-objectness] với 3*N output."
        )


class VideoComparisonService:
    """Giữ model/engine trong VRAM và xử lý tuần tự các request Gradio."""

    def __init__(self, config: dict, fp32_checkpoint: Path, qat_checkpoint: Path,
                 engine_path: Path, output_dir: Path, height: int, width: int):
        if not torch.cuda.is_available():
            raise RuntimeError("App TensorRT cần Kaggle Accelerator = GPU T4.")
        self.device = torch.device("cuda:0")
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.height = int(height)
        self.width = int(width)

        print(f"Loading FP32 PyTorch CUDA checkpoint: {fp32_checkpoint}", flush=True)
        self.fp32_model = build_fasterrcnn_convnext(config)
        load_checkpoint(fp32_checkpoint, self.fp32_model, map_location="cpu", strict=True)
        self.fp32_model = self.fp32_model.to(self.device).eval()

        print(f"Loading QAT M3 topology: {qat_checkpoint}", flush=True)
        self.hybrid_model = load_qat_model(config, qat_checkpoint, self.device)
        print(f"Loading TensorRT M3 INT8 engine: {engine_path}", flush=True)
        self.int8_runner = TensorRTBackboneRunner(engine_path)
        _validate_m3_engine(self.int8_runner, engine_path)

    @torch.inference_mode()
    def _predict_fp32(self, image: torch.Tensor):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        output = self.fp32_model([image.to(self.device)])[0]
        torch.cuda.synchronize(self.device)
        return output, (time.perf_counter() - started) * 1000.0

    def _preprocess_m3(self, image: torch.Tensor):
        mean = torch.tensor(self.hybrid_model.transform.image_mean, device=self.device).view(-1, 1, 1)
        std = torch.tensor(self.hybrid_model.transform.image_std, device=self.device).view(-1, 1, 1)
        image = image.to(self.device)
        original_size = tuple(int(value) for value in image.shape[-2:])
        image = (image - mean) / std
        image = F.interpolate(
            image.unsqueeze(0), size=(self.height, self.width),
            mode="bilinear", align_corners=False,
        )
        return ImageList(image, [(self.height, self.width)]), [original_size]

    @torch.inference_mode()
    def _predict_int8_m3(self, image: torch.Tensor):
        image_list, original_sizes = self._preprocess_m3(image)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        trt_outputs = self.int8_runner(image_list.tensors)
        fpn_features, shared_features, objectness = _split_m3_outputs(trt_outputs)
        features = _features_from_fpn_tuple(fpn_features)
        pred_bbox_deltas = [
            self.hybrid_model.rpn.head.bbox_pred(feature)
            for feature in shared_features
        ]
        proposals = _rpn_proposals_from_precomputed_head(
            self.hybrid_model, image_list, features, objectness, pred_bbox_deltas,
        )
        outputs, _ = self.hybrid_model.roi_heads(
            features, proposals, image_list.image_sizes, None,
        )
        outputs = self.hybrid_model.transform.postprocess(
            outputs, image_list.image_sizes, original_sizes,
        )
        torch.cuda.synchronize(self.device)
        return outputs[0], (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _draw(frame, prediction, title: str, latency_ms: float, threshold: float,
              labels: list[str]):
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
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(canvas.shape[1] - 1, x2), min(canvas.shape[0] - 1, y2)
            color = COLORS[(int(class_id) - 1) % len(COLORS)]
            label = labels[int(class_id) - 1] if 0 < int(class_id) <= len(labels) else f"class {class_id}"
            text = f"{label} {float(score):.2f}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, text, (x1, max(22, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, color, 2, cv2.LINE_AA)
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        header = f"{title} | {latency_ms:.1f} ms | {fps:.2f} FPS | detections={kept}"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (20, 20, 20), -1)
        cv2.putText(canvas, header, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas, kept

    @staticmethod
    def _metrics_table(metrics: dict) -> pd.DataFrame:
        rows = []
        fp32 = metrics.get("fp32_pytorch_cuda")
        int8 = metrics.get("int8_tensorrt_m3_hybrid")
        if fp32:
            rows.append([
                "FP32 PyTorch CUDA",
                round(float(fp32["mean_latency_ms"]), 2),
                round(float(fp32["fps"]), 2),
                int(fp32["detections"]),
            ])
        if int8:
            rows.append([
                "INT8 TensorRT M3 hybrid",
                round(float(int8["mean_latency_ms"]), 2),
                round(float(int8["fps"]), 2),
                int(int8["detections"]),
            ])
        return pd.DataFrame(rows, columns=["Model", "Latency (ms)", "FPS", "Detections"])

    @staticmethod
    def _metrics_html(metrics: dict) -> str:
        mode = metrics.get("mode", "n/a").upper()
        gpu = metrics.get("gpu", "n/a")
        source_fps = float(metrics.get("source_fps", 0.0))
        processed = int(metrics.get("processed_frames", 0))
        peak_vram = float(metrics.get("peak_vram_gb", 0.0))
        score = float(metrics.get("score_threshold", 0.0))
        output_fps = float(metrics.get("output_fps", 0.0))
        fp32 = metrics.get("fp32_pytorch_cuda")
        int8 = metrics.get("int8_tensorrt_m3_hybrid")
        speedup = metrics.get("speedup")

        def card(title: str, value: str, tint: str) -> str:
            return (
                f"<div style='flex:1;min-width:170px;padding:14px 16px;border-radius:16px;"
                f"border:1px solid #e5e7eb;background:linear-gradient(180deg,#fff,{tint}14);'>"
                f"<div style='font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.04em;'>{title}</div>"
                f"<div style='font-size:24px;font-weight:700;color:#111827;margin-top:6px;'>{value}</div>"
                f"</div>"
            )

        top_cards = [
            card("Mode", mode, "#f97316"),
            card("GPU", gpu, "#22c55e"),
            card("Processed", f"{processed}", "#3b82f6"),
            card("Peak VRAM", f"{peak_vram:.2f} GB", "#a855f7"),
            card("Output FPS", f"{output_fps:.1f}", "#06b6d4"),
        ]
        if speedup is not None:
            top_cards.append(card("Speedup", f"{float(speedup):.2f}x", "#f59e0b"))

        sections = [
            "<div style='display:flex;flex-wrap:wrap;gap:12px;margin-bottom:14px;'>",
            *top_cards,
            "</div>",
            "<div style='border:1px solid #e5e7eb;border-radius:16px;padding:14px 16px;background:#fff;'>",
            "<div style='font-weight:700;font-size:16px;margin-bottom:8px;'>Thông số chính</div>",
            "<table style='width:100%;border-collapse:collapse;font-size:14px;'>",
            f"<tr><td style='padding:6px 0;color:#6b7280;'>Score threshold</td><td style='padding:6px 0;text-align:right;font-weight:600;'>{score:.2f}</td></tr>",
            f"<tr><td style='padding:6px 0;color:#6b7280;'>Source FPS</td><td style='padding:6px 0;text-align:right;font-weight:600;'>{source_fps:.1f}</td></tr>",
        ]
        if fp32:
            sections.extend([
                f"<tr><td style='padding:6px 0;color:#6b7280;'>FP32 latency</td><td style='padding:6px 0;text-align:right;font-weight:600;'>{float(fp32['mean_latency_ms']):.2f} ms</td></tr>",
                f"<tr><td style='padding:6px 0;color:#6b7280;'>FP32 FPS</td><td style='padding:6px 0;text-align:right;font-weight:600;'>{float(fp32['fps']):.2f}</td></tr>",
            ])
        if int8:
            sections.extend([
                f"<tr><td style='padding:6px 0;color:#6b7280;'>INT8 latency</td><td style='padding:6px 0;text-align:right;font-weight:600;'>{float(int8['mean_latency_ms']):.2f} ms</td></tr>",
                f"<tr><td style='padding:6px 0;color:#6b7280;'>INT8 FPS</td><td style='padding:6px 0;text-align:right;font-weight:600;'>{float(int8['fps']):.2f}</td></tr>",
            ])
        sections.extend([
            "</table>",
            "</div>",
        ])
        return "".join(sections)

    @staticmethod
    def _browser_mp4(raw_path: Path, final_path: Path) -> Path:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raw_path.replace(final_path)
            return final_path
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_path),
        ]
        subprocess.run(command, check=True)
        raw_path.unlink(missing_ok=True)
        return final_path

    def _process(self, mode, video_path, score_threshold, frame_stride, target_fps,
                 max_frames, labels_text, progress):
        if not video_path:
            raise gr.Error("Hãy tải một video lên trước.")
        source = Path(video_path)
        if not source.is_file():
            raise gr.Error(f"Không đọc được video: {source}")

        if mode not in {"comparison", "fp32", "int8"}:
            raise ValueError(f"Unsupported processing mode: {mode}")

        labels = [value.strip() for value in str(labels_text).split(",") if value.strip()]
        labels = labels or DEFAULT_LABELS
        stride = max(1, int(frame_stride))
        max_frames = 0 if max_frames is None else int(max_frames)
        max_count = None if max_frames <= 0 else max_frames
        torch.cuda.reset_peak_memory_stats(self.device)

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise gr.Error(f"OpenCV không mở được video: {source}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        total_source = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_fps = max(source_fps / stride, 1.0)
        expected = (total_source + stride - 1) // stride if total_source else 0
        if max_count is not None:
            expected = min(expected, max_count) if expected else max_count

        request_id = uuid.uuid4().hex[:10]
        raw_path = self.output_dir / f"{mode}_{request_id}_raw.mp4"
        final_path = self.output_dir / f"{mode}_{request_id}.mp4"
        output_width = width * 2 if mode == "comparison" else width
        writer = cv2.VideoWriter(
            str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps, (output_width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise gr.Error(f"Không tạo được output video: {raw_path}")

        timings = {"fp32": [], "int8": []}
        detections = {"fp32": 0, "int8": 0}
        input_index = processed = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or (max_count is not None and processed >= max_count):
                    break
                should_process = input_index % stride == 0
                if not should_process:
                    input_index += 1
                    continue
                input_index += 1

                image = to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                panels = []
                if mode in {"comparison", "fp32"}:
                    fp32_prediction, fp32_ms = self._predict_fp32(image)
                    fp32_frame, fp32_count = self._draw(
                        frame, fp32_prediction, "FP32 PyTorch CUDA", fp32_ms,
                        float(score_threshold), labels,
                    )
                    panels.append(fp32_frame)
                    timings["fp32"].append(fp32_ms)
                    detections["fp32"] += fp32_count
                if mode in {"comparison", "int8"}:
                    int8_prediction, int8_ms = self._predict_int8_m3(image)
                    int8_frame, int8_count = self._draw(
                        frame, int8_prediction, "INT8 TensorRT M3 hybrid", int8_ms,
                        float(score_threshold), labels,
                    )
                    panels.append(int8_frame)
                    timings["int8"].append(int8_ms)
                    detections["int8"] += int8_count
                writer.write(np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0])
                processed += 1
                if expected:
                    progress(min(processed / expected, 1.0), desc=f"Đang xử lý frame {processed}/{expected}")
        finally:
            capture.release()
            writer.release()

        if processed == 0:
            raw_path.unlink(missing_ok=True)
            raise gr.Error("Video không có frame hợp lệ.")
        self._browser_mp4(raw_path, final_path)

        metrics = {
            "mode": mode,
            "source_fps": source_fps,
            "processed_frames": processed,
            "output_fps": output_fps,
            "score_threshold": float(score_threshold),
            "sampling": f"fixed_stride_{stride}",
            "gpu": torch.cuda.get_device_name(self.device),
            "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 2**30,
        }
        if timings["fp32"]:
            fp32_mean = float(np.mean(timings["fp32"]))
            metrics["fp32_pytorch_cuda"] = {
                "mean_latency_ms": fp32_mean,
                "fps": 1000.0 / fp32_mean,
                "detections": detections["fp32"],
            }
        if timings["int8"]:
            int8_mean = float(np.mean(timings["int8"]))
            metrics["int8_tensorrt_m3_hybrid"] = {
                "mean_latency_ms": int8_mean,
                "fps": 1000.0 / int8_mean,
                "detections": detections["int8"],
            }
        if timings["fp32"] and timings["int8"]:
            metrics["speedup"] = fp32_mean / int8_mean
        final_path.with_suffix(".json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        return str(final_path), self._metrics_html(metrics), self._metrics_table(metrics)

    def process_comparison(self, video_path, score_threshold, frame_stride, max_frames,
                           labels_text, progress=gr.Progress(track_tqdm=False)):
        return self._process(
            "comparison", video_path, score_threshold, frame_stride, 0,
            max_frames, labels_text, progress,
        )

    def process_fp32(self, video_path, score_threshold, frame_stride, max_frames,
                     labels_text, progress=gr.Progress(track_tqdm=False)):
        return self._process(
            "fp32", video_path, score_threshold, frame_stride, 0,
            max_frames, labels_text, progress,
        )

    def process_int8(self, video_path, score_threshold, frame_stride, max_frames,
                     labels_text, progress=gr.Progress(track_tqdm=False)):
        return self._process(
            "int8", video_path, score_threshold, frame_stride, 0,
            max_frames, labels_text, progress,
        )

def build_demo(service: VideoComparisonService):
    with gr.Blocks(title="SeaDronesSee FP32 vs TensorRT INT8 M3") as demo:
        gr.Markdown(
            "# SeaDronesSee: FP32 PyTorch GPU vs INT8 TensorRT M3 hybrid\n"
            "Chạy so sánh hoặc chạy riêng từng model trên GPU T4."
        )
        with gr.Row():
            input_video = gr.Video(label="Video đầu vào", sources=["upload", "webcam"], format="mp4")
            output_video = gr.Video(label="Video kết quả")
        with gr.Row():
            threshold = gr.Slider(0.05, 0.95, value=0.40, step=0.05, label="Score threshold")
            stride = gr.Slider(1, 10, value=1, step=1, label="Frame stride (chế độ thường)")
            max_frames = gr.Number(value=0, precision=0, label="Max frames (0 = toàn bộ)")
        labels = gr.Textbox(
            value=", ".join(DEFAULT_LABELS),
            label="Tên 5 lớp foreground (phân cách bằng dấu phẩy)",
        )
        with gr.Row():
            compare_button = gr.Button("So sánh FP32 | INT8", variant="primary")
            fp32_button = gr.Button("Chạy riêng FP32")
            int8_button = gr.Button("Chạy riêng INT8")
        metrics_html = gr.HTML(label="Tóm tắt hiệu năng")
        metrics_table = gr.Dataframe(
            headers=["Model", "Latency (ms)", "FPS", "Detections"],
            datatype=["str", "number", "number", "number"],
            interactive=False,
            row_count=(1, "dynamic"),
            col_count=(4, "fixed"),
            label="Bảng tóm tắt",
        )
        regular_inputs = [input_video, threshold, stride, max_frames, labels]
        compare_button.click(
            service.process_comparison,
            inputs=regular_inputs,
            outputs=[output_video, metrics_html, metrics_table],
        )
        fp32_button.click(
            service.process_fp32,
            inputs=regular_inputs,
            outputs=[output_video, metrics_html, metrics_table],
        )
        int8_button.click(
            service.process_int8,
            inputs=regular_inputs,
            outputs=[output_video, metrics_html, metrics_table],
        )
    return demo


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = resolve_checkpoint_root(args.dataset, args.checkpoint_dir)
    fp32_checkpoint = Path(args.fp32_checkpoint).resolve() if args.fp32_checkpoint else _find_checkpoint(
        checkpoint_root, ("fp32_best.pt", "fp32_last.pt"),
    )
    qat_checkpoint = Path(args.qat_checkpoint).resolve() if args.qat_checkpoint else _find_checkpoint(
        checkpoint_root, ("qat_best.pt", "qat_last.pt"),
    )
    print(f"FP32 checkpoint: {fp32_checkpoint}", flush=True)
    print(f"QAT checkpoint: {qat_checkpoint}", flush=True)

    config, runtime_path = make_runtime_config(
        args.config, work_dir, fp32_checkpoint, qat_checkpoint, args.height, args.width,
    )
    artifact_dir = work_dir / "tensorrt_artifacts"
    engine_path = Path(args.engine).resolve() if args.engine else artifact_dir / "seadronessee_m3_int8.engine"
    if args.rebuild_engine or not engine_path.is_file():
        if not torch.cuda.is_available():
            raise RuntimeError("Hãy bật Kaggle Accelerator GPU trước khi build TensorRT engine.")
        engine_path = build_int8_engine(
            runtime_path, qat_checkpoint, artifact_dir,
            args.height, args.width, args.workspace_mb,
        )

    service = VideoComparisonService(
        config, fp32_checkpoint, qat_checkpoint, engine_path,
        work_dir / "outputs", args.height, args.width,
    )
    demo = build_demo(service)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        # Video được ghi vào work-dir của Kaggle, nằm ngoài cwd của repo.
        # Gradio 5+ chỉ phục vụ file ngoài cwd khi đường dẫn được cho phép rõ ràng.
        allowed_paths=[str(service.output_dir.resolve())],
    )


if __name__ == "__main__":
    main()
