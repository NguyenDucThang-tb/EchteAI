# EchteAI

## Selective-QAT Faster R-CNN with ConvNeXt-FPN

This repository includes a complete COCO training pipeline for a ConvNeXt-Tiny
(or ConvNeXt-Small) Faster R-CNN. Quantization is deliberately bounded: M3, the
default, converts ConvNeXt Conv/Linear operations, FPN convolutions, the shared
RPN convolution and RPN classifier to INT8. ConvNeXt LayerNorm/GELU/residual
math, transforms, anchors, RPN regression/decode/NMS, ROI Align, ROI heads and
final postprocessing remain FP32.

Each selected operation is surrounded by a QuantStub/DeQuantStub pair. During
QAT this simulates activation and per-channel symmetric weight quantization;
after conversion it executes a real quantized CPU kernel and returns FP32 at the
boundary. This avoids sending quantized tensors into unsupported detection and
normalization operators.

Requirements are Python 3.10+, PyTorch, torchvision, Pillow and PyYAML.
`pycocotools` is optional: when present, evaluation reports canonical COCO AP;
otherwise the included 101-point metric implementation is used.

Configure COCO paths, class count (including background), anchors and training
settings in `configs/fasterrcnn_convnext_qat.yaml`. COCO category IDs are
remapped to contiguous labels for training and mapped back for evaluation.

```bash
cd /home/nguyenducthang/EchteAI

# FP32 baseline; best checkpoint is selected by validation mAP.
python scripts/train_fp32.py --config configs/fasterrcnn_convnext_qat.yaml

# Observer calibration, staged QAT, observer freeze, and INT8 conversion.
python scripts/train_qat.py --config configs/fasterrcnn_convnext_qat.yaml

# Ablations: M0 FP32, M1 backbone, M2 +FPN, M3 +RPN conv/cls, M4 full RPN.
python scripts/train_qat.py --config configs/fasterrcnn_convnext_qat.yaml --variant M2

# Evaluate both checkpoints on the same split.
python scripts/evaluate.py --model fp32 --split test
python scripts/evaluate.py --model int8 --split test

# CPU latency, FPS, model size, and speedup JSON.
python scripts/benchmark.py --config configs/fasterrcnn_convnext_qat.yaml
```

Use `--limit N` for quick subset experiments. Converted INT8 checkpoints are
CPU-only. If pretrained weights are not cached and the machine is offline, set
`model.pretrained_backbone` to `false`.

Evaluation reports mAP@0.5, mAP@0.5:0.95, size AP, precision, recall, and RPN
Recall@100/@300/@1000, F1, and proposal statistics. The benchmark reports fair
model-only size, parameter count, latency p50/p95, FPS, and peak RSS after
configurable warmup/timed iterations.

See [`docs/PIPELINE_REVIEW_VI.md`](docs/PIPELINE_REVIEW_VI.md) for the directory
map, execution flow, quantization boundaries, checkpoint format, and a detailed
Vietnamese code review.

## Google Colab / SeaDronesSee v2

Use [`colab/SeaDronesSee_ConvNeXt_QAT.ipynb`](colab/SeaDronesSee_ConvNeXt_QAT.ipynb)
for the complete resumable Colab workflow. It downloads only the compressed
dataset to Colab local SSD and stores every durable checkpoint/result in Google
Drive. The matching configuration is
[`configs/seadronessee_colab.yaml`](configs/seadronessee_colab.yaml).
