# Review pipeline Selective QAT Faster R-CNN ConvNeXt-FPN

## 1. Cấu trúc sau khi dọn dẹp

```text
EchteAI/
├── configs/
│   └── fasterrcnn_convnext_qat.yaml
├── Python/EchteAI/pipelines/convnext_qat/
│   ├── models/
│   │   ├── convnext_fpn_backbone.py
│   │   └── fasterrcnn_convnext.py
│   ├── quantization/
│   │   └── selective_qat.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── metrics.py
│   └── checkpoint.py
├── scripts/
│   ├── train_fp32.py
│   ├── train_qat.py
│   ├── evaluate.py
│   └── benchmark.py
└── tests/
    └── smoke_selective_qat.py
```

Chỉ thư mục `pipelines/convnext_qat/quantization` còn chứa code quantization.
Các implementation cũ như quantized ResNet50, dynamic/PTQ, ONNX quantization,
Quark, notebook và Gradio comparison đã được bỏ.

## 2. Luồng tensor của model

```text
ảnh [0, 1] FP32
    │
    ▼
GeneralizedRCNNTransform FP32 (resize, normalize, batch)
    │
    ▼
ConvNeXt C2-C5 ── selected Conv/Linear INT8 islands
    │              LayerNorm/GELU/residual vẫn FP32
    ▼
FPN P2-P6 ────── selected Conv INT8 islands, cộng top-down FP32
    ├───────────────────────────────┐
    │                               │
    ▼                               ▼
RPN shared conv INT8          ROI Align FP32
    ├── cls logits INT8             │
    └── bbox delta FP32 (M3)         ▼
    │                          ROI heads FP32
    ▼                               │
anchor/decode/clip/NMS FP32          ▼
    └──────── proposals ────── final decode/NMS FP32
                                    │
                                    ▼
                         boxes, labels, scores
```

Mỗi phép toán được chọn dùng wrapper:

```python
class QuantizedOperation(nn.Module):
    def forward(self, x):
        return self.dequant(self.operation(self.quant(x)))
```

Khi QAT, `quant` và observer mô phỏng sai số INT8. Sau `convert`, `operation`
trở thành kernel quantized thật trên CPU. Output được dequantize ngay về FP32 để
LayerNorm, GELU, residual, FPN addition và detection ops không nhận tensor
quantized không tương thích. Đổi lại, đây là nhiều INT8 island thay vì một graph
INT8 liên tục; an toàn backend cao hơn nhưng có thêm overhead Quant/DeQuant.

## 3. Backbone và detector

`models/convnext_fpn_backbone.py` tạo ConvNeXt-Tiny hoặc Small. Các output sau
feature index `1, 3, 5, 7` lần lượt là C2-C5. FPN nhận channel
`[96, 192, 384, 768]`, đưa tất cả về 256 channel và thêm P6 bằng max-pool.

`trainable_backbone_layers` được hiểu theo bốn group:

- stem + stage C2;
- downsample + stage C3;
- downsample + stage C4;
- downsample + stage C5.

`models/fasterrcnn_convnext.py` ghép backbone với `FasterRCNN`, tạo một anchor
size cho mỗi P2-P6 và dùng ROI Align trên P2-P5. `num_classes` luôn gồm class
background.

## 4. QConfig và các ablation

`selective_qconfig()` cấu hình:

- activation: uint8, per-tensor affine, MovingAverageMinMaxObserver;
- weight: int8, per-channel symmetric, channel axis 0;
- bias/accumulator: do quantized backend quản lý bằng int32.

Các vùng được chọn nằm trong `VARIANT_REGIONS`:

| Variant | ConvNeXt | FPN | RPN shared | RPN cls | RPN bbox |
|---|---:|---:|---:|---:|---:|
| M0 | FP32 | FP32 | FP32 | FP32 | FP32 |
| M1 | INT8 | FP32 | FP32 | FP32 | FP32 |
| M2 | INT8 | INT8 | FP32 | FP32 | FP32 |
| M3 | INT8 | INT8 | INT8 | INT8 | FP32 |
| M4 | INT8 | INT8 | INT8 | INT8 | INT8 |

`prepare_selective_qat()` chỉ đệ quy thay `Conv2d/Linear` trong vùng tương ứng.
Nó không gán qconfig cho toàn Faster R-CNN, vì làm vậy sẽ quantize nhầm ROI heads
và postprocessing. M0 trả về model FP32 trực tiếp.

## 5. Các phase QAT

`scripts/train_qat.py` chạy theo thứ tự:

1. Build model FP32 và load checkpoint tốt nhất.
2. Wrap đúng module theo M0-M4 và gọi `prepare_qat`.
3. Calibration warmup: observer ON, fake quant OFF.
4. Weight-only warmup: weight fake quant ON, activation fake quant OFF.
5. Full QAT: weight/activation fake quant ON, observer ON.
6. Epoch cuối: fake quant ON, observer OFF để scale/zero-point ổn định.
7. Chuyển model về CPU, gọi `convert`, lưu selective INT8 checkpoint.

`set_qat_phase()` phân biệt weight fake quant bằng tên module kết thúc bằng
`weight_fake_quant`; mọi fake quant còn lại là activation.

## 6. Dataset và metric

`data.py` đọc COCO JSON mà không cần pycocotools. Category ID gốc có thể rời
rạc; loader map chúng về label liên tục `1..N`, giữ `0` cho background và map
ngược khi chạy COCO evaluation.

Train split áp dụng horizontal flip và color jitter nhẹ. Faster R-CNN transform
chọn ngẫu nhiên một `train_min_sizes` khi train, còn validation/test luôn dùng
size lớn nhất trong danh sách. Box được đổi tọa độ đồng bộ khi flip.

`metrics.py` có evaluator 101-point nội bộ cho mAP@0.5, mAP@0.5:0.95,
AP small/medium/large, precision, recall và F1. Nếu cài `pycocotools`, kết quả AP
canonical của COCO sẽ ghi đè AP nội bộ. RPN Recall@100/@300 được tính bằng cách
chạy transform → backbone → RPN rồi so proposal với ground truth tại IoU 0.5.
Pipeline cũng báo Recall@1000, số proposal trung bình và mean/median/p75 của
proposal IoU.

## 7. Checkpoint và load INT8

Checkpoint chứa `model`, `epoch`, `metrics`, `optimizer` (nếu có) và `extra`.
INT8 checkpoint ghi thêm `variant`, `backend`, `format`. Khi load INT8,
`scripts/evaluate.py` phải build model, prepare đúng variant, convert topology,
rồi mới load packed quantized parameters. Variant được đọc từ checkpoint nên
checkpoint M1/M2/M4 không bị load nhầm theo default M3 trong YAML.

## 8. Train, evaluate và benchmark

```bash
python scripts/train_fp32.py --config configs/fasterrcnn_convnext_qat.yaml
python scripts/train_qat.py --config configs/fasterrcnn_convnext_qat.yaml
python scripts/evaluate.py --model fp32 --split test
python scripts/evaluate.py --model int8 --split test
python scripts/benchmark.py --config configs/fasterrcnn_convnext_qat.yaml
```

FP32 training chọn checkpoint theo validation mAP. Benchmark bắt buộc CPU,
warmup trước khi đo, báo latency trung bình, FPS, model size và speedup ra JSON.
QAT cũng evaluate từng epoch; chỉ epoch frozen-observer được chọn làm best để
scale/zero-point đã cố định. `qat_best` được reload trước khi convert. Model size
được đo từ model state thuần, không gồm optimizer/checkpoint metadata. Benchmark
còn báo p50/p95 latency, logical parameter count và peak RSS.

## 9. Điểm cần lưu ý khi chạy thật

- Sửa toàn bộ dataset path và `num_classes` trong YAML trước khi train.
- INT8 converted model chỉ chạy CPU với backend đã lưu trong checkpoint.
- QAT vẫn có thể train GPU vì lúc đó model chỉ dùng fake quant; chỉ `convert`
  và inference INT8 mới chuyển CPU.
- Per-operation islands ưu tiên tính ổn định. Muốn giảm overhead hơn nữa cần
  viết graph/backend-specific fusion cho ConvNeXt và FPN, phức tạp hơn đáng kể.
- PyTorch eager quantized Conv/Linear lưu bias qua API ở FP32; backend lượng tử
  bias nội bộ theo input-scale × weight-scale và tích lũy integer. Vì vậy không
  diễn giải dtype bias trong state dict như dtype accumulator của kernel.
- `pycocotools` không bắt buộc, nhưng nên cài khi cần số liệu paper chuẩn COCO.
