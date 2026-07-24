# EchteAI: ConvNeXt Faster R-CNN QAT + TensorRT Hybrid

EchteAI là repository nghiên cứu và benchmark phát hiện đối tượng với Faster R-CNN, backbone ConvNeXt-Tiny và FPN. Dự án hiện có hai hướng chính:

1. **SeaDronesSee benchmark**: pipeline gốc dùng Faster R-CNN + ConvNeXt-Tiny + FPN, anchor tự động, focal loss và selective QAT.
2. **Pascal VOC QAT TensorRT**: pipeline benchmark mới trên PASCAL VOC 2007 + 2012, dùng QAT checkpoint để export backbone sang TensorRT INT8 hybrid và đo tốc độ trên GPU.

Điểm cần nhớ: **TensorRT không dùng trong quá trình train**. FP32 và QAT vẫn train bằng PyTorch. TensorRT chỉ xuất hiện sau khi đã có checkpoint, ở bước export ONNX, build engine và benchmark/deploy.

## Demo

Ảnh ví dụ SeaDronesSee:

![Ảnh ví dụ SeaDronesSee](./image_sea.png)

GIF demo FP32 vs INT8:

![Demo FP32 vs INT8](./demo_fp32_vs_int8_ver3epoch.gif)

Video demo:

[Tải video demo](./video_fp32_cpu_vs_int8_cpu_ver3epoch.mp4)

## 1. Kiến trúc chung của detector

Detector chính là Faster R-CNN hai giai đoạn:

```text
Ảnh đầu vào
  -> resize + normalize
  -> ConvNeXt-Tiny backbone
  -> FPN
  -> RPN
  -> RoI Align
  -> RoI Heads
  -> boxes, labels, scores
```

Vai trò các khối chính:

| Thành phần | Vai trò |
|---|---|
| ConvNeXt-Tiny | Trích xuất đặc trưng ảnh ở nhiều tầng |
| FPN | Ghép đặc trưng đa tỉ lệ, hữu ích cho vật thể nhỏ |
| RPN | Sinh proposal ứng viên |
| RoI Align | Lấy đặc trưng theo từng proposal |
| RoI Heads | Phân loại object và hồi quy bounding box cuối |

Kiến trúc này được chọn vì Faster R-CNN dễ phân tích, dễ benchmark và phù hợp với bài toán có nhiều vật thể nhỏ hoặc khó phát hiện.

## 2. Nhánh SeaDronesSee benchmark

### 2.1. Mục tiêu

SeaDronesSee là hướng gốc của repo. Mục tiêu là tạo một baseline FP32 đủ mạnh, sau đó train bù lỗi bằng selective QAT để tạo checkpoint INT8 và so sánh chất lượng/tốc độ.

Luồng tổng quát:

```text
SeaDronesSee dataset
  -> train FP32
  -> fp32_best.pt / fp32_last.pt
  -> train selective QAT
  -> qat_best.pt / qat_last.pt
  -> convert selective_int8.pt
  -> evaluate / benchmark / visualize
```

### 2.2. Anchor tự động

Repo hỗ trợ anchor tự động qua cấu hình:

```yaml
model:
  anchor_sizes: auto
```

Khi bật chế độ này, pipeline sẽ đọc annotation train, thống kê kích thước bounding box thật sau resize và sinh anchor scale phù hợp hơn với dữ liệu.

Ví dụ log:

```text
bbox-driven anchors=[38, 83, 164, 304, 545] boxes=47223
```

Anchor tự động quan trọng vì SeaDronesSee có nhiều vật thể nhỏ. Nếu anchor quá lệch so với phân bố bbox thật, RPN dễ bỏ sót object ngay từ bước proposal.

### 2.3. Focal loss

Repo giữ regression loss theo Faster R-CNN chuẩn, nhưng thay classification loss bằng focal loss.

Tổng loss:

```text
L = L_rpn_cls + L_rpn_box + L_roi_cls + L_roi_box
```

| Loss | Cách dùng trong repo |
|---|---|
| RPN classification | Sigmoid Focal Loss |
| RPN box regression | Smooth L1 Loss |
| RoI classification | Softmax Focal Loss |
| RoI box regression | Smooth L1 Loss |

Focal loss giúp giảm ảnh hưởng của background/easy negatives. Điều này hợp lý với dữ liệu drone/hàng hải vì số vùng nền thường lớn hơn rất nhiều so với vùng chứa object.

### 2.4. Selective QAT

Selective QAT hiện là baseline ổn định nhất của repo. Ý tưởng là chỉ lượng tử hóa một số vùng quan trọng thay vì ép toàn bộ Faster R-CNN thành INT8 end-to-end.

Với variant M3, các vùng thường được lượng tử hóa là:

```text
backbone.convnext
backbone.fpn
rpn.shared_conv
rpn.classification
```

Luồng eager selective QAT:

```text
FP32 tensor
  -> QuantStub
  -> INT8 Conv/Linear
  -> DeQuantStub
  -> FP32 tensor
```

Cách này dễ train, dễ debug và phù hợp làm baseline. Nhược điểm là có nhiều lần quantize/dequantize nhỏ lẻ nên tốc độ end-to-end chưa chắc tăng mạnh, nhất là khi RPN decode, RoI Align, RoI heads và NMS vẫn chạy FP32.

### 2.5. Kết quả SeaDronesSee baseline cũ

| Metric | FP32 | Selective INT8 |
|---|---:|---:|
| mAP@50:95 | 0.5606 | 0.3505 |
| mAP@50 | 0.8210 | 0.7310 |
| Mean latency | 6705.29 ms/img | 6323.67 ms/img |
| Full model size | 171.99 MB | 83.89 MB |
| Size reduction | - | 51.23% |

Kết quả này cho thấy selective INT8 giúp giảm kích thước model rõ rệt, nhưng tốc độ CPU end-to-end chỉ cải thiện nhẹ vì detector vẫn còn nhiều phần FP32 và postprocess nặng.

## 3. Nhánh Pascal VOC QAT TensorRT

### 3.1. Dataset Pascal VOC 2007 + 2012

Nhánh TensorRT benchmark dùng dataset Kaggle:

[PASCAL VOC 2007 and 2012](https://www.kaggle.com/datasets/vijayabhaskar96/pascal-voc-2007-and-2012)

Theo mô tả trên Kaggle, đây là bản PASCAL VOC 2007 và 2012 gốc, không chỉnh sửa. Dataset gồm:

- VOC2007 trainval;
- VOC2007 test;
- VOC2012 trainval.

Nguồn tar gốc được Kaggle liệt kê từ Oxford VOC:

- `VOCtrainval_06-Nov-2007.tar`
- `VOCtest_06-Nov-2007.tar`
- `VOCtrainval_11-May-2012.tar`

Theo tài liệu PASCAL VOC 2007 chính thức, bài toán detection gồm 20 lớp object trong cảnh tự nhiên:

```text
person
bird, cat, cow, dog, horse, sheep
aeroplane, bicycle, boat, bus, car, motorbike, train
bottle, chair, dining table, potted plant, sofa, tv/monitor
```

Trong notebook của repo, annotation XML của VOC được chuyển sang COCO JSON để dùng chung dataloader, COCO evaluator và code anchor tự động.

```text
VOC XML
  -> instances_train.json
  -> instances_val.json
  -> Faster R-CNN dataloader
```

Notebook chính:

```text
kaggle/PascalVOC_ConvNeXt_Selective_QAT_Kaggle.ipynb
```

Checkpoint/result dataset:

```text
nguyenducthangtb/echteai-pascal-voc-convnext-qat
```

### 3.2. Train FP32 và QAT trên Pascal VOC

Quá trình train vẫn dùng PyTorch:

```text
Pascal VOC COCO JSON
  -> train FP32
  -> fp32_best.pt
  -> train bù lỗi QAT từng epoch
  -> qat_epoch_02.pt / qat_epoch_03.pt / qat_epoch_04.pt
```

TensorRT không can thiệp vào optimizer, loss hay backward. Vì vậy có thể hiểu đơn giản:

- **train FP32**: học detector nền;
- **train QAT**: fine-tune để model quen với nhiễu lượng tử hóa;
- **TensorRT**: chỉ dùng sau train để chạy inference nhanh hơn trên GPU NVIDIA.

### 3.3. TensorRT hybrid là gì?

Repo không export toàn bộ Faster R-CNN sang TensorRT. Chỉ riêng backbone ConvNeXt được export sang TensorRT engine. Các phần còn lại vẫn dùng PyTorch CUDA.

Kiến trúc benchmark:

```text
Ảnh CUDA
  -> resize + normalize cố định
  -> TensorRT INT8 ConvNeXt backbone
  -> feature maps C2-C5
  -> PyTorch FPN
  -> PyTorch RPN
  -> PyTorch RoI Align + RoI Heads
  -> PyTorch NMS
  -> detections
```

Ở mức tensor:

```text
input image tensor: [1, 3, 640, 1024], CUDA, FP32
  -> TensorRT engine nhận input cố định
  -> backbone tính bằng INT8 bên trong engine
  -> output feature tensors C2-C5 trả lại CUDA
  -> FPN/RPN/RoI chạy PyTorch CUDA
```

Đây là **TensorRT hybrid**, không phải TensorRT detector end-to-end. Lý do chọn hybrid là Faster R-CNN có nhiều bước khó export ổn định sang TensorRT như proposal decode, RoI Align, NMS và postprocess. Backbone là phần nặng và ổn định nhất để đưa sang TensorRT trước.

### 3.4. Luồng export TensorRT

Sau khi có checkpoint FP32 và QAT:

```text
fp32_best.pt
  -> export ConvNeXt backbone FP32 ONNX
  -> build TensorRT FP32 engine

qat_epoch_XX.pt
  -> export ConvNeXt backbone ONNX Q/DQ
  -> build TensorRT INT8 engine
```

ONNX Q/DQ là graph có chèn Quantize/DeQuantize node để TensorRT hiểu vùng nào cần chạy INT8. TensorRT sẽ build engine tối ưu cho GPU hiện tại, vì vậy engine thường phụ thuộc môi trường CUDA/TensorRT/GPU.

### 3.5. Cấu hình benchmark Pascal VOC

| Mục | Giá trị |
|---|---:|
| Detector | Faster R-CNN + ConvNeXt-Tiny + FPN |
| Dataset | Pascal VOC 2007 + 2012 |
| Số lớp | 20 foreground + background |
| Train image sizes | 512, 608, 640 |
| Max image size | 1024 |
| TensorRT input | 1 x 3 x 640 x 1024 |
| Benchmark images | 1000 |
| Benchmark batch size | 1 |
| FP32 backend | PyTorch CUDA |
| INT8 backend | TensorRT backbone + PyTorch CUDA heads |

### 3.6. Benchmark Pascal VOC 1000 ảnh

Baseline FP32:

```text
checkpoint: fp32_best.pt
backend: PyTorch CUDA
```

INT8 TensorRT hybrid:

```text
checkpoint: qat_epoch_02.pt / qat_epoch_03.pt / qat_epoch_04.pt
backend: TensorRT INT8 backbone + PyTorch CUDA detector heads
```

Kết quả:

| Metric | FP32 PyTorch GPU | INT8 TRT epoch 2 | INT8 TRT epoch 3 | INT8 TRT epoch 4 |
|---|---:|---:|---:|---:|
| mAP@50:95 | 0.4924 | 0.5336 | 0.5285 | 0.5284 |
| mAP@50 | 0.8045 | 0.8140 | 0.8040 | 0.8080 |
| AP small | 0.1585 | 0.2300 | 0.2350 | 0.2280 |
| AP medium | 0.4025 | 0.4320 | 0.4250 | 0.4190 |
| AP large | 0.5720 | 0.6170 | 0.6080 | 0.6120 |
| Latency | 135.54 ms/img | 82.53 ms/img | 84.93 ms/img | 83.37 ms/img |
| FPS | 7.38 | 12.12 | 11.77 | 12.00 |
| Speedup | 1.0000x | 1.6423x | 1.5960x | 1.6257x |

Bảng theo run:

| Run ID | Backend | QAT epoch | Images | mAP@50:95 | mAP@50 | Latency | FPS | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VOC-FP32-GPU | FP32 PyTorch CUDA | - | 1000 | 0.4924 | 0.8045 | 135.54 ms/img | 7.38 | 1.0000x |
| VOC-TRT-INT8-E2 | INT8 TensorRT hybrid CUDA | 2 | 1000 | 0.5336 | 0.8140 | 82.53 ms/img | 12.12 | 1.6423x |
| VOC-TRT-INT8-E3 | INT8 TensorRT hybrid CUDA | 3 | 1000 | 0.5285 | 0.8040 | 84.93 ms/img | 11.77 | 1.5960x |
| VOC-TRT-INT8-E4 | INT8 TensorRT hybrid CUDA | 4 | 1000 | 0.5284 | 0.8080 | 83.37 ms/img | 12.00 | 1.6257x |

Nhận xét:

- TensorRT INT8 hybrid nhanh hơn FP32 PyTorch GPU khoảng `1.60x - 1.64x`.
- Epoch 2 đang là run tốt nhất về tốc độ và mAP trong bảng này.
- mAP INT8 cao hơn FP32 trong benchmark này không có nghĩa INT8 luôn chính xác hơn FP32. Checkpoint QAT đã được fine-tune thêm sau FP32 và benchmark TensorRT dùng input cố định `640 x 1024`, nên đây là so sánh triển khai thực nghiệm chứ không phải ablation thuần precision.

## 4. Lệnh chạy chính

Train FP32:

```bash
python scripts/train_fp32.py --config configs/seadronessee_colab.yaml
```

Train selective QAT một GPU:

```bash
python scripts/train_qat.py \
  --config configs/seadronessee_colab.yaml \
  --variant M3
```

Train selective QAT hai GPU:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_qat_ddp.py \
  --config configs/seadronessee_colab.yaml \
  --variant M3
```

Export backbone QAT sang ONNX Q/DQ cho TensorRT:

```bash
python scripts/export_convnext_tensorrt_onnx.py \
  --config configs/seadronessee_colab.yaml \
  --model qat_graph \
  --qat-checkpoint /path/to/qat_epoch_02.pt \
  --output /path/to/convnext_qat_epoch02_int8_qdq.onnx \
  --height 640 \
  --width 1024 \
  --batch-size 1 \
  --tensorrt-friendly-int8
```

Build TensorRT INT8 engine:

```bash
python scripts/build_tensorrt_engine.py \
  --onnx /path/to/convnext_qat_epoch02_int8_qdq.onnx \
  --engine /path/to/convnext_qat_epoch02_int8.engine \
  --precision int8 \
  --workspace-mb 4096
```

Benchmark hai engine TensorRT hybrid bằng script:

```bash
python scripts/benchmark_convnext_tensorrt_hybrid.py \
  --config configs/seadronessee_colab.yaml \
  --fp32-engine /path/to/convnext_backbone_fp32.engine \
  --int8-engine /path/to/convnext_qat_epoch02_int8.engine \
  --fp32-checkpoint /path/to/fp32_best.pt \
  --qat-checkpoint /path/to/qat_epoch_02.pt \
  --images 1000 \
  --height 640 \
  --width 1024
```

Trong notebook Pascal VOC, phần benchmark chính của báo cáo là:

```text
FP32 PyTorch CUDA từ fp32_best.pt
  vs
INT8 TensorRT hybrid CUDA từ qat_epoch_XX.pt
```

Cell này đo cả mAP và thời gian inference trên cùng 1000 ảnh để tạo bảng kết quả ở mục 3.6.

## 5. Ưu điểm và hạn chế

### Ưu điểm

- Faster R-CNN + ConvNeXt-Tiny + FPN dễ đọc và dễ phân tích.
- Anchor tự động giúp khớp tốt hơn với phân bố bbox thật.
- Focal loss phù hợp với bài toán mất cân bằng foreground/background.
- Selective QAT dễ debug và làm baseline ổn định.
- TensorRT hybrid cho phép chạy INT8 trên GPU NVIDIA mà không cần export toàn bộ detector.
- Benchmark Pascal VOC cho thấy speedup khoảng `1.60x - 1.64x`.

### Hạn chế

- TensorRT hybrid chưa phải TensorRT end-to-end.
- FPN, RPN, RoI heads và NMS vẫn chạy PyTorch CUDA.
- TensorRT engine phụ thuộc GPU, CUDA và version TensorRT.
- So sánh FP32 PyTorch GPU với INT8 TensorRT hybrid là so sánh deploy thực nghiệm, không phải phép đo cô lập chỉ riêng precision.

## 6. Môi trường

Để train FP32/QAT, cần PyTorch, TorchVision và các package COCO evaluation.

Để build và chạy TensorRT engine, runtime phải import được TensorRT:

```python
import tensorrt
```

Nếu môi trường không có TensorRT, vẫn có thể train FP32/QAT và export ONNX, nhưng không build được `.engine` và không benchmark được TensorRT hybrid.

## Kết luận

Repo hiện có hai nhánh rõ ràng. SeaDronesSee là pipeline gốc để xây dựng baseline FP32 và selective QAT. Pascal VOC QAT TensorRT là pipeline benchmark/deploy mới, trong đó ConvNeXt backbone được chạy bằng TensorRT INT8 trên GPU, còn FPN/RPN/RoI/NMS giữ PyTorch CUDA để đảm bảo ổn định. Với Pascal VOC 2007 + 2012 trên 1000 ảnh, TensorRT hybrid đạt khoảng `1.60x - 1.64x` speedup so với FP32 PyTorch GPU trong cấu hình benchmark hiện tại.
