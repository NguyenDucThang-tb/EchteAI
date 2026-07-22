# EchteAI

EchteAI là baseline phát hiện đối tượng trên SeaDronesSee được xây dựng quanh Faster R-CNN với backbone ConvNeXt-Tiny và FPN. Mục tiêu của repository này không phải là tạo ra một detector nhẹ tuyệt đối, mà là xây dựng một pipeline đủ mạnh ở FP32, sau đó áp dụng lượng tử hóa chọn lọc để khảo sát tradeoff giữa chất lượng và độ trễ khi triển khai INT8. Ở trạng thái hiện tại, nhánh ổn định nhất của repo là selective eager QAT; nhánh PT2E graph đã có trong code nhưng nhạy cảm hơn với môi trường.

## Minh họa nhanh

Ảnh ví dụ từ dataset SeaDronesSee:

![Ảnh ví dụ SeaDronesSee](./image_sea.png)

GIF demo ngắn so sánh FP32 và INT8:

![Demo FP32 vs INT8](./demo_fp32_vs_int8_ver3epoch.gif)

Video đầy đủ:

[Tải video demo (.mp4)](./video_fp32_cpu_vs_int8_cpu_ver3epoch.mp4)

## Tổng quan nhanh

Ý chính của repo có thể tóm tắt ngắn gọn như sau:

- Detector gốc là Faster R-CNN.
- Backbone là ConvNeXt-Tiny, neck là FPN.
- Baseline mạnh được huấn luyện ở FP32 trước.
- Sau đó mô hình được fine-tune bằng selective eager QAT.
- Cuối cùng checkpoint QAT được convert sang INT8 để benchmark và inference trên CPU.

Nếu nhìn theo luồng xử lý mức cao, pipeline detector hiện tại là:

`image -> transform -> ConvNeXt backbone -> FPN -> RPN -> proposals -> RoI Align -> RoI heads -> final detections`

Kiến trúc chính của hệ thống được triển khai trong [fasterrcnn_convnext.py](D:/Quanti_FasterRCNN/EchteAI/pipelines/convnext_qat/models/fasterrcnn_convnext.py). Ảnh đầu vào trước hết đi qua khối transform của Faster R-CNN để resize và chuẩn hóa, sau đó được đưa qua ConvNeXt-Tiny để sinh ra đặc trưng đa mức `C2-C5`. Những đặc trưng này đi vào FPN để tạo `P2-P6`, rồi chuyển sang RPN để sinh proposal. Proposal tiếp tục đi qua RoI Align và RoI heads để cho ra dự đoán cuối cùng. Kiến trúc này được chọn vì SeaDronesSee có nhiều vật thể nhỏ, trong khi FPN giúp duy trì đặc trưng ở các mức phân giải cao và Faster R-CNN thường cho baseline ổn định hơn so với việc bắt đầu ngay bằng một detector quá nhẹ.

## Kiến trúc detector hiện tại

Detector hiện tại có ba cụm thành phần chính. Cụm đầu tiên là backbone ConvNeXt-Tiny chịu trách nhiệm trích xuất đặc trưng. Cụm thứ hai là FPN và RPN, nơi đặc trưng được tái tổ chức theo tháp đặc trưng và dùng để sinh proposal. Cụm thứ ba là RoI stage, nơi proposal được cắt đặc trưng bằng RoI Align rồi đưa vào head phân loại và hồi quy hộp giới hạn. Mặc dù đây là một detector hai giai đoạn tương đối nặng, nó có lợi thế rõ ràng trong bài toán nghiên cứu vì dễ đọc, dễ phân tích và các thành phần như anchor, proposal hay loss đều có thể can thiệp một cách có kiểm soát.

Một điểm quan trọng trong kiến trúc này là anchor không bị cố định cứng theo giá trị mặc định của torchvision. Repo hỗ trợ `model.anchor_sizes: auto`, tức là quét annotation huấn luyện, lấy thống kê kích thước đối tượng sau khi resize, rồi tự suy ra năm anchor scales tương ứng với các mức `P2-P6`. Với bài toán vật thể nhỏ, việc để anchor lệch xa phân bố thật của dữ liệu thường làm recall của RPN yếu ngay từ đầu, vì vậy khâu fit anchor là một phần rất quan trọng của baseline.

## Thiết kế hàm loss

Thiết kế loss là phần khác biệt quan trọng của repo này so với Faster R-CNN mặc định. Ý tưởng chung là giữ phần hồi quy hộp giới hạn theo hướng tiêu chuẩn để đảm bảo ổn định, nhưng thay loss phân loại bằng focal loss để giảm tác động của các mẫu quá dễ, đặc biệt là nền âm tính áp đảo. Với bài toán drone hoặc hàng hải, số lượng background dễ thường cực lớn so với số foreground thật, nên focal loss giúp quá trình học tập trung hơn vào những proposal hoặc anchor khó.

### Loss ở RPN

Trong nhánh RPN, phần objectness dùng sigmoid focal loss. Nếu ký hiệu `x` là logit dự đoán objectness, `y ∈ {0,1}` là nhãn thật, `p = sigmoid(x)`, còn `p_t = p` khi `y = 1` và `p_t = 1 - p` khi `y = 0`, thì focal loss được dùng trong code có dạng:

```text
FL(x, y) = alpha_t * (1 - p_t)^gamma * BCEWithLogits(x, y)
```

Ở đây `gamma = 2.0` theo mặc định, còn `alpha` hiện tại để `None`, tức là không thêm foreground/background reweighting ngoài thành phần focal modulation. Khi `alpha = None`, thành phần `alpha_t` được xem như bằng `1`, vì vậy loss chủ yếu tác động bằng cách giảm trọng số của các mẫu có `p_t` đã cao, tức là các mẫu đã được phân loại rất dễ.

Song song với objectness, phần hồi quy hộp giới hạn của RPN vẫn dùng Smooth L1 loss với `beta = 1/9`. Nếu `d` là độ lệch giữa dự đoán và regression target, Smooth L1 có thể viết gọn như sau:

```text
SmoothL1(d) = 0.5 * d^2 / beta      nếu |d| < beta
              |d| - 0.5 * beta      nếu |d| >= beta
```

Loss hồi quy này chỉ áp dụng trên positive anchors, vì negative anchors không có target bounding box có ý nghĩa. Về mặt ý nghĩa, thiết kế ở RPN đang giải quyết đồng thời hai việc khác nhau: phân biệt foreground và background trong một không gian cực mất cân bằng bằng focal loss, đồng thời tinh chỉnh proposal bằng một loss ổn định hơn là Smooth L1.

### Loss ở RoI head

Ở RoI stage, repo thay cross-entropy mặc định bằng softmax focal loss. Nếu `CE(z, y)` là cross-entropy giữa vector class logits `z` và nhãn thật `y`, còn `p_t = exp(-CE(z, y))`, thì loss phân loại có dạng:

```text
FL_softmax(z, y) = (1 - p_t)^gamma * CE(z, y)
```

Nếu `alpha` được bật thì code còn nhân thêm một hệ số theo foreground/background, nhưng trong cấu hình hiện tại `alpha` đang để `None`, nên loss tập trung vào thành phần điều chế theo `gamma`. Điều này có nghĩa là ngay cả sau khi đã qua RPN, RoI stage vẫn không học như nhau trên mọi mẫu, mà ưu tiên nhiều hơn cho các mẫu khó và các trường hợp dễ nhầm lẫn.

Tương tự RPN, phần hồi quy bounding box ở RoI head vẫn dùng Smooth L1 với `beta = 1/9`. Trong code, regression loss được tính trên các RoI dương và sau đó chuẩn hóa theo số lượng mẫu hợp lệ. Cách thiết kế này giúp phần classification trở nên nhạy hơn với hard example, còn phần localization vẫn giữ tính ổn định và ít dao động hơn trong huấn luyện.

### Cách áp dụng loss trong pipeline

Repo không thay toàn bộ kiến trúc Faster R-CNN, mà cài focal loss bằng cách thay lớp RPN và lớp RoI heads mặc định bằng `FocalRegionProposalNetwork` và `FocalRoIHeads`. Điều đó có nghĩa là backbone, FPN, proposal flow, RoI Align và toàn bộ control flow của detector vẫn được giữ nguyên; chỉ đúng hai điểm phát sinh loss classification được thay cách tính. Đây là một thiết kế thực dụng: thay đổi đủ nhiều để phù hợp hơn với bài toán vật thể nhỏ, nhưng không phá cấu trúc ổn định của detector gốc.

## Pipeline huấn luyện và lượng tử hóa

Pipeline hoạt động của repo có thể chia thành ba giai đoạn.

- Giai đoạn thứ nhất là huấn luyện baseline FP32 bằng [train_fp32.py](D:/Quanti_FasterRCNN/EchteAI/scripts/train_fp32.py). Ở bước này mô hình được train hoàn toàn ở FP32, sau mỗi epoch sẽ evaluate trên validation set và lưu `fp32_last.pt` cùng `fp32_best.pt`.
- Giai đoạn thứ hai là selective QAT bằng [train_qat.py](D:/Quanti_FasterRCNN/EchteAI/scripts/train_qat.py) hoặc [train_qat_ddp.py](D:/Quanti_FasterRCNN/EchteAI/scripts/train_qat_ddp.py) nếu chạy nhiều GPU. Bước này load checkpoint FP32 tốt nhất, chèn fake-quant vào các vùng được chọn, thực hiện observer warmup rồi đi qua các phase `weight_only`, `full` và `frozen`.
- Giai đoạn cuối là convert checkpoint QAT tốt nhất ở phase frozen sang checkpoint INT8 thật `selective_int8.pt`, sau đó dùng [evaluate.py](D:/Quanti_FasterRCNN/EchteAI/scripts/evaluate.py), [compare_fp32_int8.py](D:/Quanti_FasterRCNN/EchteAI/scripts/compare_fp32_int8.py), [visualize_fp32_int8.py](D:/Quanti_FasterRCNN/EchteAI/scripts/visualize_fp32_int8.py) hoặc [infer_video_fp32_int8.py](D:/Quanti_FasterRCNN/EchteAI/scripts/infer_video_fp32_int8.py) để phân tích.

Điểm cần nhấn mạnh là nhánh selective QAT trong repo hiện nay mang mục tiêu rất rõ: tạo ra một baseline INT8 đủ dễ chạy, chứ không phải một hệ thống graph quantization tối ưu end-to-end ngay từ đầu. Đây là lý do phần convert cuối cùng tập trung vào checkpoint INT8 phục vụ benchmark và inference trên CPU.

## Kiến trúc eager island

Selective QAT hiện tại được triển khai theo kiểu eager island trong [selective_qat.py](D:/Quanti_FasterRCNN/EchteAI/pipelines/convnext_qat/quantization/selective_qat.py). Thay vì biến cả detector thành một graph INT8 liền mạch, code bọc từng `Conv2d` hoặc `Linear` trong vùng được chọn bằng một lớp `QuantizedOperation`, bên trong gồm `QuantStub`, phép toán gốc rồi `DeQuantStub`.

Nếu nhìn một island riêng lẻ, luồng dữ liệu có dạng:

```text
Tensor FP32
   ->
QuantStub
   ->
Tensor đã lượng tử hóa
   ->
Conv/Linear lượng tử hóa
   ->
DeQuantStub
   ->
Tensor FP32
```

Nếu nhìn ở mức toàn detector, pipeline eager hiện tại có thể hình dung như sau:

```text
Ảnh đầu vào FP32
   ->
Transform / Resize / Normalize (FP32)
   ->
Backbone ConvNeXt
   -> [nhiều INT8 island nhỏ nằm trong các Conv]
FPN
   -> [nhiều INT8 island nhỏ nằm trong các Conv]
RPN
   -> [một phần conv / cls được lượng tử hóa]
Proposal decode + NMS (FP32)
   ->
RoI Align (FP32)
   ->
RoI heads (FP32)
   ->
Final decode + NMS (FP32)
   ->
Detections
```

Cách hoạt động thực sự của eager island là như sau. Ở giai đoạn QAT, mỗi island được gắn fake-quant observer để mô phỏng tác động của lượng tử hóa lên activation và weight. Khi convert, từng island được thay bằng quantized operator thật trên CPU. Tuy nhiên dữ liệu không nằm mãi trong không gian INT8. Nó liên tục đi từ FP32 sang INT8 rồi quay lại FP32 ở ranh giới của từng island. Chính ranh giới này làm eager dễ triển khai hơn graph quantization, nhưng cũng là gốc của nhược điểm về hiệu năng.

Điểm thường gây thắc mắc là tại sao không lượng tử hóa một lần ở đầu block rồi để tensor đi qua toàn bộ chuỗi phép toán ở dạng INT8. Lý do là eager selective quantization trong repo này chỉ bọc trực tiếp các phép toán được hỗ trợ tốt, chủ yếu là `Conv2d` và `Linear`. Trong một block thực tế của ConvNeXt hoặc FPN, giữa các phép tích chập còn có những phép như `LayerNorm`, `GELU`, cộng tắt residual, reshape hoặc tái tổ chức đặc trưng. Những phép này không phải lúc nào cũng được giữ liền mạch trong cùng miền INT8 ở nhánh eager hiện tại. Vì vậy cách triển khai an toàn là chỉ quantize ngay trước phép toán hỗ trợ, thực thi phép toán đó ở INT8, rồi dequantize trở lại FP32 để phần còn lại của block tiếp tục chạy bình thường. Nói cách khác, eager baseline hiện tại hoạt động theo logic “quantize ngay trước op hỗ trợ” chứ không phải “giữ cả block trong INT8 từ đầu đến cuối”.

Nếu nhìn ở mức một block ConvNeXt hoặc một đoạn conv trong FPN, có thể hình dung rõ hơn như sau:

```text
Tensor FP32
   ->
LayerNorm / Permute / GELU / Residual add (FP32)
   ->
QuantStub
   ->
Conv hoặc Linear được hỗ trợ INT8
   ->
DeQuantStub
   ->
Các phép còn lại của block quay về FP32
   ->
QuantStub
   ->
Conv tiếp theo được hỗ trợ INT8
   ->
DeQuantStub
   ->
Output FP32
```

Sơ đồ này cho thấy một điều rất quan trọng: không phải “mỗi tensor chỉ chứa vài frame” hay chỉ có một island cho cả block. Thực tế là trong cùng một block có thể có nhiều phép toán khác loại nối tiếp nhau. Chỉ những op nào khớp điều kiện hỗ trợ của backend mới được bọc thành island INT8. Các op còn lại vẫn chạy ở FP32, nên dữ liệu phải liên tục qua lại giữa hai miền số học. Đây là lý do số island thường nhiều hơn trực giác ban đầu.

Nhược điểm lớn nhất của eager island là overhead quantize/dequantize lặp lại rất nhiều lần. Với một detector hai giai đoạn như Faster R-CNN, backbone chỉ là một phần của tổng latency, còn các phần như proposal decode, NMS, RoI Align, RoI heads và final postprocess vẫn ở FP32. Vì vậy dù trọng số của một số phần đã được nén xuống INT8, độ trễ end-to-end chưa chắc giảm tương xứng. Một nhược điểm khác là score distribution sau lượng tử hóa có thể bị lệch, kéo theo recall và AP giảm khá mạnh, đặc biệt trong bài toán vật thể nhỏ nơi chỉ một sai lệch nhỏ về activation range cũng có thể làm confidence tụt đáng kể.

Nói ngắn gọn, eager island là một baseline nghiên cứu tốt vì dễ kiểm soát và dễ tách ablation. Nhưng nếu mục tiêu là tối ưu latency thật sự, kiến trúc này thường chạm trần sớm hơn graph quantization vì không giữ được một đoạn INT8 liền mạch đủ dài.

## Kiến trúc graph dự kiến áp dụng tiếp theo

Hướng tiếp theo của repo là chuyển phần lượng tử hóa từ eager island sang graph quantization, cụ thể là nhánh PT2E trong [train_pt2e_qat.py](D:/Quanti_FasterRCNN/EchteAI/scripts/train_pt2e_qat.py) và [pt2e_qat.py](D:/Quanti_FasterRCNN/EchteAI/pipelines/convnext_qat/quantization/pt2e_qat.py). Mục tiêu không phải đổi detector sang mô hình khác, mà vẫn giữ Faster R-CNN + ConvNeXt-Tiny + FPN, chỉ thay cách chèn quantization để backend nhìn thấy một graph lớn hơn và có cơ hội fuse hoặc giữ miền INT8 dài hơn.

Nếu eager hoạt động theo kiểu “đụng Conv nào thì bọc Conv đó”, thì graph quantization hoạt động theo kiểu “xuất graph tính toán, đánh dấu vùng có thể lượng tử hóa, rồi để hệ thống chèn fake-quant và convert theo cấu trúc graph”. Về mặt trực quan, pipeline mong muốn sẽ gần với dạng sau:

```text
Ảnh đầu vào FP32
   ->
Transform / Resize / Normalize (FP32)
   ->
[Một đoạn graph backbone hoặc backbone+FPN được giữ liền mạch lâu hơn]
   ->
Quantize một lần ở đầu vùng
   ->
Nhiều op liên tiếp chạy trong cùng miền INT8
   ->
Dequantize ở cuối vùng
   ->
RPN / Proposal / RoI / Postprocess
   ->
Detections
```

Nếu nhìn ở mức block, mục tiêu của graph branch là biến:

```text
FP32 -> Quant -> Conv -> DeQuant -> FP32 -> Quant -> Conv -> DeQuant -> FP32
```

thành gần hơn với:

```text
FP32 -> Quant -> Conv -> Act/Norm/Residual compatible path -> Conv -> ... -> DeQuant -> FP32
```

Lợi ích kỳ vọng của cách này là giảm số lần quantize/dequantize trung gian, giảm memory traffic, và tăng cơ hội để backend tối ưu thực thi tốt hơn. Nếu làm đúng, graph branch thường có triển vọng latency tốt hơn eager branch, nhất là khi phần backbone chiếm tỷ trọng lớn trong compute. Đổi lại, nhánh này khó vận hành hơn vì phụ thuộc rất mạnh vào phiên bản PyTorch, TorchAO, backend quantization, khả năng export graph, và tập op thực sự được nhận diện trong runtime đang dùng. Đây cũng là lý do repo hiện tại vẫn xem eager selective QAT là baseline ổn định, còn PT2E graph là hướng nâng cấp tiếp theo để đẩy hiệu năng.

## Dataset, anchor và dữ liệu đầu vào

Pipeline kỳ vọng dataset ở định dạng COCO, với cấu trúc tối thiểu như sau:

```text
dataset_root/
  images/
    train/
    val/
  annotations/
    instances_train.json
    instances_val.json
```

Điều này là điều kiện cần vì toàn bộ dataloader, metric evaluator và phần tự động suy anchor đều dựa trên cách tổ chức này. Một khi annotation đúng COCO, repo có thể kiểm tra số lớp foreground, ánh xạ category id sang label nội bộ và thống kê kích thước đối tượng để tính anchor.

Khâu anchor đặc biệt quan trọng đối với SeaDronesSee. Khi `anchor_sizes` để `auto`, code trong [anchors.py](D:/Quanti_FasterRCNN/EchteAI/pipelines/convnext_qat/anchors.py) quét bounding box của tập huấn luyện, quy đổi về không gian kích thước sau resize rồi tạo ra năm anchor scales tương ứng với các mức của FPN. Việc làm này giúp RPN nhìn thấy anchor gần với vật thể thật hơn, từ đó proposal recall tốt hơn. Nếu thay anchor sau khi đã train, nên coi đó là một cấu hình mới và train lại baseline FP32 thay vì resume checkpoint cũ.

## Môi trường triển khai

Repo này có thể chạy trên local workstation, Colab hoặc Kaggle. Tuy nhiên về mặt thực dụng, selective eager QAT hiện là nhánh ổn định nhất. Quy ước triển khai của nhánh này là huấn luyện FP32 và QAT trên GPU CUDA, sau đó convert INT8 và benchmark trên CPU. Đây cũng là lý do các script benchmark chính đều xoay quanh so sánh FP32 và INT8 ở môi trường CPU để công bằng hơn.

Nhánh DDP trong [train_qat_ddp.py](D:/Quanti_FasterRCNN/EchteAI/scripts/train_qat_ddp.py) hiểu `training.qat_batch_size` là batch size trên mỗi GPU chứ không phải global batch size. Vì vậy nếu chạy trên hai GPU và đặt `qat_batch_size = 1` thì global batch thật là `2`. Điều này cần được ghi nhớ khi đối chiếu kết quả giữa single-GPU và multi-GPU.

Repo cũng có một nhánh PT2E graph trong [train_pt2e_qat.py](D:/Quanti_FasterRCNN/EchteAI/scripts/train_pt2e_qat.py) và [pt2e_qat.py](D:/Quanti_FasterRCNN/EchteAI/pipelines/convnext_qat/quantization/pt2e_qat.py). Mục tiêu của nhánh này là giảm overhead eager island bằng graph quantization cho phần backbone hoặc backbone+FPN. Tuy nhiên trong thực tế nó đòi hỏi môi trường phần mềm sạch và ổn định hơn, nên hiện tại eager selective QAT vẫn là baseline dễ vận hành nhất trong các notebook runtime bị giới hạn.

## TensorRT hybrid cho ConvNeXt

Repo hiện có thêm đường deploy TensorRT cho nhánh ConvNeXt selective QAT. Đường này không thay đổi quá trình train hiện tại. Sau khi đã có `fp32_best.pt` và `qat_best.pt` hoặc `qat_last.pt`, ta export riêng vùng backbone/backbone+FPN sang ONNX, build TensorRT engine, rồi benchmark theo dạng hybrid:

```text
image CUDA
  -> TensorRT ConvNeXt backbone
  -> PyTorch FPN / RPN / RoI heads / NMS
  -> detections
```

Mặc định config dùng `quantization.compiler.scope: backbone`, nghĩa là chỉ backbone body chạy TensorRT, còn FPN và detector heads vẫn chạy PyTorch để giữ pipeline ổn định. Đây là cách thực dụng để INT8 chạy được trên GPU NVIDIA mà không cần export toàn bộ Faster R-CNN, vì RPN decode, RoIAlign và NMS khó đưa vào TensorRT end-to-end.

## Ví dụ lệnh chạy

Huấn luyện baseline FP32:

```bash
python scripts/train_fp32.py --config configs/seadronessee_colab.yaml
```

Huấn luyện selective eager QAT trên một GPU:

```bash
python scripts/train_qat.py --config configs/seadronessee_colab.yaml --variant M3
```

Huấn luyện selective eager QAT trên hai GPU bằng DDP:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_qat_ddp.py \
  --config configs/seadronessee_colab.yaml \
  --variant M3
```

Evaluate checkpoint INT8:

```bash
python scripts/evaluate.py \
  --config configs/seadronessee_colab.yaml \
  --model int8 \
  --checkpoint /path/to/selective_int8.pt \
  --split val
```

So sánh FP32 và INT8 trên CPU:

```bash
python scripts/compare_fp32_int8.py \
  --config configs/seadronessee_colab.yaml \
  --fp32-checkpoint /path/to/fp32_best.pt \
  --int8-checkpoint /path/to/selective_int8.pt \
  --images 100 \
  --threads 1
```

Export backbone FP32 sang ONNX:

```bash
python scripts/export_convnext_tensorrt_onnx.py \
  --config configs/seadronessee_colab.yaml \
  --model fp32 \
  --fp32-checkpoint /path/to/fp32_best.pt \
  --output /path/to/convnext_backbone_fp32.onnx \
  --height 960 \
  --width 1600
```

Export backbone selective-QAT sang ONNX Q/DQ cho TensorRT INT8:

```bash
python scripts/export_convnext_tensorrt_onnx.py \
  --config configs/seadronessee_colab.yaml \
  --model qat_graph \
  --qat-checkpoint /path/to/qat_best.pt \
  --output /path/to/convnext_backbone_qat.onnx \
  --height 960 \
  --width 1600 \
  --tensorrt-friendly-int8
```

Build TensorRT engine:

```bash
python scripts/build_tensorrt_engine.py \
  --onnx /path/to/convnext_backbone_fp32.onnx \
  --engine /path/to/convnext_backbone_fp32.engine \
  --precision fp32

python scripts/build_tensorrt_engine.py \
  --onnx /path/to/convnext_backbone_qat.onnx \
  --engine /path/to/convnext_backbone_int8.engine \
  --precision int8
```

Benchmark FP32 TensorRT hybrid và INT8 TensorRT hybrid trên 100 ảnh:

```bash
python scripts/benchmark_convnext_tensorrt_hybrid.py \
  --config configs/seadronessee_colab.yaml \
  --fp32-engine /path/to/convnext_backbone_fp32.engine \
  --int8-engine /path/to/convnext_backbone_int8.engine \
  --fp32-checkpoint /path/to/fp32_best.pt \
  --qat-checkpoint /path/to/qat_best.pt \
  --images 100 \
  --height 960 \
  --width 1600
```

## Bảng kết quả để tự điền

### Baseline FP32

| Run ID | Backbone | Ảnh | Epoch | mAP@50:95 | mAP@50 | Ghi chú |
|---|---|---:|---:|---:|---:|---|
| FP32-CPU-01 | ConvNeXt-Tiny + FPN | 960 / 1600 | best | 0.5606 | 0.8210 | Acc=0.7351, Prec=0.8027, IoU=0.8146, 6705.2876 ms/img |

### Checkpoint INT8 sau convert

| Run ID | Nguồn QAT | Backend | mAP@50:95 | mAP@50 | Latency ms/img | Model MB | Ghi chú |
|---|---|---|---:|---:|---:|---:|---|
| INT8-CPU-OLD-E3 | `qat_last` epoch 3 old | eager selective | 0.2141 |  | 6326.2100 | 83.8874 | Acc=0.4459, Prec=0.7788, IoU=0.6961 |
| INT8-CPU-E3 | `qat_last` epoch 3 | eager selective | 0.3505 | 0.7310 | 6323.6697 | 83.8874 | Acc=0.5432, Prec=0.8375, IoU=0.7539 |
| INT8-CPU-E4 | `qat_last` epoch 4 | eager selective | 0.3310 | 0.6900 | 6708.8203 | 83.8874 | Acc=0.4718, Prec=0.8073, IoU=0.7589 |

### Tóm tắt FP32 vs INT8

| Metric | FP32 | INT8 | Delta |
|---|---:|---:|---:|
| mAP@50:95 | 0.5606 | 0.3505 | -0.2101 |
| mAP@50 | 0.8210 | 0.7310 | -0.0900 |
| Mean latency ms/img | 6705.2876 | 6323.6697 | -381.6179 |
| Kích thước backbone MB | 116.5970 | 30.1662 | -86.4308 |
| Giảm kích thước backbone |  | 74.1278% |  |
| Kích thước full model MB | 171.9961 | 83.8874 | -88.1087 |
| Giảm kích thước full model |  | 51.2271% |  |

## Kết luận

EchteAI hiện tại là một baseline selective eager-QAT thực dụng cho SeaDronesSee. Phần detector FP32 đủ mạnh để làm mốc tham chiếu, phần loss được điều chỉnh theo hướng phù hợp hơn với dữ liệu mất cân bằng và vật thể nhỏ, còn phần quantization cho phép tạo checkpoint INT8 chạy được trên CPU mà không phải viết lại toàn bộ detector theo graph. Đổi lại, đây vẫn là một kiến trúc hai giai đoạn nặng, nên lợi ích latency end-to-end bị giới hạn và chất lượng sau INT8 có thể tụt đáng kể nếu calibration hoặc QAT chưa đủ tốt. Chính vì vậy, baseline này phù hợp nhất như một điểm xuất phát có kiểm soát để so sánh, phân tích và tiếp tục cải thiện.
