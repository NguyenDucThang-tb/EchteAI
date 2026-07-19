"""QAT chọn lọc kiểu Eager cho Faster R-CNN dùng ConvNeXt-FPN.

Mỗi ``Conv2d``/``Linear`` được chọn là một "đảo INT8" độc lập sau khi convert:
FP32 -> Quantize -> toán tử INT8 -> DeQuantize -> FP32. Vì vậy LayerNorm, GELU,
phép cộng residual, giải mã proposal, ROI Align và ROI heads vẫn chạy bằng FP32.
Đây là pipeline baseline dễ kiểm soát nhưng có thể tốn nhiều chi phí Q/DQ.
"""

import copy

import torch
from torch import nn
from torch.ao.quantization import (
    DeQuantStub,
    FakeQuantize,
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
    QConfig,
    QuantStub,
    convert,
    prepare_qat,
)


VARIANT_REGIONS = {
    "M0": set(),
    "M1": {"backbone"},
    "M2": {"backbone", "fpn"},
    "M3": {"backbone", "fpn", "rpn_conv", "rpn_cls"},
    "M4": {"backbone", "fpn", "rpn_conv", "rpn_cls", "rpn_bbox"},
}


class QuantizedOperation(nn.Module):
    """Bọc một toán tử thành vùng Quantize -> operation -> DeQuantize."""

    def __init__(self, operation):
        super().__init__()
        self.quant = QuantStub()
        self.operation = operation
        self.dequant = DeQuantStub()

    def forward(self, x):
        """Mô phỏng hoặc thực thi Q/DQ bao quanh toán tử."""
        return self.dequant(self.operation(self.quant(x)))


def selective_qconfig():
    """Tạo cấu hình W8A8: activation per-tensor và weight per-channel."""
    return QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            quant_min=0,
            quant_max=255,
        ),
        weight=FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            quant_min=-128,
            quant_max=127,
            ch_axis=0,
        ),
    )


def _wrap_operations(module, qconfig):
    """Duyệt đệ quy và bọc mọi Conv/Linear thuộc vùng đã chọn."""
    for name, child in list(module.named_children()):
        if isinstance(child, QuantizedOperation):
            continue
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            wrapped = QuantizedOperation(child)
            wrapped.qconfig = qconfig
            setattr(module, name, wrapped)
        else:
            _wrap_operations(child, qconfig)


def _validate_variant(variant):
    """Chuẩn hóa và kiểm tra tên cấu hình M0-M4."""
    variant = str(variant).upper()
    if variant not in VARIANT_REGIONS:
        raise ValueError(f"Unknown QAT variant {variant!r}; choose {sorted(VARIANT_REGIONS)}")
    return variant


def _regions_from_module_names(module_names):
    """Đổi tên module trong YAML thành tên vùng nội bộ của pipeline."""
    regions = set()
    for raw_name in module_names:
        name = str(raw_name).lower()
        if name == "backbone.convnext":
            regions.add("backbone")
        elif name == "backbone.fpn":
            regions.add("fpn")
        elif name in {"rpn.shared_conv", "rpn.head.conv"}:
            regions.add("rpn_conv")
        elif name in {"rpn.classification", "rpn.head.cls_logits"}:
            regions.add("rpn_cls")
        elif name in {"rpn.bbox", "rpn.head.bbox_pred"}:
            regions.add("rpn_bbox")
        else:
            raise ValueError(f"Unsupported selective quantization module: {raw_name!r}")
    return regions


def prepare_selective_qat(
    model, variant="M3", backend="auto", inplace=False, quantized_modules=None,
):
    """Chèn fake-quant vào đúng các vùng được chọn bởi M0-M4.

    ``backend='auto'`` ưu tiên backend CPU phù hợp với runtime. ROI heads được
    kiểm tra lại sau prepare để tránh lượng tử hóa ngoài ý muốn.
    """
    variant = _validate_variant(variant)
    supported = list(torch.backends.quantized.supported_engines)
    requested_backend = str(backend).lower()
    if requested_backend == "auto":
        backend = next(
            (candidate for candidate in ("x86", "onednn", "fbgemm", "qnnpack") if candidate in supported),
            None,
        )
    elif requested_backend == "x86" and "x86" not in supported and "onednn" in supported:
        print("quantized backend x86 unavailable; using onednn", flush=True)
        backend = "onednn"
    if backend not in supported:
        raise ValueError(
            f"Quantized backend {backend!r} is unavailable; supported: "
            f"{supported}"
        )
    torch.backends.quantized.engine = backend
    qat_model = model if inplace else copy.deepcopy(model)
    qat_model.qconfig = None
    # Chọn vùng theo variant M0-M4 hoặc danh sách module cụ thể từ config.
    regions = (
        _regions_from_module_names(quantized_modules)
        if quantized_modules is not None
        else VARIANT_REGIONS[variant]
    )
    if not regions:
        qat_model.qat_variant = variant
        qat_model.quantized_backend = backend
        return qat_model
    qconfig = selective_qconfig()

    # Transform, decode, ROI và NMS không xuất hiện ở đây nên luôn giữ FP32.
    if "backbone" in regions:
        _wrap_operations(qat_model.backbone.body, qconfig)
    if "fpn" in regions:
        _wrap_operations(qat_model.backbone.fpn, qconfig)
    if "rpn_conv" in regions:
        _wrap_operations(qat_model.rpn.head.conv, qconfig)
    if "rpn_cls" in regions and not isinstance(qat_model.rpn.head.cls_logits, QuantizedOperation):
        wrapper = QuantizedOperation(qat_model.rpn.head.cls_logits)
        wrapper.qconfig = qconfig
        qat_model.rpn.head.cls_logits = wrapper
    if "rpn_bbox" in regions and not isinstance(qat_model.rpn.head.bbox_pred, QuantizedOperation):
        wrapper = QuantizedOperation(qat_model.rpn.head.bbox_pred)
        wrapper.qconfig = qconfig
        qat_model.rpn.head.bbox_pred = wrapper

    qat_model.train()
    prepare_qat(qat_model, inplace=True)
    if any(isinstance(module, QuantizedOperation) for module in qat_model.roi_heads.modules()):
        raise RuntimeError("Selective QAT boundary violation: ROI heads were quantized")
    qat_model.qat_variant = variant
    qat_model.quantized_backend = backend
    qat_model.quantized_module_names = list(quantized_modules or [])
    return qat_model


def set_qat_phase(model, phase):
    """Thiết lập observer/fake-quant cho từng phase QAT.

    ``calibration`` chỉ thu range; ``weight_only`` chỉ giả lập weight INT8;
    ``full`` giả lập W8A8; ``frozen`` cố định range nhưng vẫn fake-quant.
    """
    if phase not in {"calibration", "weight_only", "full", "frozen"}:
        raise ValueError("phase must be calibration, weight_only, full, or frozen")
    for name, module in model.named_modules():
        if not isinstance(module, FakeQuantize):
            continue
        is_weight = name.endswith("weight_fake_quant")
        observer_on = phase != "frozen"
        fake_on = phase in {"full", "frozen"} or (phase == "weight_only" and is_weight)
        module.observer_enabled.fill_(int(observer_on))
        module.fake_quant_enabled.fill_(int(fake_on))


def convert_selective_qat(model, inplace=False):
    """Convert các đảo đã prepare thành module INT8 để suy luận trên CPU."""
    converted = model if inplace else copy.deepcopy(model)
    converted.to("cpu").eval()
    if any(isinstance(module, QuantizedOperation) for module in converted.modules()):
        convert(converted, inplace=True)
    return converted


def quantized_region_summary(model):
    """Liệt kê tên các module đã thực sự được convert sang quantized module."""
    return [name for name, module in model.named_modules() if module.__class__.__module__.startswith("torch.ao.nn.quantized")]
