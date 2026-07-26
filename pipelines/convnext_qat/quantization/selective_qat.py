"""Eager selective QAT boundaries for ConvNeXt-FPN Faster R-CNN.

Every selected Conv/Linear is an INT8 island after conversion. Its input is
quantized immediately before the operator and dequantized immediately after it.
Consequently LayerNorm, GELU, residual addition, proposal code, ROI Align and ROI
heads retain their normal FP32 tensors and kernels.
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
    """Quant/operation/dequant island; QAT prepares and converts only this scope."""

    def __init__(self, operation):
        super().__init__()
        self.quant = QuantStub()
        self.operation = operation
        self.dequant = DeQuantStub()

    def forward(self, x):
        return self.dequant(self.operation(self.quant(x)))


def selective_qconfig(weight_bits=8, activation_bits=8, profile="eager"):
    if not 2 <= int(weight_bits) <= 8:
        raise ValueError("weight_bits must be between 2 and 8")
    if not 2 <= int(activation_bits) <= 8:
        raise ValueError("activation_bits must be between 2 and 8")
    profile = str(profile).strip().lower()
    if profile not in {"eager", "tensorrt"}:
        raise ValueError("profile must be 'eager' or 'tensorrt'")
    weight_quant_min = -(1 << (int(weight_bits) - 1))
    weight_quant_max = (1 << (int(weight_bits) - 1)) - 1
    if profile == "tensorrt":
        activation_dtype = torch.qint8
        activation_qscheme = torch.per_tensor_symmetric
        activation_quant_min = -(1 << (int(activation_bits) - 1))
        activation_quant_max = (1 << (int(activation_bits) - 1)) - 1
    else:
        activation_dtype = torch.quint8
        activation_qscheme = torch.per_tensor_affine
        activation_quant_min = 0
        activation_quant_max = (1 << int(activation_bits)) - 1
    return QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            dtype=activation_dtype,
            qscheme=activation_qscheme,
            quant_min=activation_quant_min,
            quant_max=activation_quant_max,
        ),
        weight=FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            quant_min=weight_quant_min,
            quant_max=weight_quant_max,
            ch_axis=0,
        ),
    )


def _wrap_operations(module, default_qconfig, prefix="", module_qconfig_map=None):
    for name, child in list(module.named_children()):
        child_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, QuantizedOperation):
            continue
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            qconfig = (
                (module_qconfig_map or {}).get(child_name, default_qconfig)
            )
            if qconfig is None:
                continue
            wrapped = QuantizedOperation(child)
            wrapped.qconfig = qconfig
            setattr(module, name, wrapped)
        else:
            _wrap_operations(child, default_qconfig, prefix=child_name, module_qconfig_map=module_qconfig_map)


def _validate_variant(variant):
    variant = str(variant).upper()
    if variant not in VARIANT_REGIONS:
        raise ValueError(f"Unknown QAT variant {variant!r}; choose {sorted(VARIANT_REGIONS)}")
    return variant


def _regions_from_module_names(module_names):
    regions = set()
    for raw_name in module_names:
        name = str(raw_name).lower()
        if name in {"backbone.convnext", "backbone.resnet50", "backbone.body"}:
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
    module_qconfig_map=None,
    qconfig_profile="eager",
):
    """Insert fake quant only into regions selected by M0-M4."""
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
    regions = (
        _regions_from_module_names(quantized_modules)
        if quantized_modules is not None
        else VARIANT_REGIONS[variant]
    )
    if not regions:
        qat_model.qat_variant = variant
        qat_model.quantized_backend = backend
        return qat_model
    qconfig = selective_qconfig(profile=qconfig_profile)

    if "backbone" in regions:
        _wrap_operations(
            qat_model.backbone.body,
            qconfig if module_qconfig_map is None else None,
            prefix="backbone.body",
            module_qconfig_map=module_qconfig_map,
        )
    if "fpn" in regions:
        _wrap_operations(
            qat_model.backbone.fpn,
            qconfig if module_qconfig_map is None else None,
            prefix="backbone.fpn",
            module_qconfig_map=module_qconfig_map,
        )
    if "rpn_conv" in regions:
        _wrap_operations(qat_model.rpn.head.conv, qconfig, prefix="rpn.head.conv")
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
    qat_model.mixed_precision_module_qconfigs = sorted((module_qconfig_map or {}).keys())
    qat_model.qat_qconfig_profile = str(qconfig_profile).strip().lower()
    return qat_model


def set_qat_phase(model, phase):
    """Configure calibration, weight-only warmup, full QAT, or frozen observers."""
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
    """Convert prepared islands to quantized CPU inference modules."""
    converted = model if inplace else copy.deepcopy(model)
    converted.to("cpu").eval()
    if any(isinstance(module, QuantizedOperation) for module in converted.modules()):
        convert(converted, inplace=True)
    return converted


def quantized_region_summary(model):
    return [name for name, module in model.named_modules() if module.__class__.__module__.startswith("torch.ao.nn.quantized")]
