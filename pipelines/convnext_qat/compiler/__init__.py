"""Adapters tensor-only de export cac vung ConvNeXt/FPN/RPN sang compiler backend.

Faster R-CNN co nhieu control flow kho dua truc tiep vao TensorRT nhu RPN
decode, RoIAlign va NMS. Cac adapter o day chi tach vung backbone/FPN thanh
module nhan tensor va tra tuple tensor, de ONNX/TensorRT co mot graph on dinh.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class BackboneBodyAdapter(nn.Module):
    """Tra C2-C5 cua backbone body duoi dang tuple tensor."""

    def __init__(self, backbone):
        super().__init__()
        self.body = backbone.body
        self.feature_indices = tuple(getattr(backbone, "feature_indices", (1, 3, 5, 7)))

    def forward(self, x: torch.Tensor):
        """Chay body va lay cac stage feature ma Faster R-CNN dang dung."""
        outputs = []
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in self.feature_indices:
                outputs.append(x)
        return tuple(outputs)


class BackboneFPNAdapter(nn.Module):
    """Tra P2-P6 cua backbone+FPN duoi dang tuple tensor."""

    def __init__(self, backbone):
        super().__init__()
        self.body = BackboneBodyAdapter(backbone)
        self.fpn = backbone.fpn

    def forward(self, x: torch.Tensor):
        """Chay body -> FPN, bo OrderedDict de compiler chi thay tuple tensor."""
        c_features = self.body(x)
        feature_dict = OrderedDict((str(index), tensor) for index, tensor in enumerate(c_features))
        outputs = self.fpn(feature_dict)
        return tuple(outputs.values())


class BackboneFPNRPNM3Adapter(nn.Module):
    """Export vung tuong ung selective QAT M3.

    M3 trong repo luong tu hoa ``backbone.convnext``, ``backbone.fpn``,
    ``rpn.shared_conv`` va ``rpn.classification``. Adapter nay vi vay tra ve:

    1. cac feature map FPN P2-P6;
    2. cac feature sau RPN shared conv;
    3. cac objectness logits cua RPN classification.

    Nhánh bbox regression của RPN không thuộc M3 nên vẫn để PyTorch CUDA tính
    từ shared features trong benchmark/deploy hybrid.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone_fpn = BackboneFPNAdapter(model.backbone)
        self.rpn_shared_conv = model.rpn.head.conv
        self.rpn_cls_logits = model.rpn.head.cls_logits

    def forward(self, x: torch.Tensor):
        """Chay backbone+FPN+RPN shared/cls va tra tuple tensor compiler-friendly."""
        fpn_features = self.backbone_fpn(x)
        shared_features = tuple(self.rpn_shared_conv(feature) for feature in fpn_features)
        objectness_logits = tuple(self.rpn_cls_logits(feature) for feature in shared_features)
        return tuple(fpn_features) + tuple(shared_features) + tuple(objectness_logits)


def resolve_compiler_scope(config):
    """Doc scope TensorRT trong config."""
    compiler_cfg = config.get("quantization", {}).get("compiler", {})
    scope = str(compiler_cfg.get("scope", "backbone")).lower()
    aliases = {
        "m3": "backbone_fpn_rpn_m3",
        "backbone_fpn_rpn": "backbone_fpn_rpn_m3",
        "backbone_fpn_rpn_cls": "backbone_fpn_rpn_m3",
    }
    scope = aliases.get(scope, scope)
    if scope not in {"backbone", "backbone_fpn", "backbone_fpn_rpn_m3"}:
        raise ValueError(
            "quantization.compiler.scope must be backbone, backbone_fpn, "
            "or backbone_fpn_rpn_m3"
        )
    return scope


def build_compiler_target_module(model, config):
    """Tao module de export ONNX cho vung compiler da chon."""
    scope = resolve_compiler_scope(config)
    if scope == "backbone":
        return BackboneBodyAdapter(model.backbone)
    if scope == "backbone_fpn":
        return BackboneFPNAdapter(model.backbone)
    return BackboneFPNRPNM3Adapter(model)
