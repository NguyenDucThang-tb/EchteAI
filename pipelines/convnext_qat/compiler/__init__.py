"""Adapters tensor-only de export backbone/FPN sang compiler backend.

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


def resolve_compiler_scope(config):
    """Doc scope TensorRT trong config: backbone hoac backbone_fpn."""
    compiler_cfg = config.get("quantization", {}).get("compiler", {})
    scope = str(compiler_cfg.get("scope", "backbone")).lower()
    if scope not in {"backbone", "backbone_fpn"}:
        raise ValueError("quantization.compiler.scope must be backbone or backbone_fpn")
    return scope


def build_compiler_target_module(model, config):
    """Tao module de export ONNX cho vung compiler da chon."""
    scope = resolve_compiler_scope(config)
    backbone = model.backbone
    if scope == "backbone":
        return BackboneBodyAdapter(backbone)
    return BackboneFPNAdapter(backbone)

