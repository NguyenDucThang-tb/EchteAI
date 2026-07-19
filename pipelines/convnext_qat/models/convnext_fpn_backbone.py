"""Trích xuất đặc trưng C2-C5 và xây dựng tháp đặc trưng FPN P2-P6."""

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    convnext_small,
    convnext_tiny,
    resnet50,
)
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


_VARIANTS = {
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, [96, 192, 384, 768]),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, [96, 192, 384, 768]),
}


class ConvNeXtFPNBackbone(nn.Module):
    """Backbone ConvNeXt: lấy bốn stage C2-C5 rồi chuẩn hóa bằng FPN."""

    def __init__(self, variant="convnext_tiny", out_channels=256, pretrained=True, trainable_layers=4):
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(f"Unsupported backbone {variant!r}; choose {sorted(_VARIANTS)}")
        if not 0 <= trainable_layers <= 4:
            raise ValueError("trainable_backbone_layers must be between 0 and 4")

        constructor, default_weights, stage_channels = _VARIANTS[variant]
        network = constructor(weights=default_weights if pretrained else None)
        self.body = network.features
        self.fpn = FeaturePyramidNetwork(
            stage_channels, out_channels, extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = out_channels
        self.feature_indices = (1, 3, 5, 7)
        self.pt2e_spatial_divisor = 32
        self.pt2e_region_kind = "convnext"

        # Mỗi stage gồm projection đầu vào và các block; mở đủ bốn stage đồng
        # nghĩa stem/downsample cũng nhận gradient.
        stage_groups = ((0, 1), (2, 3), (4, 5), (6, 7))
        trainable_stage_indices = {
            index
            for group in stage_groups[-trainable_layers:] if trainable_layers
            for index in group
        }
        for index, child in enumerate(self.body):
            requires_grad = index in trainable_stage_indices
            for parameter in child.parameters():
                parameter.requires_grad_(requires_grad)

    def forward(self, x: torch.Tensor):
        """Nhận batch ``[N,3,H,W]`` và trả OrderedDict các mức P2-P6."""
        features = OrderedDict()
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in self.feature_indices:
                features[str(len(features))] = x
        return self.fpn(features)


class _ResNetStem(nn.Module):
    """Gói conv7x7 + BN + ReLU + max-pool đầu ResNet thành một stage."""
    def __init__(self, network):
        super().__init__()
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool

    def forward(self, x):
        """Chạy lần lượt stem của ResNet."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ResNetFPNBackbone(nn.Module):
    """Backbone ResNet50 tương thích cùng giao diện C2-C5/P2-P6."""

    def __init__(self, variant="resnet50", out_channels=256, pretrained=True, trainable_layers=5):
        super().__init__()
        if variant != "resnet50":
            raise ValueError("Only resnet50 is currently supported for ResNet FPN")
        if not 0 <= trainable_layers <= 5:
            raise ValueError("trainable_backbone_layers must be between 0 and 5 for ResNet")

        network = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.body = nn.Sequential(
            _ResNetStem(network),
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        stage_channels = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            stage_channels, out_channels, extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = out_channels
        self.feature_indices = (1, 2, 3, 4)
        self.pt2e_spatial_divisor = 32
        self.pt2e_region_kind = "resnet50"

        stage_indices = tuple(range(len(self.body)))
        trainable_stage_indices = set(stage_indices[-trainable_layers:]) if trainable_layers else set()
        for index, child in enumerate(self.body):
            requires_grad = index in trainable_stage_indices
            for parameter in child.parameters():
                parameter.requires_grad_(requires_grad)

    def forward(self, x: torch.Tensor):
        """Chạy stem/layer1-4 rồi đưa bốn feature map qua FPN."""
        features = OrderedDict()
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in self.feature_indices:
                features[str(len(features))] = x
        return self.fpn(features)


def build_fpn_backbone(model_config):
    """Factory chọn ConvNeXt hoặc ResNet50 theo ``model.backbone``."""
    variant = str(model_config.get("backbone", "convnext_tiny")).lower()
    common_kwargs = dict(
        out_channels=int(model_config.get("fpn_out_channels", 256)),
        pretrained=bool(model_config.get("pretrained_backbone", True)),
        trainable_layers=int(
            model_config.get("trainable_backbone_layers", 5 if variant == "resnet50" else 4)
        ),
    )
    if variant in _VARIANTS:
        return ConvNeXtFPNBackbone(variant=variant, **common_kwargs)
    if variant == "resnet50":
        return ResNetFPNBackbone(variant=variant, **common_kwargs)
    raise ValueError(
        f"Unsupported backbone {variant!r}; choose {sorted(_VARIANTS) + ['resnet50']}"
    )


def build_convnext_fpn_backbone(model_config):
    """Alias tương thích code cũ; thực tế hỗ trợ cả ConvNeXt và ResNet50."""
    return build_fpn_backbone(model_config)
