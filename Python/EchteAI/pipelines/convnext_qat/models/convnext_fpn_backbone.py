"""ConvNeXt C2-C5 feature extraction with a torchvision FPN."""

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_small,
    convnext_tiny,
)
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


_VARIANTS = {
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, [96, 192, 384, 768]),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, [96, 192, 384, 768]),
}


class ConvNeXtFPNBackbone(nn.Module):
    """Expose ConvNeXt stages as C2-C5 and return P2-P6 in an OrderedDict."""

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

        # A trainable stage includes the projection feeding it. This ensures that
        # trainable_layers=4 also trains the stem and every downsample projection.
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
        features = OrderedDict()
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in (1, 3, 5, 7):
                features[str(len(features))] = x
        return self.fpn(features)


def build_convnext_fpn_backbone(model_config):
    return ConvNeXtFPNBackbone(
        variant=model_config.get("backbone", "convnext_tiny"),
        out_channels=int(model_config.get("fpn_out_channels", 256)),
        pretrained=bool(model_config.get("pretrained_backbone", True)),
        trainable_layers=int(model_config.get("trainable_backbone_layers", 4)),
    )
