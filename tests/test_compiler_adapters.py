"""Test nho cho adapter compiler dung khi export TensorRT."""

import unittest
from collections import OrderedDict

import torch
from torch import nn

from pipelines.convnext_qat.compiler import (
    BackboneBodyAdapter,
    BackboneFPNAdapter,
    build_compiler_target_module,
    resolve_compiler_scope,
)


class _ToyFPN(nn.Module):
    """FPN gia lap: nhan OrderedDict va tra lai OrderedDict cung key."""

    def forward(self, features):
        return OrderedDict((key, value + 1.0) for key, value in features.items())


class _ToyBackbone(nn.Module):
    """Backbone toi gian co cung giao dien body/fpn/feature_indices."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 4, 1),
            nn.Conv2d(4, 4, 1),
            nn.Conv2d(4, 4, 1),
        )
        self.fpn = _ToyFPN()
        self.feature_indices = (0, 2)


class _ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ToyBackbone()


class CompilerAdapterTests(unittest.TestCase):
    """Khoa cac contract co ban cua TensorRT export adapter."""

    def test_backbone_body_adapter_returns_selected_features(self):
        adapter = BackboneBodyAdapter(_ToyBackbone()).eval()
        outputs = adapter(torch.randn(1, 3, 8, 8))
        self.assertEqual(len(outputs), 2)
        self.assertEqual(tuple(outputs[0].shape), (1, 4, 8, 8))
        self.assertEqual(tuple(outputs[1].shape), (1, 4, 8, 8))

    def test_backbone_fpn_adapter_returns_tuple_outputs(self):
        adapter = BackboneFPNAdapter(_ToyBackbone()).eval()
        outputs = adapter(torch.randn(1, 3, 8, 8))
        self.assertEqual(len(outputs), 2)
        self.assertTrue(torch.all(outputs[0] > -1000))

    def test_build_compiler_target_module_uses_config_scope(self):
        detector = _ToyDetector()
        config = {"quantization": {"compiler": {"scope": "backbone_fpn"}}}
        self.assertEqual(resolve_compiler_scope(config), "backbone_fpn")
        self.assertIsInstance(build_compiler_target_module(detector, config), BackboneFPNAdapter)


if __name__ == "__main__":
    unittest.main()

