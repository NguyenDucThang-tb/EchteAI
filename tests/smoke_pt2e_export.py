"""Smoke-check the tensor-only backbone PT2E export boundary without TorchAO."""

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.convnext_qat.models.convnext_fpn_backbone import ConvNeXtFPNBackbone, ResNetFPNBackbone
from pipelines.convnext_qat.quantization.pt2e_qat import (
    BackboneBodyFPNRegion,
    BackboneBodyRegion,
    ResNet50BodyFPNRegion,
    ResNet50BodyRegion,
    _dynamic_shapes,
)


def run_body(backbone, expected_channels, name):
    original = copy.deepcopy(backbone).eval()
    if getattr(backbone, "pt2e_region_kind", "") == "resnet50":
        region = ResNet50BodyRegion(backbone.body).eval()
    else:
        region = BackboneBodyRegion(backbone.body, backbone.feature_indices).eval()
    example = torch.randn(2, 3, 256, 320)
    if getattr(backbone, "pt2e_region_kind", "") == "resnet50":
        reference = []
        value = example
        with torch.inference_mode():
            for index, layer in enumerate(original.body):
                value = layer(value)
                if index in original.feature_indices:
                    reference.append(value)
            rewritten = region(example)
        for expected, actual in zip(reference, rewritten):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    exported = torch.export.export(
        region,
        (example,),
        dynamic_shapes=_dynamic_shapes(2, 2, 224, 1024),
    ).module()
    outputs = exported(torch.randn(1, 3, 288, 352))
    assert len(outputs) == 4
    assert [tensor.shape[1] for tensor in outputs] == expected_channels
    assert [tuple(tensor.shape[-2:]) for tensor in outputs] == [
        (72, 88), (36, 44), (18, 22), (9, 11),
    ]
    print(f"PT2E dynamic {name} export smoke test passed")


def run_fpn(backbone, name):
    if getattr(backbone, "pt2e_region_kind", "") == "resnet50":
        region = ResNet50BodyFPNRegion(backbone.body, backbone.fpn).eval()
    else:
        region = BackboneBodyFPNRegion(
            backbone.body, backbone.fpn, backbone.feature_indices,
        ).eval()
    example = torch.randn(2, 3, 256, 320)
    exported = torch.export.export(
        region,
        (example,),
        dynamic_shapes=_dynamic_shapes(2, 2, 256, 1024, spatial_divisor=64),
    ).module()
    outputs = exported(torch.randn(1, 3, 320, 384))
    assert len(outputs) == 5
    assert all(tensor.shape[1] == 256 for tensor in outputs)
    print(f"PT2E dynamic {name}+FPN export smoke test passed")


def main():
    try:
        _dynamic_shapes(1, 2, 224, 1024)
    except ValueError as error:
        assert "example_batch_size >= 2" in str(error)
    else:
        raise AssertionError("dynamic batch export with an example batch of one must fail")

    run_body(ConvNeXtFPNBackbone(pretrained=False), [96, 192, 384, 768], "ConvNeXt")
    run_body(ResNetFPNBackbone(pretrained=False), [256, 512, 1024, 2048], "ResNet50")
    run_fpn(ConvNeXtFPNBackbone(pretrained=False), "ConvNeXt")
    run_fpn(ResNetFPNBackbone(pretrained=False), "ResNet50")


if __name__ == "__main__":
    main()
