"""Small CPU smoke test; no dataset or pretrained download is required."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from EchteAI.pipelines.convnext_qat.models import build_fasterrcnn_convnext
from EchteAI.pipelines.convnext_qat.quantization.selective_qat import (
    convert_selective_qat,
    prepare_selective_qat,
    quantized_region_summary,
    set_qat_phase,
)


def smoke_config():
    return {
        "dataset": {"num_classes": 3},
        "model": {
            "backbone": "convnext_tiny",
            "pretrained_backbone": False,
            "trainable_backbone_layers": 4,
            "fpn_out_channels": 256,
            "min_size": 64,
            "max_size": 64,
            "anchor_sizes": [8, 16, 32, 64, 128],
            "aspect_ratios": [0.5, 1.0, 2.0],
            "rpn_pre_nms_top_n_train": 40,
            "rpn_pre_nms_top_n_test": 20,
            "rpn_post_nms_top_n_train": 20,
            "rpn_post_nms_top_n_test": 10,
        },
    }


def main():
    torch.set_num_threads(1)
    model = build_fasterrcnn_convnext(smoke_config())
    image = torch.rand(3, 64, 64)
    target = {"boxes": torch.tensor([[8.0, 8.0, 40.0, 48.0]]), "labels": torch.tensor([1])}
    model.train()
    losses = model([image], [target])
    assert all(torch.isfinite(value) for value in losses.values())

    qat = prepare_selective_qat(model, "M3", "x86")
    set_qat_phase(qat, "full")
    qat.train()
    qat_losses = qat([image], [target])
    qat_total = sum(qat_losses.values())
    assert torch.isfinite(qat_total)
    qat_total.backward()
    assert qat.roi_heads.box_predictor.cls_score.weight.grad is not None
    qat.eval()
    with torch.no_grad():
        qat([image])
    converted = convert_selective_qat(qat)
    assert quantized_region_summary(converted), "no operators were converted"
    with torch.no_grad():
        output = converted([image])
    assert {"boxes", "labels", "scores"} <= output[0].keys()
    assert all(torch.isfinite(output[0][key]).all() for key in ("boxes", "scores"))
    names = quantized_region_summary(converted)
    assert not any(name.startswith("roi_heads") for name in names)
    assert not any(name.startswith("rpn.head.bbox_pred") for name in names)
    print("smoke test passed", len(quantized_region_summary(converted)), "quantized modules")


if __name__ == "__main__":
    main()
