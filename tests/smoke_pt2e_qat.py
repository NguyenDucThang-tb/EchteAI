"""Smoke test PT2E QAT end-to-end: resume, convert, artifact và inference CPU."""

import os
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.checkpoint import load_checkpoint, save_checkpoint
from pipelines.convnext_qat.models import build_fasterrcnn_convnext
from pipelines.convnext_qat.quantization import (
    convert_pt2e_backbone, inspect_pt2e_graph, load_pt2e_int8_artifact,
    prepare_pt2e_backbone_qat, pt2e_observers_disabled, pt2e_qat_phase,
    save_pt2e_int8_artifact, set_pt2e_qat_phase, validate_pt2e_schedule,
)


def config(backbone="convnext_tiny"):
    """Tạo config tối thiểu, không tải pretrained weight/dataset."""
    return {
        "dataset": {"num_classes": 3},
        "model": {
            "backbone": backbone, "pretrained_backbone": False,
            "trainable_backbone_layers": 4, "fpn_out_channels": 256,
            "min_size": 64, "max_size": 64,
            "anchor_sizes": [8, 16, 32, 64, 128], "aspect_ratios": [0.5, 1.0, 2.0],
            "rpn_pre_nms_top_n_train": 40, "rpn_pre_nms_top_n_test": 20,
            "rpn_post_nms_top_n_train": 20, "rpn_post_nms_top_n_test": 10,
        },
        "training": {"qat_batch_size": 1},
        "quantization": {"pt2e": {
            "region": "backbone", "example_batch_size": 1, "maximum_batch_size": 1,
            "minimum_image_side": 64, "maximum_image_side": 128,
            "example_height": 64, "example_width": 64,
        }},
    }


def prepared(cfg):
    """Dựng detector nhỏ và prepare vùng PT2E."""
    return prepare_pt2e_backbone_qat(build_fasterrcnn_convnext(cfg), cfg)


def main():
    """Kiểm tra backward, checkpoint, phase, convert, save/reload và shape động."""
    torch.set_num_threads(1)
    assert [pt2e_qat_phase(i, 3, 1, 1) for i in range(3)] == [
        "observer_warmup", "full", "frozen",
    ]
    assert pt2e_qat_phase(0, 1, 0, 0) == "full"
    try:
        validate_pt2e_schedule(1, 0, 1)
    except ValueError as error:
        assert "never be calibrated" in str(error)
    else:
        raise AssertionError("a one-epoch all-frozen schedule must be rejected")

    cfg = config()
    image = torch.rand(3, 64, 64)
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 40.0, 48.0]]),
        "labels": torch.tensor([1]),
    }
    model = prepared(cfg)
    assert set_pt2e_qat_phase(model, "observer_warmup") > 0
    model([image], [target])
    set_pt2e_qat_phase(model, "full")
    observer_states = [
        module.observer_enabled.detach().clone()
        for module in model.modules() if hasattr(module, "observer_enabled")
    ]
    with pt2e_observers_disabled(model) as observer_count:
        assert observer_count == len(observer_states) > 0
    restored_states = [
        module.observer_enabled.detach().clone()
        for module in model.modules() if hasattr(module, "observer_enabled")
    ]
    assert all(torch.equal(before, after) for before, after in zip(observer_states, restored_states))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss = sum(model([image], [target]).values())
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "qat.pt"
        artifact = Path(directory) / "int8.pt"
        incompatible_artifact = Path(directory) / "int8_incompatible.pt"
        save_checkpoint(checkpoint, model, optimizer, epoch=1)
        resumed = prepared(cfg)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-5)
        load_checkpoint(checkpoint, resumed, resumed_optimizer)
        set_pt2e_qat_phase(resumed, "frozen")
        resumed.eval()
        resumed([image])
        converted = convert_pt2e_backbone(resumed, inplace=True)
        output = converted([image])[0]
        assert {"boxes", "scores", "labels"} <= output.keys()
        graph = inspect_pt2e_graph(converted)
        assert graph["quantize"] > 0 and graph["dequantize"] > 0, graph
        save_pt2e_int8_artifact(artifact, converted, {"map_50_95": 0.0})
        loaded, payload = load_pt2e_int8_artifact(artifact, cfg)
        assert payload["extra"]["format"] == "pt2e_int8_exported_region"
        assert payload["extra"]["format_version"] == 3
        reloaded_output = loaded([image])[0]
        assert torch.equal(output["labels"], reloaded_output["labels"])
        torch.testing.assert_close(output["boxes"], reloaded_output["boxes"], rtol=0, atol=1e-5)
        torch.testing.assert_close(output["scores"], reloaded_output["scores"], rtol=0, atol=1e-6)
        dynamic_input = torch.rand(1, 3, 96, 128)
        for expected, actual in zip(
            converted.backbone.body_region(dynamic_input),
            loaded.backbone.body_region(dynamic_input),
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=1e-6)

        incompatible_payload = torch.load(artifact, map_location="cpu", weights_only=False)
        incompatible_payload["extra"]["model_signature"]["anchor_sizes"][0] = [999.0]
        torch.save(incompatible_payload, incompatible_artifact)
        try:
            load_pt2e_int8_artifact(incompatible_artifact, cfg)
        except ValueError as error:
            assert "incompatible with the active config" in str(error)
        else:
            raise AssertionError("artifact/config signature mismatch must be rejected")

        try:
            load_pt2e_int8_artifact(artifact, config("resnet50"))
        except ValueError as error:
            assert "backbone does not match" in str(error)
        else:
            raise AssertionError("artifact/backbone mismatch must be rejected")
        if os.environ.get("RUN_PT2E_COMPILE") == "1":
            from pipelines.convnext_qat.quantization import compile_pt2e_region
            compile_pt2e_region(loaded)
            loaded([image])

    # The remote branch also supports ResNet-50. Exercise its explicit C2-C5
    # region through prepare, calibration, convert, artifact save, and reload.
    resnet_cfg = config("resnet50")
    resnet = prepared(resnet_cfg)
    set_pt2e_qat_phase(resnet, "full")
    resnet([image], [target])
    set_pt2e_qat_phase(resnet, "frozen")
    resnet.eval()
    resnet_converted = convert_pt2e_backbone(resnet, inplace=True)
    resnet_output = resnet_converted([image])[0]
    with tempfile.TemporaryDirectory() as directory:
        resnet_artifact = Path(directory) / "resnet50_int8.pt"
        save_pt2e_int8_artifact(resnet_artifact, resnet_converted)
        resnet_loaded, resnet_payload = load_pt2e_int8_artifact(resnet_artifact, resnet_cfg)
        assert resnet_payload["extra"]["model_signature"]["backbone_kind"] == "resnet50"
        resnet_reloaded_output = resnet_loaded([image])[0]
        assert torch.equal(resnet_output["labels"], resnet_reloaded_output["labels"])
        torch.testing.assert_close(
            resnet_output["scores"], resnet_reloaded_output["scores"], rtol=0, atol=1e-6,
        )
    print("PT2E QAT end-to-end smoke test passed")


if __name__ == "__main__":
    main()
