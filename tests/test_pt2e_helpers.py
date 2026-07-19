"""Các test hồi quy PT2E nhanh, không phải export toàn detector."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.convnext_qat.quantization import (
    inspect_pt2e_graph,
    load_pt2e_int8_artifact,
    pt2e_observers_disabled,
    pt2e_qat_phase,
    set_pt2e_qat_phase,
    synchronize_pt2e_observers,
    validate_pt2e_schedule,
)
from scripts.debug_pt2e_convert import choose_device, compare_sample_records


class FakeQuantizer(nn.Module):
    """Fake quantizer tối giản để test điều khiển observer/phase."""
    def __init__(self):
        super().__init__()
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.int64))
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.int64))
        self.activation_post_process = nn.Module()
        self.activation_post_process.register_buffer("min_val", torch.tensor([-1.0]))
        self.activation_post_process.register_buffer("max_val", torch.tensor([2.0]))

    def enable_observer(self):
        """Bật cờ observer giả."""
        self.observer_enabled.fill_(1)

    def disable_observer(self):
        """Tắt cờ observer giả."""
        self.observer_enabled.fill_(0)

    def enable_fake_quant(self):
        """Bật cờ fake-quant giả."""
        self.fake_quant_enabled.fill_(1)

    def disable_fake_quant(self):
        """Tắt cờ fake-quant giả."""
        self.fake_quant_enabled.fill_(0)


class PT2EHelperTests(unittest.TestCase):
    """Bảo vệ lịch phase, đồng bộ DDP và tính toàn vẹn artifact PT2E."""
    def test_schedule_covers_warmup_full_and_frozen(self):
        """Lịch nhiều epoch phải đi qua đủ warmup, full và frozen."""
        self.assertEqual(
            [pt2e_qat_phase(epoch, 4, 1, 1) for epoch in range(4)],
            ["observer_warmup", "full", "full", "frozen"],
        )

    def test_single_epoch_must_not_be_all_frozen(self):
        """Một epoch không được đóng observer khi chưa từng calibration."""
        with self.assertRaisesRegex(ValueError, "never be calibrated"):
            validate_pt2e_schedule(1, 0, 1)
        self.assertEqual(pt2e_qat_phase(0, 1, 0, 0), "full")

    def test_phase_flags_and_validation_context_are_restored(self):
        """Context validation phải khôi phục cờ observer sau khi thoát."""
        model = nn.Sequential(FakeQuantizer(), FakeQuantizer())
        self.assertEqual(set_pt2e_qat_phase(model, "observer_warmup"), 2)
        self.assertTrue(all(module.observer_enabled.item() for module in model))
        self.assertTrue(all(not module.fake_quant_enabled.item() for module in model))
        set_pt2e_qat_phase(model, "full")
        with pt2e_observers_disabled(model) as count:
            self.assertEqual(count, 2)
            self.assertTrue(all(not module.observer_enabled.item() for module in model))
        self.assertTrue(all(module.observer_enabled.item() for module in model))

    def test_graph_audit_handles_non_graph_model(self):
        """Graph audit trả số 0 an toàn với model thường."""
        self.assertEqual(
            inspect_pt2e_graph(nn.Linear(2, 2)),
            {"nodes": 0, "quantize": 0, "dequantize": 0, "quantized_ops": 0},
        )

    def test_ddp_observer_sync_reduces_min_and_max(self):
        """DDP sync phải all-reduce MIN và MAX đúng thứ tự."""
        model = nn.Sequential(FakeQuantizer())
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.all_reduce") as all_reduce,
        ):
            self.assertEqual(synchronize_pt2e_observers(model), 1)
        self.assertEqual(all_reduce.call_count, 2)
        self.assertIs(all_reduce.call_args_list[0].kwargs["op"], torch.distributed.ReduceOp.MIN)
        self.assertIs(all_reduce.call_args_list[1].kwargs["op"], torch.distributed.ReduceOp.MAX)

    def test_malformed_artifact_fails_before_model_reconstruction(self):
        """Artifact thiếu graph phải dừng trước khi dựng detector."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save({"extra": {"format": "pt2e_int8_exported_region"}}, path)
            with self.assertRaisesRegex(ValueError, "missing exported_region"):
                load_pt2e_int8_artifact(path, {})

    def test_legacy_state_dict_artifact_is_rejected(self):
        """Artifact state_dict cũ bị từ chối vì làm mất qparam graph."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"model": {}, "extra": {"format": "pt2e_int8_state_dict"}}, path)
            with self.assertRaisesRegex(ValueError, "FX graph qparam constants"):
                load_pt2e_int8_artifact(path, {})

    def test_reload_comparison_detects_numeric_drift(self):
        """Bộ so sánh reload phải phát hiện sai lệch score."""
        record = [{
            "pred_boxes": 1,
            "top_labels": [1],
            "top_scores": [0.8],
            "top_boxes": [[1.0, 2.0, 3.0, 4.0]],
        }]
        self.assertTrue(compare_sample_records(record, record)["equivalent"])
        drifted = [{**record[0], "top_scores": [0.7]}]
        result = compare_sample_records(record, drifted)
        self.assertFalse(result["equivalent"])
        self.assertEqual(result["reason"], "numeric_delta")

    def test_x86_convert_debug_rejects_cuda_comparison(self):
        """Debug converted x86 không cho chạy nhầm trên CUDA."""
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            choose_device("cuda")


if __name__ == "__main__":
    unittest.main()
