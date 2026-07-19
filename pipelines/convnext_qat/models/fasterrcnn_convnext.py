"""Lắp Faster R-CNN + FPN và thay classification loss bằng Focal Loss."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import roi_heads as roi_heads_module
from torchvision.models.detection import rpn as rpn_module
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .convnext_fpn_backbone import build_fpn_backbone
from ..anchors import resolve_anchor_sizes


_FOCAL_ALPHA = None
_FOCAL_GAMMA = 2.0


def _anchors(config):
    """Tạo một scale anchor cho mỗi mức P2-P6 và dùng chung aspect ratio."""
    sizes = resolve_anchor_sizes(config)
    model_cfg = config["model"]
    ratios = tuple(float(value) for value in model_cfg.get("aspect_ratios", (0.5, 1.0, 2.0)))
    if len(sizes) != 5:
        raise ValueError("anchor_sizes must contain five values for P2-P6")
    if not ratios or any(value <= 0 for value in ratios):
        raise ValueError("aspect_ratios must contain positive values")
    return AnchorGenerator(tuple((size,) for size in sizes), (ratios,) * len(sizes))


def _sigmoid_focal_loss(inputs, targets, alpha=_FOCAL_ALPHA, gamma=_FOCAL_GAMMA):
    """Focal BCE nhị phân cho objectness foreground/background của RPN."""
    targets = targets.to(dtype=inputs.dtype)
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    probs = torch.sigmoid(inputs)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(gamma)
    if alpha is None:
        alpha_factor = 1.0
    else:
        alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_factor * modulating * bce_loss


def _softmax_focal_loss(class_logits, labels, alpha=_FOCAL_ALPHA, gamma=_FOCAL_GAMMA):
    """Focal Cross Entropy đa lớp cho nhánh phân loại RoI."""
    valid = labels >= 0
    if not torch.any(valid):
        return class_logits.sum() * 0.0

    class_logits = class_logits[valid]
    labels = labels[valid]

    ce_loss = F.cross_entropy(class_logits, labels, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = (1.0 - pt).pow(gamma) * ce_loss
    if alpha is not None:
        alpha_factor = torch.where(
            labels > 0,
            torch.full_like(loss, alpha),
            torch.full_like(loss, 1.0 - alpha),
        )
        loss = alpha_factor * loss
    return loss.sum()


def _focal_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, alpha=_FOCAL_ALPHA, gamma=_FOCAL_GAMMA):
    """Ghép Focal classification với Smooth-L1 box regression nguyên bản."""
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = _softmax_focal_loss(class_logits, labels, alpha=alpha, gamma=gamma)

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    box_regression = box_regression.reshape(class_logits.shape[0], -1, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1.0 / 9.0,
        reduction="sum",
    )
    normalizer = max(int(labels.numel()), 1)
    return classification_loss / normalizer, box_loss / normalizer


class FocalRegionProposalNetwork(rpn_module.RegionProposalNetwork):
    """RPN giữ sampling/regression gốc, chỉ đổi objectness sang Focal Loss."""
    def __init__(self, *args, focal_alpha: float = _FOCAL_ALPHA, focal_gamma: float = _FOCAL_GAMMA, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        """Tính loss trên tập positive/negative anchor đã được sampler chọn."""
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        objectness_loss = _sigmoid_focal_loss(
            objectness[sampled_inds],
            labels[sampled_inds].to(dtype=objectness.dtype),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        ).sum()

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9.0,
            reduction="sum",
        )
        normalizer = max(int(sampled_inds.numel()), 1)
        return objectness_loss / normalizer, box_loss / normalizer


class FocalRoIHeads(roi_heads_module.RoIHeads):
    """RoI heads dùng Focal Loss khi train và decode detection như bản gốc."""
    def __init__(self, *args, focal_alpha: float = _FOCAL_ALPHA, focal_gamma: float = _FOCAL_GAMMA, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        """Train trả dict loss; eval trả boxes/labels/scores cho từng ảnh."""
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must be of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must be of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: list[dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = _focal_fastrcnn_loss(
                class_logits,
                box_regression,
                labels,
                regression_targets,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i]})

        return result, losses


def _install_focal_loss(model, model_cfg):
    """Thay RPN/RoIHeads theo từng model instance, không monkeypatch toàn cục."""
    configured_alpha = model_cfg.get("focal_alpha")
    focal_alpha = None if configured_alpha is None else float(configured_alpha)
    focal_gamma = float(model_cfg.get("focal_gamma", 2.0))

    original_rpn = model.rpn
    model.rpn = FocalRegionProposalNetwork(
        original_rpn.anchor_generator,
        original_rpn.head,
        original_rpn.proposal_matcher.high_threshold,
        original_rpn.proposal_matcher.low_threshold,
        original_rpn.fg_bg_sampler.batch_size_per_image,
        original_rpn.fg_bg_sampler.positive_fraction,
        original_rpn._pre_nms_top_n,
        original_rpn._post_nms_top_n,
        original_rpn.nms_thresh,
        original_rpn.score_thresh,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )
    model.rpn.train(original_rpn.training)

    original_roi_heads = model.roi_heads
    model.roi_heads = FocalRoIHeads(
        original_roi_heads.box_roi_pool,
        original_roi_heads.box_head,
        original_roi_heads.box_predictor,
        original_roi_heads.proposal_matcher.high_threshold,
        original_roi_heads.proposal_matcher.low_threshold,
        original_roi_heads.fg_bg_sampler.batch_size_per_image,
        original_roi_heads.fg_bg_sampler.positive_fraction,
        original_roi_heads.box_coder.weights,
        original_roi_heads.score_thresh,
        original_roi_heads.nms_thresh,
        original_roi_heads.detections_per_img,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )
    model.roi_heads.train(original_roi_heads.training)


def build_fasterrcnn_convnext(config):
    """Xây model FP32; selective/PT2E QAT được áp dụng ở bước riêng sau đó."""
    model_cfg = config["model"]
    num_classes = int(config["dataset"]["num_classes"])
    if num_classes < 2:
        raise ValueError("dataset.num_classes must include background and at least one class")

    backbone = build_fpn_backbone(model_cfg)
    min_size = model_cfg.get("train_min_sizes", model_cfg.get("min_size", 640))
    if isinstance(min_size, list):
        min_size = tuple(int(value) for value in min_size)
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=int(model_cfg.get("max_size", 1024)),
        rpn_anchor_generator=_anchors(config),
        box_roi_pool=roi_pooler,
        rpn_pre_nms_top_n_train=int(model_cfg.get("rpn_pre_nms_top_n_train", 2000)),
        rpn_pre_nms_top_n_test=int(model_cfg.get("rpn_pre_nms_top_n_test", 1000)),
        rpn_post_nms_top_n_train=int(model_cfg.get("rpn_post_nms_top_n_train", 1000)),
        rpn_post_nms_top_n_test=int(model_cfg.get("rpn_post_nms_top_n_test", 300)),
    )
    if model_cfg.get("use_focal_loss", True):
        _install_focal_loss(model, model_cfg)
    model.logical_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return model
