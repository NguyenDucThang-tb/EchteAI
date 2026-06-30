"""Faster R-CNN assembly with a configurable ConvNeXt-FPN backbone."""

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .convnext_fpn_backbone import build_convnext_fpn_backbone


def _anchors(config):
    sizes = tuple(int(value) for value in config.get("anchor_sizes", (16, 32, 64, 128, 256)))
    ratios = tuple(float(value) for value in config.get("aspect_ratios", (0.5, 1.0, 2.0)))
    if len(sizes) != 5:
        raise ValueError("anchor_sizes must contain five values for P2-P6")
    if not ratios or any(value <= 0 for value in ratios):
        raise ValueError("aspect_ratios must contain positive values")
    return AnchorGenerator(tuple((size,) for size in sizes), (ratios,) * len(sizes))


def build_fasterrcnn_convnext(config):
    """Build an unquantized model. Selective QAT is applied as a separate step."""
    model_cfg = config["model"]
    num_classes = int(config["dataset"]["num_classes"])
    if num_classes < 2:
        raise ValueError("dataset.num_classes must include background and at least one class")

    backbone = build_convnext_fpn_backbone(model_cfg)
    min_size = model_cfg.get("train_min_sizes", model_cfg.get("min_size", 640))
    if isinstance(min_size, list):
        min_size = tuple(int(value) for value in min_size)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=int(model_cfg.get("max_size", 1024)),
        rpn_anchor_generator=_anchors(model_cfg),
        box_roi_pool=roi_pooler,
        rpn_pre_nms_top_n_train=int(model_cfg.get("rpn_pre_nms_top_n_train", 2000)),
        rpn_pre_nms_top_n_test=int(model_cfg.get("rpn_pre_nms_top_n_test", 1000)),
        rpn_post_nms_top_n_train=int(model_cfg.get("rpn_post_nms_top_n_train", 1000)),
        rpn_post_nms_top_n_test=int(model_cfg.get("rpn_post_nms_top_n_test", 300)),
    )
    model.logical_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return model
