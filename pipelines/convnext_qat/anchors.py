"""Thống kê bounding box để sinh anchor phù hợp cho detector ConvNeXt-FPN."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def _kmeans_1d(values: torch.Tensor, clusters: int, iterations: int = 100) -> torch.Tensor:
    """Gom cụm kích thước hộp trong log-space để giảm ảnh hưởng của hộp quá lớn."""
    if values.numel() < clusters:
        raise ValueError(f"Need at least {clusters} valid boxes, found {values.numel()}")
    values = values.log()
    quantiles = torch.linspace(0.05, 0.95, clusters, dtype=values.dtype)
    centers = torch.quantile(values, quantiles)
    for _ in range(iterations):
        assignments = (values[:, None] - centers[None, :]).abs().argmin(dim=1)
        updated = torch.stack([
            values[assignments == index].mean() if (assignments == index).any() else centers[index]
            for index in range(clusters)
        ])
        if torch.allclose(updated, centers, atol=1e-5, rtol=0.0):
            centers = updated
            break
        centers = updated
    return centers.exp().sort().values


def infer_anchor_statistics(annotation_path, target_min_size=960, max_size=1600, levels=5,
                            ignore_category_ids=()):
    """Suy ra scale/tỉ lệ anchor sau khi mô phỏng đúng phép resize của detector.

    Kích thước đại diện của một hộp là ``sqrt(width * height)``. Giá trị này
    được nhân với hệ số resize ảnh trước khi chạy k-means thành năm mức P2-P6.
    """
    annotation_path = Path(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    image_sizes = {
        int(image["id"]): (float(image["width"]), float(image["height"]))
        for image in data.get("images", [])
    }
    ignored = {int(value) for value in ignore_category_ids}
    scales, ratios = [], []
    for annotation in data.get("annotations", []):
        if annotation.get("iscrowd", 0) or int(annotation["category_id"]) in ignored:
            continue
        width, height = map(float, annotation["bbox"][2:])
        image_size = image_sizes.get(int(annotation["image_id"]))
        if width <= 0 or height <= 0 or image_size is None:
            continue
        image_width, image_height = image_size
        resize = min(
            float(target_min_size) / min(image_width, image_height),
            float(max_size) / max(image_width, image_height),
        )
        scales.append(math.sqrt(width * height) * resize)
        ratios.append(width / height)
    scale_tensor = torch.tensor(scales, dtype=torch.float64)
    ratio_tensor = torch.tensor(ratios, dtype=torch.float64)
    centers = _kmeans_1d(scale_tensor, levels)
    # AnchorGenerator expects integer side lengths. Keep measured values rather
    # than snapping to powers of two, while enforcing strictly increasing sizes.
    anchor_sizes = []
    for center in centers:
        value = max(2, int(round(float(center))))
        if anchor_sizes:
            value = max(value, anchor_sizes[-1] + 1)
        anchor_sizes.append(value)
    return {
        "boxes": len(scales),
        "anchor_sizes": anchor_sizes,
        "scale_percentiles": {
            str(percentile): float(torch.quantile(scale_tensor, percentile / 100.0))
            for percentile in (10, 25, 50, 75, 90)
        },
        "aspect_ratio_percentiles": {
            str(percentile): float(torch.quantile(ratio_tensor, percentile / 100.0))
            for percentile in (10, 25, 50, 75, 90)
        },
    }


def resolve_anchor_sizes(config):
    """Đọc anchor cố định hoặc tự tính anchor từ annotation train khi đặt ``auto``."""
    model_cfg = config["model"]
    configured = model_cfg.get("anchor_sizes", (16, 32, 64, 128, 256))
    if configured != "auto":
        return tuple(int(value) for value in configured)
    statistics = infer_anchor_statistics(
        config["dataset"]["train_annotations"],
        target_min_size=model_cfg.get("anchor_statistics_min_size", model_cfg.get("min_size", 960)),
        max_size=model_cfg.get("max_size", 1600),
        levels=5,
        ignore_category_ids=config["dataset"].get("ignore_category_ids", []),
    )
    sizes = tuple(statistics["anchor_sizes"])
    print(f"bbox-driven anchors={list(sizes)} boxes={statistics['boxes']}", flush=True)
    return sizes
