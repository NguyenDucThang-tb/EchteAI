"""Các hàm dựng backbone và Faster R-CNN dùng chung cho toàn pipeline."""

from .fasterrcnn_convnext import build_fasterrcnn_convnext

__all__ = ["build_fasterrcnn_convnext"]
