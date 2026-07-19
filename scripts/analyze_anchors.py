#!/usr/bin/env python3
"""Thống kê bbox COCO và in bộ anchor FPN phù hợp với dữ liệu thật."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.convnext_qat.anchors import infer_anchor_statistics
from pipelines.convnext_qat.config import load_config


def main():
    """Đọc annotation/config rồi xuất kích thước anchor theo percentile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seadronessee_colab.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    model = config["model"]
    result = infer_anchor_statistics(
        config["dataset"]["train_annotations"],
        target_min_size=model.get("anchor_statistics_min_size", model.get("min_size", 960)),
        max_size=model.get("max_size", 1600),
        ignore_category_ids=config["dataset"].get("ignore_category_ids", []),
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    print(f"Recommended YAML: anchor_sizes: {result['anchor_sizes']}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
