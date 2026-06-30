"""Detection metrics with optional canonical pycocotools evaluation."""

import json
from pathlib import Path

import torch
from torchvision.ops import box_iou

from .data import unwrap_coco_dataset


def _average_precision(records, total_gt):
    if total_gt == 0 or not records:
        return float("nan") if total_gt == 0 else 0.0
    records.sort(key=lambda item: item[0], reverse=True)
    tp = torch.tensor([item[1] for item in records], dtype=torch.float64).cumsum(0)
    fp = torch.tensor([not item[1] for item in records], dtype=torch.float64).cumsum(0)
    recall = tp / total_gt
    precision = tp / (tp + fp).clamp(min=1)
    # COCO-style 101-point interpolated AP.
    return float(torch.stack([precision[recall >= level].max() if (recall >= level).any() else torch.tensor(0.0) for level in torch.linspace(0, 1, 101)]).mean())


def _ap_at_iou(predictions, targets, threshold, area_range=(0.0, float("inf"))):
    classes = sorted({int(label) for target in targets for label in target["labels"]})
    class_aps = []
    for label in classes:
        records, total_gt = [], 0
        for prediction, target in zip(predictions, targets):
            gt_mask = target["labels"] == label
            if "area" in target:
                gt_mask &= (target["area"] >= area_range[0]) & (target["area"] < area_range[1])
            gt_boxes = target["boxes"][gt_mask]
            total_gt += len(gt_boxes)
            pred_mask = prediction["labels"] == label
            boxes = prediction["boxes"][pred_mask]
            scores = prediction["scores"][pred_mask]
            order = scores.argsort(descending=True)
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
            ious = box_iou(boxes[order], gt_boxes) if len(gt_boxes) and len(boxes) else torch.empty((len(boxes), len(gt_boxes)))
            for row, score in enumerate(scores[order]):
                is_tp = False
                if len(gt_boxes):
                    best_iou, best_index = ious[row].max(0)
                    if best_iou >= threshold and not matched[best_index]:
                        matched[best_index] = True
                        is_tp = True
                records.append((float(score), is_tp))
        ap = _average_precision(records, total_gt)
        if not torch.isnan(torch.tensor(ap)):
            class_aps.append(ap)
    return sum(class_aps) / len(class_aps) if class_aps else float("nan")


def native_detection_metrics(predictions, targets, score_threshold=0.5):
    thresholds = [0.5 + 0.05 * index for index in range(10)]
    aps = [_ap_at_iou(predictions, targets, threshold) for threshold in thresholds]
    tp = fp = total_gt = 0
    matched_ious = []
    for prediction, target in zip(predictions, targets):
        keep = prediction["scores"] >= score_threshold
        pred_boxes, pred_labels = prediction["boxes"][keep], prediction["labels"][keep]
        matched = torch.zeros(len(target["boxes"]), dtype=torch.bool)
        total_gt += len(target["boxes"])
        ious = box_iou(pred_boxes, target["boxes"])
        for index, label in enumerate(pred_labels):
            candidates = (target["labels"] == label) & ~matched
            values = ious[index].clone()
            values[~candidates] = -1
            if len(values) and values.max() >= 0.5:
                best_index = values.argmax()
                matched[best_index] = True
                matched_ious.append(float(values[best_index]))
                tp += 1
            else:
                fp += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gt, 1)
    false_negatives = max(total_gt - tp, 0)
    detection_accuracy = tp / max(tp + fp + false_negatives, 1)
    mean_iou = sum(matched_ious) / max(len(matched_ious), 1)
    area_aps = {
        "ap_small": [_ap_at_iou(predictions, targets, threshold, (0, 32**2)) for threshold in thresholds],
        "ap_medium": [_ap_at_iou(predictions, targets, threshold, (32**2, 96**2)) for threshold in thresholds],
        "ap_large": [_ap_at_iou(predictions, targets, threshold, (96**2, float("inf"))) for threshold in thresholds],
    }
    def nanmean(values):
        valid = [value for value in values if not torch.isnan(torch.tensor(value))]
        return sum(valid) / len(valid) if valid else float("nan")

    return {
        "map_50_95": sum(aps) / len(aps),
        "map_50": aps[0],
        **{name: nanmean(values) for name, values in area_aps.items()},
        "precision": precision,
        "recall": recall,
        "accuracy": detection_accuracy,
        "mean_iou": mean_iou,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def _coco_metrics(predictions, targets, dataset):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None
    results = []
    for prediction, target in zip(predictions, targets):
        image_id = int(target["image_id"])
        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            x1, y1, x2, y2 = map(float, box)
            results.append({
                "image_id": image_id,
                "category_id": dataset.label_to_category_id[int(label)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    if not results:
        return None
    coco_gt = COCO(str(dataset.annotation_path))
    # Some SeaDronesSee COCO exports omit optional fields expected by
    # pycocotools. Normal images are not crowd regions; derive area from bbox
    # when it is absent instead of requiring a rewritten annotation file.
    for annotation in coco_gt.dataset.get("annotations", []):
        annotation.setdefault("iscrowd", 0)
        if "area" not in annotation:
            _, _, width, height = annotation["bbox"]
            annotation["area"] = max(float(width), 0.0) * max(float(height), 0.0)
    coco_gt.createIndex()
    evaluator = COCOeval(coco_gt, coco_gt.loadRes(results), "bbox")
    evaluator.params.imgIds = [int(target["image_id"]) for target in targets]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {"map_50_95": float(stats[0]), "map_50": float(stats[1]), "ap_small": float(stats[3]), "ap_medium": float(stats[4]), "ap_large": float(stats[5])}


@torch.inference_mode()
def rpn_recall(model, loader, device, limits=(100, 300, 1000)):
    hits = {limit: 0 for limit in limits}
    total = 0
    proposal_counts = []
    best_ious = []
    model.eval()
    for images, targets in loader:
        images = [image.to(device) for image in images]
        image_list, resized_targets = model.transform(images, [{key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets])
        features = model.backbone(image_list.tensors)
        proposals, _ = model.rpn(image_list, features, resized_targets)
        for proposal, target in zip(proposals, resized_targets):
            proposal_counts.append(len(proposal))
            gt = target["boxes"]
            total += len(gt)
            if not len(gt):
                continue
            if not len(proposal):
                best_ious.extend([0.0] * len(gt))
                continue
            ious = box_iou(gt, proposal)
            best_ious.extend(ious.max(dim=1).values.detach().cpu().tolist())
            for limit in limits:
                hits[limit] += int((ious[:, :limit].max(dim=1).values >= 0.5).sum()) if len(proposal) else 0
    result = {f"rpn_recall_{limit}": hits[limit] / max(total, 1) for limit in limits}
    result["average_proposals"] = sum(proposal_counts) / max(len(proposal_counts), 1)
    if best_ious:
        values = torch.tensor(best_ious)
        result.update({
            "proposal_iou_mean": float(values.mean()),
            "proposal_iou_median": float(values.median()),
            "proposal_iou_p75": float(torch.quantile(values, 0.75)),
        })
    else:
        result.update({"proposal_iou_mean": 0.0, "proposal_iou_median": 0.0, "proposal_iou_p75": 0.0})
    return result


@torch.inference_mode()
def evaluate_model(model, loader, device, include_rpn=True):
    model.eval()
    predictions, targets = [], []
    for images, batch_targets in loader:
        outputs = model([image.to(device) for image in images])
        predictions.extend([{key: value.detach().cpu() for key, value in output.items()} for output in outputs])
        targets.extend([{key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()} for target in batch_targets])
    metrics = native_detection_metrics(predictions, targets)
    dataset = unwrap_coco_dataset(loader.dataset)
    canonical = _coco_metrics(predictions, targets, dataset)
    if canonical:
        metrics.update(canonical)
    if include_rpn:
        metrics.update(rpn_recall(model, loader, device))
    return metrics


def save_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
