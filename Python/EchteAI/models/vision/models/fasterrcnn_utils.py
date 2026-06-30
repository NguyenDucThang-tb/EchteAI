import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
import logging
import os
import torch
import numpy as np
import cv2
import time
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.detection.image_list import ImageList


def build_weights_only_path(model_path):
    model_path = os.fspath(model_path)
    base, ext = os.path.splitext(model_path)
    if not ext:
        ext = ".pth"
    return f"{base}_weights{ext}"


def build_epoch_checkpoint_path(model_path, epoch_number):
    model_path = os.fspath(model_path)
    base, ext = os.path.splitext(model_path)
    if not ext:
        ext = ".pth"
    return f"{base}_epoch_{epoch_number}{ext}"

def setup_fasterrcnn(dataset=None, backbone="resnet50"):
    model_choices = {
        "resnet50": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
        "mobilenet_v3": (fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    }

    if backbone not in model_choices:
        raise ValueError(f"Unknown backbone: {backbone}. Use a valid one: {list(model_choices.keys())}")

    logging.info(f"Loading {backbone}...")

    model_fn, weights = model_choices[backbone]
    model = model_fn(weights=weights)

    if dataset is not None:
        num_classes = len(dataset.dataset.class_to_idx) + 1 if hasattr(dataset, "dataset") else len(dataset.class_to_idx) + 1
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    logging.info(f"The {backbone} (faster-r-cnn model) is ready.")

    return model

def compute_iou_fasterrcnn(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def compute_metrics_fasterrcnn(data_loader, model, device, iou_threshold=0.5, score_threshold=0.5):
    was_training = model.training
    model.eval()
    total_gt = 0
    total_tp = 0
    total_pred = 0
    iou_list = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx > 0:
                break
            images = [img.to(device) for img in images]
            predictions = model(images)
            for target, prediction in zip(targets, predictions):
                if "boxes" not in target:
                    continue
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                total_gt += len(gt_boxes)
                pred_boxes = prediction["boxes"].cpu().numpy()
                pred_labels = prediction["labels"].cpu().numpy()
                pred_scores = prediction["scores"].cpu().numpy()
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                total_pred += len(pred_boxes)
                matched = [False] * len(gt_boxes)
                for pb, pl in zip(pred_boxes, pred_labels):
                    best_iou = 0
                    best_idx = -1
                    for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if matched[i]:
                            continue
                        if pl != gl:
                            continue
                        iou = compute_iou_fasterrcnn(pb, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i
                    if best_iou >= iou_threshold and best_idx != -1:
                        matched[best_idx] = True
                        total_tp += 1
                        iou_list.append(best_iou)
    if was_training:
        model.train()
    accuracy = total_tp / total_gt if total_gt > 0 else 0
    precision = total_tp / total_pred if total_pred > 0 else 0
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    return {"accuracy": accuracy, "precision": precision, "mean_iou": mean_iou}

def compute_batch_metrics_fasterrcnn(targets, predictions, iou_threshold=0.5, score_threshold=0.5):
    total_gt = 0
    total_tp = 0
    total_pred = 0
    iou_list = []
    for target, prediction in zip(targets, predictions):
        if "boxes" not in target:
            continue
        gt_boxes = target["boxes"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()
        total_gt += len(gt_boxes)
        pred_boxes = prediction["boxes"].cpu().numpy()
        pred_labels = prediction["labels"].cpu().numpy()
        pred_scores = prediction["scores"].cpu().numpy()
        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        total_pred += len(pred_boxes)
        matched = [False] * len(gt_boxes)
        for pb, pl in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_idx = -1
            for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if matched[i]:
                    continue
                if pl != gl:
                    continue
                iou = compute_iou_fasterrcnn(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_threshold and best_idx != -1:
                matched[best_idx] = True
                total_tp += 1
                iou_list.append(best_iou)
    accuracy = total_tp / total_gt if total_gt > 0 else 0
    precision = total_tp / total_pred if total_pred > 0 else 0
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    return {"accuracy": accuracy, "precision": precision, "mean_iou": mean_iou}

def train_fasterrcnn(model, train_loader, val_loader, device, num_epochs, model_path="model.pth", max_train_batches=None):
    if not model_path or not str(model_path).strip():
        raise ValueError("model_path is empty. Provide a valid checkpoint path, for example ./model_fasterrcnn_resnet50_kitti.pth")
    model_path = os.fspath(model_path)
    weights_only_path = build_weights_only_path(model_path)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    start_epoch = 0

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            logging.info(f"Resuming training from epoch {start_epoch + 1} using checkpoint {model_path}.")
        else:
            model.load_state_dict(checkpoint)
            logging.info(f"Loaded saved model weights from {model_path}. Starting training from epoch 1.")

    if start_epoch >= num_epochs:
        logging.info("Checkpoint already covers %d epochs. Skipping training.", start_epoch)
        return model

    if start_epoch < num_epochs:
        logging.info("Training started.")
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0
            processed_batches = 0
            for batch_idx, (images, targets) in enumerate(train_loader):
                if max_train_batches is not None and batch_idx >= max_train_batches:
                    break
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                running_loss += losses.item()
                processed_batches += 1
                with torch.no_grad():
                    model.eval()
                    predictions = model(images)
                    batch_metrics = compute_batch_metrics_fasterrcnn(targets, predictions)
                    model.train()
                if (batch_idx + 1) % 100 == 0:
                    logging.info(
                        f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {losses.item():.4f}, "
                        f"Acc: {batch_metrics['accuracy']:.4f}, Prec: {batch_metrics['precision']:.4f}, "
                        f"mIoU: {batch_metrics['mean_iou']:.4f}"
                    )
            if processed_batches == 0:
                raise ValueError("No training batches were processed. Check max_train_batches and dataloader size.")
            avg_loss = running_loss / processed_batches
            train_metrics = compute_metrics_fasterrcnn(train_loader, model, device)
            val_metrics = compute_metrics_fasterrcnn(val_loader, model, device)
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} finished, avg loss: {avg_loss:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, Train Prec: {train_metrics['precision']:.4f}, "
                f"Train mIoU: {train_metrics['mean_iou']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val Prec: {val_metrics['precision']:.4f}, Val mIoU: {val_metrics['mean_iou']:.4f}"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                model_path,
            )
            torch.save(model.state_dict(), weights_only_path)
            epoch_checkpoint_path = build_epoch_checkpoint_path(model_path, epoch + 1)
            epoch_weights_only_path = build_weights_only_path(epoch_checkpoint_path)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                epoch_checkpoint_path,
            )
            torch.save(model.state_dict(), epoch_weights_only_path)
            logging.info(
                "Saved epoch %d latest checkpoint to %s, latest weights-only file to %s, epoch checkpoint to %s, and epoch weights-only file to %s",
                epoch + 1,
                model_path,
                weights_only_path,
                epoch_checkpoint_path,
                epoch_weights_only_path,
            )
        logging.info("Training finished.")
    return model

def run_predictions_fasterrcnn(model, data_loader, device, dataset, output_folder, evaluate=False, num_batches = -1, batch_size=None, score_threshold=0.5):
    os.makedirs(output_folder, exist_ok=True)
    #model.to(device)
    #model.eval()
    if batch_size is not None:
        idx_to_class = dataset.idx_to_class
        class_to_idx = dataset.class_to_idx
        logging.info(f"Rebuilding data_loader with batch_size={batch_size}")
        dataset = data_loader.dataset
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=getattr(data_loader, 'collate_fn', None)
        )
        dataset.idx_to_class = idx_to_class
        dataset.class_to_idx = class_to_idx

    if evaluate:
        metrics = compute_metrics_fasterrcnn(data_loader, model, device)
        logging.info(f"Metrics on dataset: Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, mIoU: {metrics['mean_iou']:.4f}")
    with torch.no_grad():
        print("sssadsa")
        for batch_idx, (images, targets) in enumerate(data_loader):
            print(f"iteration: {batch_idx}")
            if num_batches > 0 and batch_idx > num_batches - 1:
                break
            images = [img.to(device) for img in images]
            start_time = time.time()
            predictions = model(images)
            batch_time = time.time() - start_time
            for i, (img, prediction) in enumerate(zip(images, predictions)):
                image_np = img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
                image_np = np.ascontiguousarray(image_np)
                if image_np.dtype != np.uint8:
                    image_np = image_np.astype(np.uint8)
                
                # Ground Truth -> Red
                if "boxes" in targets[i]:
                    for box in targets[i]["boxes"]:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Predictions -> Green
                for j, box in enumerate(prediction["boxes"]):
                    score = prediction["scores"][j].item()
                    if score < score_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label_int = prediction["labels"][j].item()
                    label_name = dataset.idx_to_class.get(label_int, "Unknown")
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_np, f"{label_name}: {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                output_path = os.path.join(output_folder, f"batch{batch_idx}_img{i}.png")
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, image_bgr)
            logging.info(f"Batch {batch_idx} processed in {batch_time:.4f} seconds.")

import torch.nn.functional as F

def resize_and_pad(img, target_size):
        c, h, w = img.shape
        scale = min(target_size[0]/h, target_size[1]/w)
        new_h, new_w = int(h*scale), int(w*scale)
        img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        pad_h = target_size[0] - new_h
        pad_w = target_size[1] - new_w
        img_padded = F.pad(img_resized, (0, pad_w, 0, pad_h))
        return img_padded

def run_predictions_efficientdet(model, data_loader, device, dataset, output_folder, score_threshold=0.5, num_batches=-1, target_size=(512,512)):
    """
    EfficientDet predikciók futtatása és mentése képként.
    Ground-truth boxok nem jelennek meg.
    """
    os.makedirs(output_folder, exist_ok=True)
    model.to(device).eval()

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if num_batches > 0 and batch_idx >= num_batches:
                break

            images_tensor = torch.stack([resize_and_pad(img.to(device), target_size) for img in images])

            start_time = time.time()
            predictions = model(images_tensor)
            batch_time = time.time() - start_time

            for i, prediction in enumerate(predictions):
                image_np = images_tensor[i].mul(255).byte().permute(1,2,0).cpu().numpy()
                image_np = np.ascontiguousarray(image_np)

                # Predikció -> zöld boxok
                if isinstance(prediction, torch.Tensor):
                    if prediction.numel() == 0:
                        continue
                    boxes = prediction[:, :4]
                    scores = prediction[:, 4]
                    labels = prediction[:, 5].long() if prediction.shape[1] > 5 else torch.zeros_like(scores, dtype=torch.int64)
                elif isinstance(prediction, dict):
                    boxes = prediction.get("boxes", [])
                    scores = prediction.get("scores", [])
                    labels = prediction.get("labels", [])
                else:
                    raise ValueError(f"Unknown prediction format: {type(prediction)}")

                for j in range(len(boxes)):
                    box = boxes[j].cpu().numpy() if isinstance(boxes[j], torch.Tensor) else np.array(boxes[j])
                    if len(box) != 4:
                        continue
                    x1, y1, x2, y2 = box.astype(int)
                    score = float(scores[j].item()) if isinstance(scores[j], torch.Tensor) else float(scores[j])
                    if score < score_threshold:
                        continue
                    label_int = int(labels[j].item()) if isinstance(labels[j], torch.Tensor) else int(labels[j])
                    label_name = dataset.idx_to_class.get(label_int, "Unknown")

                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(image_np, f"{label_name}: {score:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                output_path = os.path.join(output_folder, f"batch{batch_idx}_img{i}.png")
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, image_bgr)

            logging.info(f"Batch {batch_idx} processed in {batch_time:.4f} seconds.")
