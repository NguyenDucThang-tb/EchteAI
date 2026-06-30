"""COCO detection dataset with contiguous training labels."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import pil_to_tensor


class CocoDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_path, training=False, augmentation=None):
        self.image_dir = Path(image_dir)
        self.annotation_path = Path(annotation_path)
        self.training = training
        augmentation = augmentation or {}
        self.ignore_category_ids = {
            int(category_id) for category_id in augmentation.get("ignore_category_ids", [])
        }
        self.horizontal_flip_probability = float(augmentation.get("horizontal_flip_probability", 0.0))
        self.color_jitter = ColorJitter(
            brightness=float(augmentation.get("brightness", 0.0)),
            contrast=float(augmentation.get("contrast", 0.0)),
            saturation=float(augmentation.get("saturation", 0.0)),
            hue=float(augmentation.get("hue", 0.0)),
        )
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"COCO image directory not found: {self.image_dir}")
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"COCO annotation file not found: {self.annotation_path}")
        with self.annotation_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.images = sorted(data.get("images", []), key=lambda item: item["id"])
        categories = sorted(
            (
                category for category in data.get("categories", [])
                if category["id"] not in self.ignore_category_ids
                and category.get("name", "").lower() != "ignored"
            ),
            key=lambda item: item["id"],
        )
        self.category_id_to_label = {category["id"]: i + 1 for i, category in enumerate(categories)}
        self.label_to_category_id = {label: category for category, label in self.category_id_to_label.items()}
        self.label_to_name = {self.category_id_to_label[c["id"]]: c["name"] for c in categories}
        self.annotations = defaultdict(list)
        for annotation in data.get("annotations", []):
            if (
                not annotation.get("iscrowd", 0)
                and annotation["category_id"] in self.category_id_to_label
            ):
                self.annotations[annotation["image_id"]].append(annotation)
        if not self.images:
            raise ValueError(f"No images found in {self.annotation_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        image_path = self.image_dir / info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image not found: {image_path}")
        with Image.open(image_path) as source:
            image_pil = source.convert("RGB")
            if self.training:
                image_pil = self.color_jitter(image_pil)
            image = pil_to_tensor(image_pil).float().div_(255.0)

        boxes, labels, areas, crowds = [], [], [], []
        width, height = image.shape[-1], image.shape[-2]
        for annotation in self.annotations[info["id"]]:
            x, y, w, h = annotation["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_label[annotation["category_id"]])
            areas.append(float(annotation.get("area", w * h)))
            crowds.append(int(annotation.get("iscrowd", 0)))
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.training and torch.rand(()) < self.horizontal_flip_probability:
            image = image.flip(-1)
            old_x1 = boxes_tensor[:, 0].clone()
            old_x2 = boxes_tensor[:, 2].clone()
            boxes_tensor[:, 0] = width - old_x2
            boxes_tensor[:, 2] = width - old_x1
        target = {
            "boxes": boxes_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(info["id"], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
        }
        return image, target


def detection_collate(batch):
    return tuple(zip(*batch))


def build_coco_loader(config, split, shuffle=None, limit=None):
    dataset_cfg = config["dataset"]
    dataset = CocoDetectionDataset(
        dataset_cfg[f"{split}_images"],
        dataset_cfg[f"{split}_annotations"],
        training=split == "train",
        augmentation={
            **(config.get("augmentation", {}) if split == "train" else {}),
            "ignore_category_ids": dataset_cfg.get("ignore_category_ids", []),
        },
    )
    if int(dataset_cfg["num_classes"]) != len(dataset.category_id_to_label) + 1:
        raise ValueError(
            f"dataset.num_classes={dataset_cfg['num_classes']} but {split} annotations "
            f"contain {len(dataset.category_id_to_label)} foreground categories"
        )
    if limit is not None:
        dataset = Subset(dataset, range(min(int(limit), len(dataset))))
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(dataset_cfg.get("workers", 4)),
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
    )


def unwrap_coco_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset
