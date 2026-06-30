from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import os
import numpy as np
import torch
import logging
import onnxruntime as ort

def setup_yolo(model_name="yolo11x.pt", pretrained=True):
    if os.path.exists("./self_"+model_name):
        model = YOLO("./self_"+model_name)
    else:
        model = YOLO(model_name)
        if not pretrained:
            model = model.reset()
    return model

def train_yolo(model, data_yaml_path, device, epochs=10, model_name="yolo11x.pt"):
    if os.path.exists("./self_"+model_name):
        return model
    model.train(data=data_yaml_path, epochs=epochs, device=device, weight_decay=0.001)
    model.save("self_"+model_name)
    return model

def compute_metrics_yolo(model, data_yaml_path, device):
    metrics = model.val(data=data_yaml_path, device=device)
    return {
        "precision": float(metrics.results_dict["metrics/precision(B)"]),
        "recall": float(metrics.results_dict["metrics/recall(B)"]),
        "mAP50": float(metrics.results_dict["metrics/mAP50(B)"]),
        "mAP50-95": float(metrics.results_dict["metrics/mAP50-95(B)"])
    }

def run_predictions_yolo(model, image_folder="downloads/yolo_dataset/images/val", output_folder="outputs/yolo", batch_size=1, num_images=40):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")])[:num_images]

    for img_path in image_paths:
        model.predict(img_path, save=True, save_txt=True, project=output_folder, name="predict", batch=batch_size)


def predict_yolo_onnx_tensor(tensor: torch.Tensor = torch.rand(2, 3, 640, 640),
                              model_path: str = "self_yolo11x.onnx"):
    input_np = tensor.detach().cpu().numpy()
    _, _, h, w = tensor.shape
    transform = LetterBox(new_shape=(h, w))
    processed = []
    for i in range(input_np.shape[0]):
        img_np = input_np[i].transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        resized = transform(image=img_np)
        resized = resized.astype(np.float32) / 255.0
        resized = resized.transpose(2, 0, 1)
        processed.append(resized)
    input_data = np.stack(processed, axis=0).astype(np.float32)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    for i, out in enumerate(outputs):
        logging.info(f"Output[{i}] shape: {out.shape}")
    return outputs

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ImagePathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0
