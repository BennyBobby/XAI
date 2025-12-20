import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import requests
from ultralytics import YOLO
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

PREPROCESS_CLF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

feature_maps = {}
gradients = {}


def save_feature_maps_cls(module, input, output):
    feature_maps["A"] = output.detach()


def save_gradients_cls(module, grad_input, grad_output):
    gradients["grads"] = grad_output[0].detach()


class ActivationAndGradientExtractor:
    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


def _find_target_layer(model, target_layer_name):
    parts = target_layer_name.split(".")
    current_layer = model
    for part in parts:
        if "[" in part:
            name, idx_str = part.split("[")
            idx = int(idx_str[:-1])
            current_layer = getattr(current_layer, name)[idx]
        else:
            current_layer = getattr(current_layer, part)
    return current_layer


def load_resnet_model(target_layer_name="layer4[2].conv3"):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.to(DEVICE)
    model.eval()
    target_layer = _find_target_layer(model, target_layer_name)
    return model, target_layer


def load_yolo_model(model_name="yolov8n.pt", target_layer_index=9):
    model_yolo = YOLO(model_name)
    model_pt = model_yolo.model.to(DEVICE).eval()
    try:
        target_layer = model_pt.model[target_layer_index]
    except IndexError:
        print(
            f"Index {target_layer_index} invalide. Utilisation de la dernière couche."
        )
        target_layer = model_pt.model[-1]
    return model_yolo, model_pt, target_layer


def preprocess_yolo_input(image_np, size=640):
    img_resized = cv2.resize(image_np, (size, size))
    input_tensor = (
        torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).div(255.0)
    )
    return input_tensor.to(DEVICE)


def batch_predict(images, model):
    batch = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).float().to(DEVICE)
    batch /= 255.0
    if batch.shape[2:] != (224, 224):
        batch = torch.nn.functional.interpolate(
            batch, size=(224, 224), mode="bilinear", align_corners=False
        )
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    with torch.no_grad():
        output = model(batch)
        if isinstance(output, tuple):
            output = output[0]
    return torch.nn.functional.softmax(output, dim=1).cpu().numpy()


def get_imagenet_class_names():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        class_names = requests.get(url, timeout=5).text.split("\n")
        return class_names
    except:
        return [f"Class {i}" for i in range(1000)]
