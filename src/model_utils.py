import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import requests
from ultralytics import YOLO
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to: {DEVICE}")


def get_imagenet_class_names():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        class_names = requests.get(url).text.split("\n")
        return class_names
    except requests.exceptions.RequestException:
        print("Avertissement: Impossible de télécharger la liste des classes ImageNet.")
        return [f"Class {i}" for i in range(1000)]


IMAGENET_CLASSES = get_imagenet_class_names()

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
        self._backward_handle = target_layer.register_backward_hook(self._save_gradient)

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
            current_layer = current_layer._modules[name][idx]
        else:
            current_layer = current_layer._modules[part]
    return current_layer


def load_resnet_model(target_layer_name="layer4[-1].conv3"):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.to(DEVICE)
    model.eval()
    target_layer = _find_target_layer(model, target_layer_name)
    return model, target_layer


def load_yolo_model(model_name="yolov8n.pt", target_layer_index=9):
    model_yolo = YOLO(model_name).to(DEVICE)
    model_pt = model_yolo.model.eval()
    try:
        target_layer = model_pt.model[target_layer_index]
    except IndexError:
        print(
            f"Avertissement: Impossible de trouver la couche cible à l'indice {target_layer_index} dans le modèle YOLO."
        )
        target_layer = None

    return model_yolo, model_pt, target_layer


PREPROCESS_CLF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_yolo_input(image_np, size=640):
    img_resized = cv2.resize(image_np, (size, size))
    rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    input_tensor = torch.from_numpy(rgb_img).float()
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).div(255.0)

    return input_tensor.to(DEVICE)


def batch_predict(images, model):
    tensor_batch = torch.stack(
        [
            transforms.ToTensor()(Image.fromarray(image)).to(torch.float32)
            for image in images
        ]
    ).to(DEVICE)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

    normalized_batch = (tensor_batch - mean) / std

    with torch.no_grad():
        logits = model(normalized_batch)

    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    return probs
