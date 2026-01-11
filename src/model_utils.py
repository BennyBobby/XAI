import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import requests
from ultralytics import YOLO
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Prétraitement ---
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


# --- Gestionnaire de Hooks (Universal) ---
class ActivationAndGradientExtractor:
    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        # Utilisation de full_backward_hook pour éviter les comportements instables
        self._backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # On prend le premier élément du tuple de gradients de sortie
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


# --- Helpers de Chargement ---
def load_yolo_model(
    model_name="yolov8n.pt", target_layer_index=15
):  # Index 15 est souvent plus riche en features
    model_yolo = YOLO(model_name)
    model_pt = model_yolo.model.to(DEVICE).eval()

    # Accès sécurisé à la structure interne de YOLOv8
    try:
        # Dans YOLOv8, les couches sont dans model_pt.model
        target_layer = model_pt.model[target_layer_index]
    except (IndexError, AttributeError):
        print(
            f"Index {target_layer_index} non trouvé, fallback sur l'avant-dernière couche."
        )
        target_layer = model_pt.model[-2]

    return model_yolo, model_pt, target_layer


def batch_predict(images, model, is_yolo=False):
    """
    Fonction de prédiction pour LIME.
    """
    batch = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).float().to(DEVICE)
    batch /= 255.0

    if batch.shape[2:] != (224, 224):
        batch = torch.nn.functional.interpolate(batch, size=(224, 224), mode="bilinear")

    # On ne normalise ImageNet QUE pour les modèles de classification type ResNet
    if not is_yolo:
        batch = (batch - IMAGENET_MEAN) / IMAGENET_STD

    with torch.no_grad():
        output = model(batch)
        # YOLO renvoie souvent une liste/tuple en mode training/eval
        if isinstance(output, (list, tuple)):
            output = output[0]

    return torch.nn.functional.softmax(output, dim=1).cpu().numpy()


def preprocess_yolo_input(image_np, size=640):
    """
    Prépare l'image pour YOLO : redimensionnement et conversion en tenseur.
    """
    img_resized = cv2.resize(image_np, (size, size))
    # Conversion HWC -> CHW et normalisation 0-1
    input_tensor = (
        torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).div(255.0)
    )
    return input_tensor.to(DEVICE)
