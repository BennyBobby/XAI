import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from lime import lime_image
from .model_utils import (
    DEVICE,
    ActivationAndGradientExtractor,
    batch_predict,
    preprocess_yolo_input,
)


def superimpose_heatmap(
    original_img_np, heatmap_np, alpha=0.5, colormap=cv2.COLORMAP_JET
):
    h, w = original_img_np.shape[:2]
    heatmap = cv2.resize(heatmap_np, (w, h))
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * original_img_np + alpha * heatmap_rgb)


# --- CLASSIFICATION (GRAD-CAM) ---
def generate_gradcam_mask_clf(model, input_tensor, target_layer):
    extractor = ActivationAndGradientExtractor(target_layer)
    input_tensor.requires_grad_(True)
    output = model(input_tensor)

    idx = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, idx].backward()

    # Calcul des poids (moyenne globale des gradients)
    weights = torch.mean(extractor.gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * extractor.activations, dim=1).squeeze()
    cam_np = F.relu(cam).detach().cpu().numpy()

    extractor.remove_hooks()
    if cam_np.max() > 0:
        cam_np /= cam_np.max()
    return cam_np, idx


# --- CLASSIFICATION (GRAD-CAM++) ---
def generate_gradcam_pp_mask_clf(model, input_tensor, target_layer):
    extractor = ActivationAndGradientExtractor(target_layer)
    input_tensor.requires_grad_(True)
    output = model(input_tensor)

    idx = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, idx].backward(retain_graph=True)

    A = extractor.activations
    grads = extractor.gradients

    # Calcul alpha_ij_kc
    grads_2 = grads.pow(2)
    grads_3 = grads.pow(3)
    alpha_denom = 2 * grads_2 + (A * grads_3).sum(dim=(2, 3), keepdim=True)
    alpha_denom = torch.where(
        alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom)
    )
    alphas = grads_2 / alpha_denom

    weights = torch.sum(alphas * torch.relu(grads), dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * A, dim=1).squeeze()
    cam_np = F.relu(cam).detach().cpu().numpy()

    extractor.remove_hooks()
    if cam_np.max() > 0:
        cam_np /= cam_np.max()
    return cam_np, idx


# --- DÉTECTION (GRAD-CAM) ---
def generate_yolo_gradcam(
    model_yolo, model_pt, input_image_bgr, target_layer, detection_size=640
):
    input_tensor = preprocess_yolo_input(input_image_bgr, size=detection_size)
    results = model_yolo(input_image_bgr, verbose=False, imgsz=detection_size)

    if len(results[0].boxes) == 0:
        return np.zeros((detection_size, detection_size)), None, None, None

    box_data = results[0].boxes[0]
    target_class_index = int(box_data.cls.cpu().item())
    extractor = ActivationAndGradientExtractor(target_layer)

    input_tensor.requires_grad_(True)
    output = model_pt(input_tensor)

    scores_for_class = output[0][0, 4 + target_class_index, :]
    target_score_tensor = torch.max(scores_for_class)

    model_pt.zero_grad()
    target_score_tensor.backward(retain_graph=True)

    weights = torch.mean(extractor.gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * extractor.activations, dim=1).squeeze()
    cam_np = F.relu(cam).detach().cpu().numpy()

    extractor.remove_hooks()
    if cam_np.max() > 0:
        cam_np /= cam_np.max()

    return (
        cam_np,
        target_class_index,
        box_data.xyxy[0].cpu().numpy().astype(int),
        results[0].names,
    )


# --- DÉTECTION (GRAD-CAM++) ---
def generate_yolo_gradcam_pp(
    model_yolo, model_pt, input_image_bgr, target_layer, detection_size=640
):
    input_tensor = preprocess_yolo_input(input_image_bgr, size=detection_size)
    results = model_yolo(input_image_bgr, verbose=False, imgsz=detection_size)

    if len(results[0].boxes) == 0:
        return np.zeros((detection_size, detection_size)), None, None, None

    box_data = results[0].boxes[0]
    target_class_index = int(box_data.cls.cpu().item())
    extractor = ActivationAndGradientExtractor(target_layer)

    input_tensor.requires_grad_(True)
    output = model_pt(input_tensor)

    scores_for_class = output[0][0, 4 + target_class_index, :]
    target_score_tensor = torch.max(scores_for_class)

    model_pt.zero_grad()
    target_score_tensor.backward(retain_graph=True)

    A = extractor.activations
    grads = extractor.gradients

    grads_2 = grads.pow(2)
    grads_3 = grads.pow(3)
    sum_A = A.sum(dim=(2, 3), keepdim=True)

    eps = 1e-8
    alpha_denom = 2 * grads_2 + sum_A * grads_3 + eps
    alphas = grads_2 / alpha_denom
    weights = torch.sum(alphas * torch.relu(grads), dim=(2, 3), keepdim=True)

    cam = torch.sum(weights * A, dim=1).squeeze()
    cam_np = F.relu(cam).detach().cpu().numpy()

    extractor.remove_hooks()
    if cam_np.max() > 0:
        cam_np /= cam_np.max()

    return (
        cam_np,
        target_class_index,
        box_data.xyxy[0].cpu().numpy().astype(int),
        results[0].names,
    )


# --- LIME ---
def generate_lime_mask_clf(model, img_numpy, num_samples=1000):
    explainer = lime_image.LimeImageExplainer()
    # On précise is_yolo=False ici (par défaut dans batch_predict)
    temp_output = batch_predict(img_numpy[np.newaxis, ...], model)
    predicted_class_idx = np.argmax(temp_output[0])

    explanation = explainer.explain_instance(
        img_numpy,
        lambda x: batch_predict(x, model),
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )

    _, mask = explanation.get_image_and_mask(
        predicted_class_idx, positive_only=True, num_features=7, hide_rest=False
    )
    return mask.astype(np.float32), predicted_class_idx
