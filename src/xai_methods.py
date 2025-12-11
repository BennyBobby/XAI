import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from lime import lime_image
from PIL import Image

from .model_utils import (
    DEVICE,
    ActivationAndGradientExtractor,
    feature_maps,
    gradients,
    batch_predict,
    save_feature_maps_cls,
    save_gradients_cls,
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

    original_img_float = original_img_np.astype(np.float32)
    heatmap_rgb_float = heatmap_rgb.astype(np.float32)

    superimposed_img = np.uint8(
        (1 - alpha) * original_img_float + alpha * heatmap_rgb_float
    )

    return superimposed_img


def generate_gradcam_mask_clf(model, input_tensor, target_layer):
    forward_handle = target_layer.register_forward_hook(save_feature_maps_cls)
    backward_handle = target_layer.register_full_backward_hook(save_gradients_cls)

    input_tensor.requires_grad_(True)
    output = model(input_tensor)

    predicted_class_idx = output.argmax(dim=1).item()
    target_logit = output[0, predicted_class_idx]

    model.zero_grad()
    target_logit.backward()

    A = feature_maps["A"]
    grads = gradients["grads"]

    forward_handle.remove()
    backward_handle.remove()

    alpha_c_k = torch.mean(grads.cpu(), dim=(2, 3), keepdim=True)
    weighted_fmaps = alpha_c_k.to(DEVICE) * A
    L_Grad_CAM_c = torch.sum(weighted_fmaps, dim=1, keepdim=True)

    grad_cam_map = nn.ReLU()(L_Grad_CAM_c)
    grad_cam_map_np = grad_cam_map.squeeze().detach().cpu().numpy()

    if np.max(grad_cam_map_np) > 0:
        grad_cam_map_np = grad_cam_map_np / np.max(grad_cam_map_np)

    return grad_cam_map_np, predicted_class_idx


def generate_gradcam_pp_mask_clf(model, input_tensor, target_layer):
    forward_handle = target_layer.register_forward_hook(save_feature_maps_cls)
    backward_handle = target_layer.register_full_backward_hook(save_gradients_cls)

    input_tensor.requires_grad_(True)
    output = model(input_tensor)

    target_class_idx = output.argmax(dim=1).item()
    target_logit = output[0, target_class_idx]

    model.zero_grad()
    target_logit.backward(retain_graph=True)

    A = feature_maps["A"].squeeze(0)
    grads = gradients["grads"].squeeze(0)

    forward_handle.remove()
    backward_handle.remove()

    g_1 = grads.clamp(min=0)
    numerator = g_1.pow(2)
    A_sum = A.sum(dim=(1, 2), keepdim=True)
    denominator = 2 * g_1.pow(2) + A_sum * g_1.pow(3)

    denominator = torch.where(
        denominator != 0.0, denominator, torch.ones_like(denominator).to(DEVICE)
    )

    alpha_ij_kc = numerator / denominator
    W_k_c = (alpha_ij_kc * g_1).sum(dim=(1, 2), keepdim=True)

    cam = (W_k_c * A).sum(dim=0)
    cam_numpy = torch.relu(cam).detach().cpu().numpy()

    if np.max(cam_numpy) > 0:
        cam_numpy = cam_numpy / np.max(cam_numpy)

    return cam_numpy, target_class_idx


def generate_lime_mask_clf(model, img_numpy, num_samples=10000):
    explainer = lime_image.LimeImageExplainer()

    temp_output = batch_predict(img_numpy[np.newaxis, ...], model)
    predicted_class_idx = np.argmax(temp_output[0])

    explanation = explainer.explain_instance(
        img_numpy,
        lambda x: batch_predict(x, model),
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )

    temp, mask = explanation.get_image_and_mask(
        predicted_class_idx, positive_only=True, num_features=7, hide_rest=False
    )

    lime_heatmap_np = np.zeros_like(mask, dtype=np.float32)
    lime_heatmap_np[mask != 0] = 1.0

    return lime_heatmap_np, predicted_class_idx


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

    try:
        target_score_tensor = output[0][0, 0, 4]
    except IndexError:
        extractor.remove_hooks()
        print("Erreur d'indexation du score cible YOLO.")
        return np.zeros((detection_size, detection_size)), None, None, None

    model_pt.zero_grad()
    target_score_tensor.backward(retain_graph=True)

    weights = torch.mean(extractor.gradients, dim=(2, 3), keepdim=True)
    weighted_activations = weights * extractor.activations
    cam = torch.sum(weighted_activations, dim=1).squeeze()
    cam = F.relu(cam)

    extractor.remove_hooks()
    cam_numpy = cam.cpu().numpy()

    if cam_numpy.max() > 0:
        cam_numpy = cam_numpy / cam_numpy.max()

    return (
        cam_numpy,
        target_class_index,
        box_data.xyxy[0].cpu().numpy().astype(int),
        results[0].names,
    )


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

    try:
        target_score_tensor = output[0][0, 0, 4]
    except IndexError:
        extractor.remove_hooks()
        return np.zeros((detection_size, detection_size)), None, None, None

    model_pt.zero_grad()
    target_score_tensor.backward(retain_graph=True)

    A = extractor.activations.squeeze(0)
    grads = extractor.gradients.squeeze(0)
    extractor.remove_hooks()

    g_1 = grads.clamp(min=0)
    numerator = g_1.pow(2)
    A_sum = A.sum(dim=(1, 2), keepdim=True)
    denominator = 2 * g_1.pow(2) + A_sum * g_1.pow(3)

    denominator = torch.where(
        denominator != 0.0, denominator, torch.ones_like(denominator).to(DEVICE)
    )

    alpha_ij_kc = numerator / denominator
    W_k_c = (alpha_ij_kc * g_1).sum(dim=(1, 2), keepdim=True)

    cam = (W_k_c * A).sum(dim=0)
    cam_numpy = torch.relu(cam).detach().cpu().numpy()

    if np.max(cam_numpy) > 0:
        cam_numpy = cam_numpy / np.max(cam_numpy)

    return (
        cam_numpy,
        target_class_index,
        box_data.xyxy[0].cpu().numpy().astype(int),
        results[0].names,
    )
