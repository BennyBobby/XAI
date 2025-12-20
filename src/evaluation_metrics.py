import torch
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.metrics import auc
from PIL import Image
from .model_utils import DEVICE, PREPROCESS_CLF


def get_perturbed_tensor(
    original_image_np, heatmap_np, fraction, mode="deletion", perturbation_type="blur"
):
    H, W, C = original_image_np.shape
    if heatmap_np.shape[:2] != (H, W):
        heatmap_np = cv2.resize(heatmap_np, (W, H))

    flat_heatmap = heatmap_np.flatten()
    sorted_indices = np.argsort(flat_heatmap)[
        ::-1
    ]  # Du plus important au moins important

    num_pixels = int(H * W * fraction)
    binary_mask = np.zeros((H * W), dtype=np.float32)

    if mode == "deletion":
        indices_to_perturb = sorted_indices[:num_pixels]
        binary_mask[indices_to_perturb] = 1.0
    elif mode == "preservation":
        indices_to_keep = sorted_indices[:num_pixels]
        binary_mask[:] = 1.0
        binary_mask[indices_to_keep] = 0.0

    binary_mask = binary_mask.reshape(H, W, 1)

    if perturbation_type == "blur":
        background = cv2.GaussianBlur(original_image_np, (51, 51), 0)
    else:
        background = np.full_like(original_image_np, 128)

    perturbed_img = (
        binary_mask * background + (1.0 - binary_mask) * original_image_np
    ).astype(np.uint8)
    perturbed_img_pil = Image.fromarray(perturbed_img)
    return PREPROCESS_CLF(perturbed_img_pil).unsqueeze(0).to(DEVICE)


def run_xai_metrics(model, original_img_np, heatmap_np, target_class_idx, steps=20):
    percentages = np.linspace(0, 1.0, steps)
    deletion_scores = []
    preservation_scores = []

    model.eval()
    with torch.no_grad():
        for fraction in percentages:
            # Délétion
            t_del = get_perturbed_tensor(
                original_img_np, heatmap_np, fraction, mode="deletion"
            )
            out_del = torch.softmax(model(t_del), dim=1)
            deletion_scores.append(out_del[0, target_class_idx].item())
            # Préservation
            t_pres = get_perturbed_tensor(
                original_img_np, heatmap_np, fraction, mode="preservation"
            )
            out_pres = torch.softmax(model(t_pres), dim=1)
            preservation_scores.append(out_pres[0, target_class_idx].item())

    return {
        "auc_del": auc(percentages, deletion_scores),
        "auc_pres": auc(percentages, preservation_scores),
    }
