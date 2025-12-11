import torch
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.draw import rectangle
from sklearn.metrics import auc
from .model_utils import DEVICE, PREPROCESS_CLF
from PIL import Image


def get_perturbed_tensor(
    original_image_np, heatmap_np, fraction, mode="deletion", perturbation_type="blur"
):
    H, W, C = original_image_np.shape

    flat_heatmap = heatmap_np.flatten()
    sorted_indices = np.argsort(flat_heatmap)

    num_pixels = int(H * W * fraction)

    if mode == "deletion":
        target_indices = sorted_indices[-num_pixels:]
        mask_value = 1.0
    elif mode == "preservation":
        target_indices = sorted_indices[:-num_pixels]
        mask_value = 0.0
    else:
        raise ValueError("Le mode doit être 'deletion' ou 'preservation'.")

    binary_mask = np.zeros((H * W), dtype=np.float32)

    if mode == "deletion":
        binary_mask[target_indices] = 1.0
    elif mode == "preservation":
        all_indices = np.arange(H * W)
        preserved_indices = sorted_indices[-num_pixels:]
        mask_to_perturb = np.setdiff1d(all_indices, preserved_indices)
        binary_mask[mask_to_perturb] = 1.0

    binary_mask = binary_mask.reshape(H, W)

    perturbed_img = original_image_np.copy()

    if perturbation_type == "blur":
        blurred_img = cv2.GaussianBlur(original_image_np, (51, 51), 0)
        for i in range(C):
            perturbed_img[:, :, i] = (
                binary_mask * blurred_img[:, :, i]
                + (1 - binary_mask) * original_image_np[:, :, i]
            )

    elif perturbation_type == "constant":
        gray_img = np.full_like(original_image_np, 128, dtype=np.uint8)
        for i in range(C):
            perturbed_img[:, :, i] = (
                binary_mask * gray_img[:, :, i]
                + (1 - binary_mask) * original_image_np[:, :, i]
            )

    perturbed_img_pil = Image.fromarray(perturbed_img.astype(np.uint8))
    input_tensor = PREPROCESS_CLF(perturbed_img_pil).unsqueeze(0).to(DEVICE)

    return input_tensor


def run_deletion_test(model, original_img_np, heatmap_np, target_class_idx, steps=20):
    percentages = np.linspace(0, 1.0, steps)
    confidence_scores = []

    for fraction in percentages:
        perturbed_tensor = get_perturbed_tensor(
            original_img_np,
            heatmap_np,
            fraction,
            mode="deletion",
            perturbation_type="blur",
        )

        with torch.no_grad():
            output = model(perturbed_tensor)
            probs = F.softmax(output, dim=1)
            confidence = probs[0, target_class_idx].item()
            confidence_scores.append(confidence)

    auc_score = auc(percentages, confidence_scores)

    return percentages, confidence_scores, auc_score


def run_preservation_test(
    model, original_img_np, heatmap_np, target_class_idx, steps=20
):
    percentages = np.linspace(0, 1.0, steps)
    confidence_scores = []

    for fraction in percentages:
        perturbed_tensor = get_perturbed_tensor(
            original_img_np,
            heatmap_np,
            fraction,
            mode="preservation",
            perturbation_type="blur",
        )

        with torch.no_grad():
            output = model(perturbed_tensor)
            probs = F.softmax(output, dim=1)
            confidence = probs[0, target_class_idx].item()
            confidence_scores.append(confidence)

    auc_score = auc(percentages, confidence_scores)

    return percentages, confidence_scores, auc_score
