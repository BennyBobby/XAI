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
    sorted_indices = np.argsort(flat_heatmap)[::-1]

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
    img_pil = Image.fromarray(perturbed_img)
    return PREPROCESS_CLF(img_pil).unsqueeze(0)


def run_xai_metrics(model, original_img_np, heatmap_np, target_class_idx, steps=20):
    percentages = np.linspace(0, 1.0, steps)
    list_t_del = []
    list_t_pres = []

    for fraction in percentages:
        list_t_del.append(
            get_perturbed_tensor(original_img_np, heatmap_np, fraction, mode="deletion")
        )
        list_t_pres.append(
            get_perturbed_tensor(
                original_img_np, heatmap_np, fraction, mode="preservation"
            )
        )

    batch_del = torch.cat(list_t_del, dim=0).to(DEVICE)
    batch_pres = torch.cat(list_t_pres, dim=0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output_del = model(batch_del)
        output_pres = model(batch_pres)

        if isinstance(output_del, tuple):
            output_del = output_del[0]
        if isinstance(output_pres, tuple):
            output_pres = output_pres[0]

        if output_del.dim() == 3 and output_del.shape[1] > 10:
            class_idx_in_yolo = 4 + target_class_idx
            idx = min(class_idx_in_yolo, output_del.shape[1] - 1)
            del_scores = (
                torch.sigmoid(output_del[:, idx, :]).max(dim=1)[0].cpu().numpy()
            )
            pres_scores = (
                torch.sigmoid(output_pres[:, idx, :]).max(dim=1)[0].cpu().numpy()
            )
        else:
            del_scores = (
                torch.softmax(output_del, dim=1)[:, target_class_idx].cpu().numpy()
            )
            pres_scores = (
                torch.softmax(output_pres, dim=1)[:, target_class_idx].cpu().numpy()
            )

    return {
        "auc_del": auc(percentages, del_scores),
        "auc_pres": auc(percentages, pres_scores),
    }
