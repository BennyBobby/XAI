import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.model_utils import load_yolo_model
from src.xai_methods import generate_yolo_gradcam, generate_yolo_gradcam_pp

IMAGE_DIRECTORY = r"data\VOC2012_test\JPEGImages"
METHOD_NAME = "Grad-CAM"
TASK_DOMAIN = "Object Detection"
OUTPUT_DIR = "evaluation_result"
MODEL_PATH = "yolov8n.pt"
THRESHOLD = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_iou_energy(heatmap, bbox, threshold=0.8):
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    binary_mask = (heatmap_norm >= threshold).astype(np.uint8)

    x1, y1, x2, y2 = map(int, bbox)
    h, w = heatmap.shape

    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_mask[y1:y2, x1:x2] = 1

    intersection = np.logical_and(binary_mask, bbox_mask).sum()
    union = np.logical_or(binary_mask, bbox_mask).sum()

    iou = intersection / union if union > 0 else 0

    total_energy = binary_mask.sum()
    energy_in_bbox = intersection
    precision_ratio = energy_in_bbox / total_energy if total_energy > 0 else 0

    return iou, precision_ratio


def benchmark_pointing_game():
    safe_name = METHOD_NAME.replace(" ", "_").replace("+", "p")
    output_csv = os.path.join(OUTPUT_DIR, f"advanced_eval_{safe_name}.csv")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model_yolo, model_pt, layer_yolo = load_yolo_model(MODEL_PATH)
    model_yolo.to(DEVICE)
    model_pt.to(DEVICE)

    img_list = [
        f
        for f in os.listdir(IMAGE_DIRECTORY)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    results = []

    for img_name in tqdm(img_list):
        path = os.path.join(IMAGE_DIRECTORY, img_name)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        try:
            func = (
                generate_yolo_gradcam_pp
                if METHOD_NAME == "Grad-CAM++"
                else generate_yolo_gradcam
            )
            heatmap, target_idx, bbox, names = func(
                model_yolo, model_pt, img_bgr, layer_yolo
            )

            if bbox is not None:
                heatmap_rescaled = cv2.resize(heatmap, (w, h))
                iou, ratio = calculate_iou_energy(
                    heatmap_rescaled, bbox, threshold=THRESHOLD
                )

                results.append(
                    {
                        "image": img_name,
                        "class": names[target_idx] if names else "N/A",
                        "iou_mask": iou,
                        "precision_ratio": ratio,
                    }
                )
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"Erreur {img_name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nMoyenne IoU: {df['iou_mask'].mean():.4f}")
    print(f"Ratio d'énergie dans Bbox: {df['precision_ratio'].mean():.4f}")


if __name__ == "__main__":
    benchmark_pointing_game()
