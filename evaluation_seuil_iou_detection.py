import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.model_utils import load_yolo_model
from src.xai_methods import generate_yolo_gradcam, generate_yolo_gradcam_pp

IMAGE_DIRECTORY = r"data\VOC2012_test\JPEGImages"
MODEL_PATH = "yolov8n.pt"
SEUILS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_iou(heatmap, bbox, threshold):
    h, w = heatmap.shape
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    binary_mask = (heatmap_norm >= threshold).astype(np.uint8)

    x1, y1, x2, y2 = map(int, bbox)
    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_mask[y1:y2, x1:x2] = 1

    intersection = np.logical_and(binary_mask, bbox_mask).sum()
    union = np.logical_or(binary_mask, bbox_mask).sum()
    return intersection / union if union > 0 else 0


def study_sensitivity():
    model_yolo, model_pt, layer_yolo = load_yolo_model(MODEL_PATH)
    img_list = [
        f for f in os.listdir(IMAGE_DIRECTORY) if f.lower().endswith((".jpg", ".jpeg"))
    ][:100]

    results = []

    for img_name in tqdm(img_list, desc="Analyse Grad-CAM vs Grad-CAM++"):
        path = os.path.join(IMAGE_DIRECTORY, img_name)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        h_orig, w_orig = img_bgr.shape[:2]
        img_input = cv2.resize(img_bgr, ((w_orig // 32) * 32, (h_orig // 32) * 32))

        try:
            h_gc, _, bbox, _ = generate_yolo_gradcam(
                model_yolo, model_pt, img_input, layer_yolo
            )
            h_gcpp, _, _, _ = generate_yolo_gradcam_pp(
                model_yolo, model_pt, img_input, layer_yolo
            )

            if bbox is not None:
                for s in SEUILS:
                    iou_gc = calculate_iou(h_gc, bbox, s)
                    iou_gcpp = calculate_iou(h_gcpp, bbox, s)
                    results.append({"seuil": s, "method": "Grad-CAM", "iou": iou_gc})
                    results.append(
                        {"seuil": s, "method": "Grad-CAM++", "iou": iou_gcpp}
                    )
        except Exception:
            continue

    df = pd.DataFrame(results)
    summary = df.groupby(["seuil", "method"])["iou"].mean().unstack()

    plt.figure(figsize=(10, 6))
    plt.plot(
        summary.index, summary["Grad-CAM"], marker="o", label="Grad-CAM", linewidth=2
    )
    plt.plot(
        summary.index,
        summary["Grad-CAM++"],
        marker="s",
        label="Grad-CAM++",
        linewidth=2,
    )

    plt.title("Sensibilité de l'IoU en fonction du seuil (YOLOv8)")
    plt.xlabel("Seuil de la Heatmap")
    plt.ylabel("IoU Moyenne")
    plt.grid(True, linestyle="--")
    plt.legend()
    plt.savefig("sensitivity_analysis_iou.png")
    plt.show()

    print(
        "\nAnalyse terminée. Graphique sauvegardé sous 'sensitivity_analysis_iou.png'."
    )
    print(summary)


if __name__ == "__main__":
    study_sensitivity()
