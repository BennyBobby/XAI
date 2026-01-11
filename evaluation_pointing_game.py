import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.model_utils import load_yolo_model
from src.xai_methods import generate_yolo_gradcam, generate_yolo_gradcam_pp

IMAGE_DIRECTORY = r"data\VOC2012_test\JPEGImages"
METHOD_NAME = "Grad-CAM"  # "Grad-CAM" ou "Grad-CAM++"
TASK_DOMAIN = "Object Detection"
OUTPUT_DIR = "evaluation_result"
MODEL_PATH = "yolov8n.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_pointing_game(heatmap, bbox):
    idx_max = np.argmax(heatmap)
    y_max, x_max = np.unravel_index(idx_max, heatmap.shape)
    x1, y1, x2, y2 = bbox
    is_hit = (x1 <= x_max <= x2) and (y1 <= y_max <= y2)
    return 1 if is_hit else 0


def benchmark_pointing_game():
    safe_method_name = METHOD_NAME.replace(" ", "_").replace("+", "p")
    output_filename = f"pointing_game_{safe_method_name}.csv"
    output_csv = os.path.join(OUTPUT_DIR, output_filename)

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

    print(f"\nLancement du Benchmark sur {DEVICE}")
    print(f"Méthode choisie : {METHOD_NAME}")

    for img_name in tqdm(img_list):
        path = os.path.join(IMAGE_DIRECTORY, img_name)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]

        try:
            if METHOD_NAME == "Grad-CAM++":
                heatmap, target_idx, bbox, names = generate_yolo_gradcam_pp(
                    model_yolo, model_pt, img_bgr, layer_yolo
                )
            else:
                heatmap, target_idx, bbox, names = generate_yolo_gradcam(
                    model_yolo, model_pt, img_bgr, layer_yolo
                )

            if bbox is not None:
                heatmap_rescaled = cv2.resize(heatmap, (w, h))
                hit = calculate_pointing_game(heatmap_rescaled, bbox)
                class_name = names[target_idx] if names else "N/A"

                results.append(
                    {
                        "image": img_name,
                        "method": METHOD_NAME,
                        "domain": TASK_DOMAIN,
                        "class": class_name,
                        "hit": hit,
                    }
                )
        except Exception as e:
            if "cuda" in str(e).lower():
                torch.cuda.empty_cache()
            print(f"Erreur sur {img_name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    accuracy = df["hit"].mean() * 100
    print(f"\nTerminé ! Précision {METHOD_NAME} : {accuracy:.2f}%")


if __name__ == "__main__":
    benchmark_pointing_game()
