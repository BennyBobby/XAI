import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.model_utils import load_yolo_model
from src.xai_methods import generate_yolo_gradcam_pp, generate_yolo_gradcam

IMAGE_DIRECTORY = r"data\VOC2012_test\JPEGImages"
METHOD_NAME = "Grad-CAM"
MODEL_PATH = "yolov8n.pt"
OUTPUT_FILE = "faithfulness_results.csv"
THRESHOLD = 0.8  # Seuil pour définir ce qui est "l'explication"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_yolo_confidence(model_pt, img_tensor, target_class):
    """
    Extrait le score de confiance maximal pour une classe donnée dans YOLOv8.
    """
    with torch.no_grad():
        outputs = model_pt(img_tensor)
        preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Les scores de classes commencent à l'index 4
        # On applique sigmoid car YOLOv8 n'utilise pas softmax en sortie de détection
        class_scores = torch.sigmoid(preds[0, 4 + target_class, :])

        return torch.max(class_scores).item()


def run_faithfulness_benchmark():

    model_yolo, model_pt, layer_yolo = load_yolo_model(MODEL_PATH)
    model_yolo.to(DEVICE)
    model_pt.to(DEVICE)

    img_list = [
        f
        for f in os.listdir(IMAGE_DIRECTORY)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    results = []

    print(f"🚀 Évaluation de la fidélité ({METHOD_NAME}) sur {DEVICE}")

    pbar = tqdm(img_list, desc="Analyse")
    for img_name in pbar:
        path = os.path.join(IMAGE_DIRECTORY, img_name)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        h_orig, w_orig = img_bgr.shape[:2]
        img_bgr = cv2.resize(img_bgr, ((w_orig // 32) * 32, (h_orig // 32) * 32))
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
                heatmap_norm = (heatmap_rescaled - heatmap_rescaled.min()) / (
                    heatmap_rescaled.max() - heatmap_rescaled.min() + 1e-8
                )
                mask = (heatmap_norm >= THRESHOLD).astype(np.float32)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_t = (
                    torch.from_numpy(img_rgb)
                    .permute(2, 0, 1)
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                    .to(DEVICE)
                )
                mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(DEVICE)

                # Calcul des scores (Original, Deleted, Preserved)
                s_orig = get_yolo_confidence(model_pt, img_t, target_idx)
                s_del = get_yolo_confidence(model_pt, img_t * (1 - mask_t), target_idx)
                s_pres = get_yolo_confidence(model_pt, img_t * mask_t, target_idx)

                # Calcul des métriques
                # Drop % : Chute de confiance quand on supprime l'explication
                drop_percent = max(0, s_orig - s_del) / (s_orig + 1e-8) * 100

                # Increase % : Confiance conservée/augmentée avec juste l'explication (Voulu : 1)
                increase_flag = 1 if s_pres > s_orig else 0

                results.append(
                    {
                        "image": img_name,
                        "class": names[target_idx],
                        "score_original": round(s_orig, 4),
                        "drop_percent": round(drop_percent, 2),
                        "increase": increase_flag,
                    }
                )

        except Exception as e:
            pbar.write(f"Erreur {img_name}: {str(e)}")
            torch.cuda.empty_cache()
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"method : {METHOD_NAME}")

    print(f"Average Drop % : {df['drop_percent'].mean():.2f}%")
    print(f"Average Increase % : {df['increase'].mean()*100:.2f}%")


if __name__ == "__main__":
    run_faithfulness_benchmark()
