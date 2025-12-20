import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import cv2

from src.model_utils import (
    load_resnet_model,
    load_yolo_model,
    PREPROCESS_CLF,
    DEVICE,
)
from src.xai_methods import (
    generate_gradcam_mask_clf,
    generate_gradcam_pp_mask_clf,
    generate_lime_mask_clf,
    generate_yolo_gradcam,
    generate_yolo_gradcam_pp,
)

from src.evaluation_metrics import run_xai_metrics

DATA_DIR = "data/images"
OUTPUT_FILE = "results/xai_comparison_results.csv"

print(f"Utilisation du device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")


def load_images(data_dir):
    if not os.path.exists(data_dir):
        print(f"Erreur : Le dossier {data_dir} n'existe pas.")
        return []
    return [
        {
            "path": os.path.join(data_dir, f),
            "name": f,
            "np_array": np.array(Image.open(os.path.join(data_dir, f)).convert("RGB")),
        }
        for f in os.listdir(data_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]


def main_evaluation():
    results_list = []

    model_resnet, layer_resnet = load_resnet_model()
    model_yolo, model_yolo_pt, layer_yolo = load_yolo_model()

    images_data = load_images(DATA_DIR)

    for image_data in tqdm(images_data, desc="Évaluation XAI"):
        img_np = image_data["np_array"]
        img_name = image_data["name"]

        input_clf = PREPROCESS_CLF(Image.fromarray(img_np)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            target_clf = model_resnet(input_clf).argmax(dim=1).item()

        methods_clf = {
            "GradCAM": lambda: generate_gradcam_mask_clf(
                model_resnet, input_clf, layer_resnet
            ),
            "GradCAMPP": lambda: generate_gradcam_pp_mask_clf(
                model_resnet, input_clf, layer_resnet
            ),
            "LIME": lambda: generate_lime_mask_clf(model_resnet, img_np),
        }

        for name, func in methods_clf.items():
            try:
                heatmap, _ = func()
                m = run_xai_metrics(model_resnet, img_np, heatmap, target_clf)
                results_list.append(
                    {
                        "Model": "ResNet50",
                        "Task": "Classification",
                        "Method": name,
                        "Image": img_name,
                        "AUC_Del": m["auc_del"],
                        "AUC_Pres": m["auc_pres"],
                        "Fidelity": (1 - m["auc_del"]) + m["auc_pres"],
                    }
                )
            except Exception as e:
                print(f"Erreur Clf {name}: {e}")

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        methods_yolo = {
            "YOLO_GradCAM": lambda: generate_yolo_gradcam(
                model_yolo, model_yolo_pt, img_bgr, layer_yolo
            ),
            "YOLO_GradCAMPP": lambda: generate_yolo_gradcam_pp(
                model_yolo, model_yolo_pt, img_bgr, layer_yolo
            ),
        }

        for name, func in methods_yolo.items():
            try:
                heatmap, target_idx, bbox, names = func()
                if target_idx is not None:
                    m = run_xai_metrics(model_yolo_pt, img_np, heatmap, target_idx)
                    results_list.append(
                        {
                            "Model": "YOLO",
                            "Task": "Detection",
                            "Method": name,
                            "Image": img_name,
                            "AUC_Del": m["auc_del"],
                            "AUC_Pres": m["auc_pres"],
                            "Fidelity": (1 - m["auc_del"]) + m["auc_pres"],
                        }
                    )
            except Exception as e:
                print(f"Erreur Yolo {name}: {e}")

    df = pd.DataFrame(results_list)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Terminé. Résultats sauvegardés dans {OUTPUT_FILE}")


if __name__ == "__main__":
    main_evaluation()
