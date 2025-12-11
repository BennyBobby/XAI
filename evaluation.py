import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
from tqdm import tqdm


from src.model_utils import (
    load_resnet_model,
    load_yolo_model,
    get_imagenet_class_names,
    PREPROCESS_CLF,
    DEVICE,
)
from src.xai_methods import (
    generate_gradcam_mask_clf,
    generate_gradcam_pp_mask_clf,
    generate_yolo_gradcam,
)
from src.evaluation_metrics import run_deletion_test, run_preservation_test

DATA_DIR = "data\images"
OUTPUT_FILE = "results/xai_comparison_results.csv"
METHODS = {
    "GradCAM": generate_gradcam_mask_clf,
    "GradCAMPP": generate_gradcam_pp_mask_clf,
}
MODELS = ["ResNet50"]


def load_images(data_dir):
    image_paths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    images = []
    print(f"Loading {len(image_paths)} images...")
    for path in image_paths:
        try:
            img_pil = Image.open(path).convert("RGB")
            img_np = np.array(img_pil)
            images.append(
                {
                    "path": path,
                    "name": os.path.basename(path),
                    "np_array": img_np,
                }
            )
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return images


def main_evaluation():

    results_list = []

    model_clf, target_layer_clf = load_resnet_model()

    images_data = load_images(DATA_DIR)

    for image_data in tqdm(images_data, desc="Processing Images"):

        img_np = image_data["np_array"]
        img_name = image_data["name"]

        input_tensor_clf = (
            PREPROCESS_CLF(Image.fromarray(img_np)).unsqueeze(0).to(DEVICE)
        )
        with torch.no_grad():
            output = model_clf(input_tensor_clf)
            target_class_idx = output.argmax(dim=1).item()

        for method_name, method_func in METHODS.items():
            try:
                if method_name in ["GradCAM", "GradCAMPP"]:
                    heatmap_mask, _ = method_func(
                        model_clf, input_tensor_clf, target_layer_clf
                    )

                _, _, deletion_auc = run_deletion_test(
                    model_clf, img_np, heatmap_mask, target_class_idx
                )

                _, _, preservation_auc = run_preservation_test(
                    model_clf, img_np, heatmap_mask, target_class_idx
                )

                results_list.append(
                    {
                        "Model": "ResNet50",
                        "Method": method_name,
                        "Image": img_name,
                        "Target_Class": target_class_idx,
                        "Deletion_AOC": deletion_auc,
                        "Preservation_AOC": preservation_auc,
                    }
                )

            except Exception as e:
                print(f"Error for {method_name} on {img_name}: {e}")
                results_list.append(
                    {
                        "Model": "ResNet50",
                        "Method": method_name,
                        "Image": img_name,
                        "Target_Class": target_class_idx,
                        "Deletion_AOC": np.nan,
                        "Preservation_AOC": np.nan,
                    }
                )

    df_results = pd.DataFrame(results_list)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"Résultats sauvegardés dans : {OUTPUT_FILE}")
    print(df_results.head())


if __name__ == "__main__":
    main_evaluation()
