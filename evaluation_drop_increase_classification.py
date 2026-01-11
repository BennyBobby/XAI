import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image
from src.xai_methods import generate_gradcam_pp_mask_clf, generate_gradcam_mask_clf


IMAGE_DIRECTORY = r"data\VOC2012_test\JPEGImages"
MODEL_NAME = "ResNet50"
THRESHOLD = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_resnet_faithfulness():

    model = (
        models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE).eval()
    )
    target_layer = model.layer4[-1]

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_list = [
        f
        for f in os.listdir(IMAGE_DIRECTORY)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    results = []

    for img_name in tqdm(img_list, desc="Évaluation ResNet"):
        path = os.path.join(IMAGE_DIRECTORY, img_name)
        raw_img = cv2.imread(path)
        if raw_img is None:
            continue

        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)

        heatmap, target_class = generate_gradcam_mask_clf(
            model, img_tensor, target_layer
        )

        heatmap_rescaled = cv2.resize(heatmap, (224, 224))
        mask = (heatmap_rescaled >= THRESHOLD).astype(np.float32)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            def get_prob(it):
                output = model(it)
                return torch.softmax(output, dim=1)[0, target_class].item()

            p_orig = get_prob(img_tensor)
            p_del = get_prob(img_tensor * (1 - mask_t))
            p_pres = get_prob(img_tensor * mask_t)

        drop = max(0, p_orig - p_del) / (p_orig + 1e-8) * 100
        increase = 1 if p_pres > p_orig else 0

        results.append({"drop": drop, "increase": increase})

    df = pd.DataFrame(results)
    print(f"\nRÉSULTATS RESNET50")
    print(f"Drop % : {df['drop'].mean():.2f}%")
    print(f"Increase % : {df['increase'].mean()*100:.2f}%")


if __name__ == "__main__":
    run_resnet_faithfulness()
