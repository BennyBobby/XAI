import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from src.xai_methods import generate_gradcam_mask_clf, generate_gradcam_pp_mask_clf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_PATH = r"data\sample_images\chat_roux.jpg"
OUTPUT_DIR = "evaluation_result"
STEPS = 50

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE)
model.eval()
target_layer = model.layer4[2]


def clear_hooks(model):
    for module in model.modules():
        module._forward_hooks.clear()
        module._backward_hooks.clear()
        module._forward_pre_hooks.clear()


def preprocess_image(path):
    raw_img = cv2.imread(path)
    if raw_img is None:
        raise FileNotFoundError(path)
    raw_img = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), (224, 224))
    img_norm = (raw_img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [
        0.229,
        0.224,
        0.225,
    ]
    return (
        torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float(),
        raw_img,
    )


def calculate_drop_increase(model, input_tensor, heatmap, target_class):
    h, w = 224, 224
    hm = cv2.resize(heatmap, (w, h))
    mask = torch.from_numpy((hm >= np.percentile(hm, 85)).astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        s_orig = torch.softmax(model(input_tensor), dim=1)[0, target_class].item()
        s_expl = torch.softmax(model(input_tensor * mask), dim=1)[
            0, target_class
        ].item()
    return max(0, s_orig - s_expl) / s_orig * 100, (1 if s_expl > s_orig else 0)


def run_deletion_test(model, input_tensor, heatmap, target_class, method_name):
    h, w = 224, 224
    hm = cv2.resize(heatmap, (w, h))
    indices = np.argsort(hm.flatten())[::-1]
    scores = []
    curr = input_tensor.clone()
    step = len(indices) // STEPS
    print(f"\nLancement Deletion: {method_name}")
    for i in range(STEPS + 1):
        with torch.no_grad():
            scores.append(torch.softmax(model(curr), dim=1)[0, target_class].item())
        if i % 10 == 0:
            print(f"  {method_name} -> Etape {i}/{STEPS}: {scores[-1]:.4f}")
        if i < STEPS:
            for idx in indices[i * step : (i + 1) * step]:
                y, x = np.unravel_index(idx, (h, w))
                curr[0, :, y, x] = 0
    return scores


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    img_t, _ = preprocess_image(IMAGE_PATH)

    print("--- ETAPE 1: GENERATION DES MASQUES ---")
    clear_hooks(model)
    hm_gc, cid = generate_gradcam_mask_clf(model, img_t, target_layer)
    clear_hooks(model)
    hm_gcpp, _ = generate_gradcam_pp_mask_clf(model, img_t, target_layer)

    print("--- ETAPE 2: METRIQUES STATIQUES ---")
    clear_hooks(model)
    d_gc, i_gc = calculate_drop_increase(model, img_t, hm_gc, cid)
    d_pp, i_pp = calculate_drop_increase(model, img_t, hm_gcpp, cid)

    print("--- ETAPE 3: TESTS DE DELETION ---")
    torch.cuda.empty_cache()
    s_gc = run_deletion_test(model, img_t, hm_gc, cid, "Grad-CAM")

    # Si ton code s'arrête ici, c'est que la ligne suivante ne s'exécute jamais
    torch.cuda.empty_cache()
    s_pp = run_deletion_test(model, img_t, hm_gcpp, cid, "Grad-CAM++")

    print("\n" + "=" * 45)
    print(f"RESULTATS | AUC(↓) | DROP(↓)")
    print(f"GC        | {np.mean(s_gc):.3f} | {d_gc:.2f}%")
    print(f"GC++      | {np.mean(s_pp):.3f} | {d_pp:.2f}%")
    print("=" * 45)

    plt.figure()
    plt.plot(s_gc, label="GC", color="blue")
    plt.plot(s_pp, label="GC++", color="red")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "result.png"))
    print("\nFichier result.png créé.")
    plt.show()


if __name__ == "__main__":
    main()
