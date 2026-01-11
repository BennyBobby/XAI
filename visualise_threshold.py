import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.model_utils import load_yolo_model
from src.xai_methods import generate_yolo_gradcam_pp

IMAGE_PATH = r"data\sample_images\classic_car.jpg"
THRESHOLD = 0.5
MODEL_PATH = "yolov8n.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualiser_seuil():

    m_yolo, m_pt, layer = load_yolo_model(MODEL_PATH)
    img_bgr = cv2.imread(IMAGE_PATH)
    h, w = img_bgr.shape[:2]

    heatmap, _, bbox, _ = generate_yolo_gradcam_pp(m_yolo, m_pt, img_bgr, layer)
    heatmap_rescaled = cv2.resize(heatmap, (w, h))
    heatmap_norm = (heatmap_rescaled - heatmap_rescaled.min()) / (
        heatmap_rescaled.max() - heatmap_rescaled.min() + 1e-8
    )
    binary_mask = (heatmap_norm >= THRESHOLD).astype(np.uint8) * 255

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    img_bbox = img_rgb.copy()
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 0, 0), 3)
    plt.imshow(img_bbox)
    plt.title("Image Originale & BBox")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_rgb)
    plt.imshow(heatmap_rescaled, cmap="jet", alpha=0.5)
    plt.title(f"Heatmap Grad-CAM++")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(binary_mask, cmap="gray")
    plt.gca().add_patch(
        plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2
        )
    )
    plt.title(f"Masque Binaire (Seuil: {THRESHOLD})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualiser_seuil()
