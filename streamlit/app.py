import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Visualiseur Grad-CAM++", layout="centered")

st.title("🔍 Visualisation Grad-CAM++")
st.markdown("""
Cette méthode permet de comprendre **où un réseau de neurones regarde** pour prendre sa décision.
Le "Heatmap" généré surligne les zones de l'image qui ont le plus influencé la classification.
""")

# --- LOGIQUE DU NOTEBOOK (GRAD-CAM++) ---
feature_maps = {}
gradients = {}

def save_feature_maps(module, input, output):
    feature_maps["A"] = output.detach()

def save_gradients(module, grad_input, grad_output):
    gradients["grads"] = grad_output[0].detach()

def grad_cam_pp(model, target_layer, input_tensor):
    forward_handle = target_layer.register_forward_hook(save_feature_maps)
    backward_handle = target_layer.register_full_backward_hook(save_gradients)

    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    target_class_idx = output.argmax(dim=1).item()
    target_logit = output[0, target_class_idx]

    model.zero_grad()
    target_logit.backward(retain_graph=True)

    A = feature_maps["A"]
    grads = gradients["grads"]
    forward_handle.remove()
    backward_handle.remove()

    A = A.squeeze(0)
    grads = grads.squeeze(0)
    
    g_1 = grads.clamp(min=0)
    numerator = g_1.pow(2)
    denominator = 2 * g_1.pow(2) + A.sum(dim=(1, 2), keepdim=True) * g_1.pow(3)
    denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))

    alpha_ij_kc = numerator / denominator
    W_k_c = (alpha_ij_kc * g_1).sum(dim=(1, 2), keepdim=True)

    L_Grad_CAM_c = (W_k_c * A).sum(dim=0)
    L_Grad_CAM_c = torch.relu(L_Grad_CAM_c).detach().cpu().numpy()
    L_Grad_CAM_c = L_Grad_CAM_c / (L_Grad_CAM_c.max() + 1e-7)

    heatmap = cv2.resize(L_Grad_CAM_c, (input_tensor.shape[3], input_tensor.shape[2]))
    return heatmap, target_class_idx

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model

model = load_model()
target_layer = model.layer4[-1].conv3

# --- INTERFACE UTILISATEUR (UPLOAD) ---
uploaded_file = st.file_uploader("Choisissez une photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image originale", use_container_width=True)

    with st.spinner('Génération de la heatmap...'):
        # Prétraitement
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img).unsqueeze(0)

        # Calcul
        heatmap, class_idx = grad_cam_pp(model, target_layer, input_tensor)

         # Superposition de la heatmap
        heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

        # Redimensionner l'originale pour fusionner
        img_res = np.array(img.resize((224, 224)))
        superimposed_img = cv2.addWeighted(img_res, 0.6, heatmap_img, 0.4, 0)

        # Affichage
        st.subheader(f"Résultat (Classe détectée : {class_idx})")
        st.image(superimposed_img, caption="Heatmap Grad-CAM++", use_container_width=True)