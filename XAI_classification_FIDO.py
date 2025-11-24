import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import copy

# ==========================================
# 1. Configuration et Chargement
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modèle de classification : ResNet50
classifier = models.resnet50(pretrained=True).to(device)
classifier.eval()
for param in classifier.parameters():
    param.requires_grad = False

# Prétraitement standard pour ImageNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(path):
    img = Image.open(path).convert('RGB')
    return preprocess(img).unsqueeze(0).to(device), img.resize((224, 224))

def denormalize(tensor):
    # Pour l'affichage
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return tensor * std + mean

# ==========================================
# 2. Le "Générateur" (In-filling)
# ==========================================
# Dans le papier, ils utilisent un GAN. Pour ce prototype rapide, 
# nous utilisons un fort flou gaussien qui sert de modèle génératif 
# de texture de fond (baseline du papier).
class GaussianInfiller(nn.Module):
    def __init__(self, kernel_size=31, sigma=10):
        super().__init__()
        self.blur = transforms.GaussianBlur(kernel_size, sigma)
        
    def forward(self, x, mask):
        # x: image originale
        # mask: masque binaire (ou continu entre 0 et 1)
        # Retourne: image mélangée
        background = self.blur(x)
        return x * mask + background * (1 - mask)

# ==========================================
# 3. Logique FIDO (Optimisation de Masque)
# ==========================================
class FIDOSolver:
    def __init__(self, classifier, infiller, device):
        self.classifier = classifier
        self.infiller = infiller
        self.device = device

    def tv_norm(self, x, beta=2):
        # Total Variation pour lisser le masque
        img_h = x.size(2)
        img_w = x.size(3)
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], beta).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], beta).sum()
        return (tv_h + tv_w)

    def optimize(self, img_tensor, target_class, mode='SSR', steps=300, lr=0.05, lambda_l1=1e-3, lambda_tv=1e-2):
        """
        mode 'SSR': Smallest Supporting Region (On veut garder le minimum de pixels pour GARDER la classe)
        mode 'SDR': Smallest Deletion Region (On veut supprimer le minimum de pixels pour TUER la classe)
        """
        
        # On optimise un masque plus petit (56x56) qu'on upsample, comme dans le papier (Regularization by downsampling)
        mask_logits = torch.ones(1, 1, 56, 56).to(self.device) * 0.5
        mask_logits.requires_grad = True
        
        optimizer = torch.optim.Adam([mask_logits], lr=lr)
        
        print(f"Optimisation du masque {mode} pour la classe {target_class}...")
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Sampling Gumbel-Softmax (Concrete Distribution)
            # Cela permet de dériver à travers l'échantillonnage d'un masque binaire
            # On sample 2 canaux (Keep vs Drop), on prend le canal 1 pour le masque
            # Forme: [Batch, 2, H, W]
            sample = F.gumbel_softmax(
                torch.stack([mask_logits, -mask_logits], dim=1).squeeze(), 
                tau=0.5, 
                hard=False, # Soft pour la dérivabilité, peut être True pour inference
                dim=0
            )
            mask_small = sample[0].unsqueeze(0).unsqueeze(0) # Garder probabilité "keep"
            
            # Upsampling à 224x224
            mask = F.interpolate(mask_small, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Génération de l'image contrefactuelle
            infilled_img = self.infiller(img_tensor, mask)
            
            # Prédiction du classifieur
            output = self.classifier(infilled_img)
            score = output[0, target_class] # Logit ou proba selon le modèle, ici Logit
            
            # Fonction de perte selon le mode
            l1_loss = torch.mean(torch.abs(mask))
            tv_loss = self.tv_norm(mask)
            
            loss = 0
            if mode == 'SSR':
                # On veut MAXIMISER le score avec le masque MINIMAL
                # Loss = -Score + lambda * Taille_Masque
                loss = -score + lambda_l1 * l1_loss + lambda_tv * tv_loss
            elif mode == 'SDR':
                # On veut MINIMISER le score en supprimant le MINIMUM de pixels (masque inversé)
                # Ici "mask" représente ce qu'on garde. 
                # Si SDR, on cherche la zone à supprimer.
                # L'équation du papier pour SDR cherche un masque Z (à supprimer).
                # Pour simplifier l'implémentation ici : 
                # SDR cherche à faire chuter le score en gardant (1-z) proche de l'image originale.
                loss = score + lambda_l1 * (1 - l1_loss) + lambda_tv * tv_loss

            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step}/{steps} | Loss: {loss.item():.4f} | Score Class: {score.item():.2f}")

        # Post-traitement: Binarisation finale pour l'affichage
        with torch.no_grad():
            final_mask_small = torch.sigmoid(mask_logits)
            final_mask = F.interpolate(final_mask_small, size=(224, 224), mode='bilinear', align_corners=False)
            
        return final_mask.detach()

# ==========================================
# 4. Exécution
# ==========================================

# --- Remplacer par votre image ---
# TÉLÉCHARGEZ UNE IMAGE D'ABORD, PAR EXEMPLE : "dog.jpg"
# !wget https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg -O dog.jpg
image_path = "data/elephant.jpg" 

try:
    x, x_pil = load_image(image_path)
except:
    print("Erreur: Image non trouvée. Veuillez placer une image 'dog.jpg' dans le dossier.")
    # Création d'une image dummy si pas de fichier
    x_pil = Image.new('RGB', (224, 224), color = 'red')
    x = preprocess(x_pil).unsqueeze(0).to(device)

# 1. Obtenir la prédiction initiale
logits = classifier(x)
target_class = torch.argmax(logits).item()
print(f"Classe prédite : {target_class}")

# 2. Initialiser le solver et l'infiller
infiller = GaussianInfiller(kernel_size=51, sigma=20).to(device) # Gros flou pour simuler un retrait d'objet
solver = FIDOSolver(classifier, infiller, device)

# 3. Calculer SSR (Smallest Supporting Region) -> Ce qui est nécessaire pour CLASSER
mask_ssr = solver.optimize(x, target_class, mode='SSR', lambda_l1=15.0) 

# 4. Calculer SDR (Smallest Deletion Region) -> Ce qui, si enlevé, CASSE la classification
# Note: Pour SDR, on veut trouver la petite région qui casse tout.
mask_sdr_logits = solver.optimize(x, target_class, mode='SDR', lambda_l1=10.0)
# Pour SDR, le masque résultant indique ce qu'on GARDE. La région "SDR" est donc l'inverse.
sdr_heatmap = 1 - mask_sdr_logits 

# ==========================================
# 5. Visualisation
# ==========================================
def apply_heatmap(img_tensor, mask, color=(1, 0, 0)):
    # Superposition simple pour visualisation
    img_np = denormalize(img_tensor).squeeze().cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    mask_np = mask.squeeze().cpu().numpy()
    
    fig, ax = plt.subplots()
    ax.imshow(img_np)
    ax.imshow(mask_np, cmap='jet', alpha=0.5)
    ax.axis('off')
    return fig

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Image originale
img_display = denormalize(x).squeeze().cpu().permute(1, 2, 0).numpy()
img_display = np.clip(img_display, 0, 1)
axes[0].imshow(img_display)
axes[0].set_title("Original Image")
axes[0].axis('off')

# SSR (Ce qu'il faut voir)
mask_ssr_np = mask_ssr.squeeze().cpu().numpy()
axes[1].imshow(img_display)
axes[1].imshow(mask_ssr_np, cmap='jet', alpha=0.5) # Le rouge indique la région de support
axes[1].set_title("SSR (Supporting Region)\nCe qui est nécessaire")
axes[1].axis('off')

# SDR (Ce qui perturbe le plus si enlevé)
mask_sdr_np = sdr_heatmap.squeeze().cpu().numpy()
axes[2].imshow(img_display)
axes[2].imshow(mask_sdr_np, cmap='jet', alpha=0.5) # Le rouge indique la région à supprimer
axes[2].set_title("SDR (Deletion Region)\nCe qui est le plus sensible")
axes[2].axis('off')

plt.tight_layout()
plt.show()