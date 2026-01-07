import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Importation du modèle pré-entraîné
from diffusers import AutoencoderKL

# ==========================================
# 1. Configuration
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modèle de classification : ResNet50
classifier = models.resnet50(pretrained=True).to(device)
classifier.eval()
for param in classifier.parameters():
    param.requires_grad = False

# Prétraitement pour ImageNet (ResNet)
# Normalisation : mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
resnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
resnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

def load_image(path):
    img = Image.open(path).convert('RGB').resize((224, 224))
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    # Normalisation ResNet
    return (tensor - resnet_mean) / resnet_std, img

def denormalize_resnet(tensor):
    return tensor * resnet_std + resnet_mean

# ==========================================
# 2. Le Modèle Générateur : VAE (Stable Diffusion)
# ==========================================
class PretrainedVAEInfiller(nn.Module):
    def __init__(self, device):
        super().__init__()
        print("Chargement du VAE pré-entraîné (StabilityAI)...")
        # On charge le VAE ft-mse (meilleure reconstruction)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            
    def forward(self, x_resnet, mask):
        """
        x_resnet : Image normalisée pour ResNet (Mean/Std spécifique)
        mask : Masque binaire (1=Keep, 0=Drop)
        """
        # 1. Conversion de format : ResNet -> VAE
        # ResNet est normalisé [0, 1] puis (x-mean)/std.
        # Le VAE attend des entrées en [-1, 1].
        
        # D'abord, on revient à [0, 1]
        x_01 = x_resnet * resnet_std + resnet_mean
        
        # Ensuite on passe à [-1, 1] pour le VAE
        x_vae_input = x_01 * 2.0 - 1.0
        
        # 2. Simulation de l'inconnu
        # Dans la zone masquée (mask=0), on injecte du bruit ou la moyenne
        # pour forcer le VAE à "halluciner" une reconstruction cohérente.
        noise = torch.randn_like(x_vae_input)
        masked_input = x_vae_input * mask + noise * (1 - mask)
        
        # 3. Passage dans le VAE (Encode -> Decode)
        # Le VAE va lisser le bruit et générer une texture plausible grâce à son Latent Space
        with torch.no_grad():
            posterior = self.vae.encode(masked_input).latent_dist
            latents = posterior.sample() * 0.18215 # Scaling factor standard SD
            reconstruction = self.vae.decode(latents / 0.18215).sample

        # 4. Conversion inverse : VAE [-1, 1] -> ResNet Norm
        reconstruction_01 = (reconstruction + 1.0) / 2.0
        reconstruction_resnet = (reconstruction_01 - resnet_mean) / resnet_std
        
        # 5. Composition finale (FIDO)
        # On garde l'original si mask=1, on prend le VAE si mask=0
        return x_resnet * mask + reconstruction_resnet * (1 - mask)

# ==========================================
# 3. Logique FIDO (Solver)
# ==========================================
class FIDOSolver:
    def __init__(self, classifier, infiller, device):
        self.classifier = classifier
        self.infiller = infiller
        self.device = device

    def tv_norm(self, x, beta=2):
        img_h = x.size(2)
        img_w = x.size(3)
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], beta).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], beta).sum()
        return (tv_h + tv_w)

    def optimize(self, img_tensor, target_class, mode='SSR', steps=50, lr=0.05, lambda_l1=1e-3, lambda_tv=1e-2):
        # Masque optimisé en basse résolution pour la régularité
        mask_logits = torch.ones(1, 1, 56, 56).to(self.device) * 0.5
        mask_logits.requires_grad = True
        
        optimizer = torch.optim.Adam([mask_logits], lr=lr)
        
        print(f"Optimisation FIDO ({mode}) avec VAE...")
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Gumbel Softmax (Differentiable Binary Mask sampling)
            sample = F.gumbel_softmax(
                torch.stack([mask_logits, -mask_logits], dim=1).squeeze(), 
                tau=0.5, hard=False, dim=0
            )
            mask_small = sample[0].unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask_small, size=(224, 224), mode='bilinear', align_corners=False)
            
            # --- Infilling via VAE ---
            infilled_img = self.infiller(img_tensor, mask)
            
            # Classification
            output = self.classifier(infilled_img)
            score = output[0, target_class]
            
            l1_loss = torch.mean(torch.abs(mask))
            tv_loss = self.tv_norm(mask)
            
            loss = 0
            # Coefficients ajustés pour le VAE
            if mode == 'SSR':
                # On veut Maximiser Score, Minimiser Masque
                loss = -score + 0.5 * l1_loss + lambda_tv * tv_loss
            elif mode == 'SDR':
                # On veut Minimiser Score, Minimiser Masque supprimé (donc Maximiser Masque gardé)
                loss = score + 0.5 * (1 - l1_loss) + lambda_tv * tv_loss

            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | Class Score: {score.item():.2f}")

        with torch.no_grad():
            final_mask_small = torch.sigmoid(mask_logits)
            final_mask = F.interpolate(final_mask_small, size=(224, 224), mode='bilinear', align_corners=False)
            
        return final_mask.detach()

# ==========================================
# 4. Exécution
# ==========================================
# !wget https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg -O dog.jpg
image_path = "data/elephant.jpg"

try:
    x, x_pil = load_image(image_path)
except:
    print("Création image dummy...")
    x_pil = Image.new('RGB', (224, 224), color = 'red')
    tensor = transforms.ToTensor()(x_pil).unsqueeze(0).to(device)
    x = (tensor - resnet_mean) / resnet_std

# Prédiction
logits = classifier(x)
target_class = torch.argmax(logits).item()
print(f"Classe cible: {target_class}")

# Initialisation Infiller (Télécharge automatiquement les poids ~300MB)
infiller = PretrainedVAEInfiller(device)

solver = FIDOSolver(classifier, infiller, device)

# SSR (Smallest Supporting Region) - Ce qu'il faut garder
mask_ssr = solver.optimize(x, target_class, mode='SSR', lambda_l1=0.2) 

# SDR (Smallest Deletion Region) - Ce qui casse la classe si enlevé
mask_sdr_logits = solver.optimize(x, target_class, mode='SDR', lambda_l1=0.2)
sdr_heatmap = 1 - mask_sdr_logits 

# ==========================================
# 5. Visualisation
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Image
img_display = denormalize_resnet(x).squeeze().cpu().permute(1, 2, 0).numpy()
img_display = np.clip(img_display, 0, 1)
axes[0].imshow(img_display)
axes[0].set_title("Original")
axes[0].axis('off')

# SSR
mask_ssr_np = mask_ssr.squeeze().cpu().numpy()
axes[1].imshow(img_display)
axes[1].imshow(mask_ssr_np, cmap='jet', alpha=0.5)
axes[1].set_title("SSR (VAE Infill)\nZone critique")
axes[1].axis('off')

# SDR
mask_sdr_np = sdr_heatmap.squeeze().cpu().numpy()
axes[2].imshow(img_display)
axes[2].imshow(mask_sdr_np, cmap='jet', alpha=0.5)
axes[2].set_title("SDR (VAE Infill)\nZone sensible")
axes[2].axis('off')

plt.tight_layout()
plt.show()