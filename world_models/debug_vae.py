import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from load_model import load_vae 
from build_dataset import RolloutsDataset

def visualize_vae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading VAE...")
    vae = load_vae(device=device)
    vae.eval() # Freezes BatchNorm/Dropout
    
    dataset = RolloutsDataset(buffer_size=100)
    dataloader = DataLoader(dataset, batch_size=8)
    
    real_imgs = next(iter(dataloader))
    real_imgs = real_imgs.to(device) # (8, 3, 64, 64)

    with torch.no_grad():
        mu, logvar = vae.encode(real_imgs)
        z = vae.reparameterize(mu, logvar)
        recons = vae.decode(z)
        
        random_z = torch.randn(8, vae.latent_dim).to(device)
        dreams = vae.decode(random_z)

    def to_np(t):
        return t.permute(0, 2, 3, 1).cpu().numpy()
    
    real_np = to_np(real_imgs)
    recon_np = to_np(recons)
    dream_np = to_np(dreams)
    
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    
    # Row 1: Real Images
    for i in range(8):
        axes[0, i].imshow(real_np[i])
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Real Inputs")

    # Row 2: Reconstructions
    for i in range(8):
        axes[1, i].imshow(np.clip(recon_np[i], 0, 1))
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Reconstructions")

    # Row 3: Random Dreams (from z ~ N(0,1))
    for i in range(8):
        axes[2, i].imshow(np.clip(dream_np[i], 0, 1))
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title("Random Dreams")

    plt.tight_layout()
    plt.savefig("vae_debug.png")
    print("Saved visualization to vae_debug.png")


visualize_vae()