import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import FullNetworkDeep, KHDeep, KHLayer

EPOCHS_UNSUPERVISED = 100 # 100
EPOCHS_SUPERVISED = 300 # 300
BATCH_SIZE = 100
HIDDEN_UNITS = 2000
P_NORM = 2
DELTA = 0
LR_UNSUPERVISED = 0.005

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def flatten(x):
    return x.view(-1)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(flatten),
])

train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


def visualize_deep_network(model, epoch, num_samples=10, filename="deep_viz.png"):
    model.eval()
    num_layers = len(model.layers)
    
    fig, axes = plt.subplots(num_layers, num_samples, figsize=(num_samples * 1.5, num_layers * 1.5))
    
    if num_layers == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            indices = torch.randperm(layer.output_dim)[:num_samples]
            projected_weights = layer.weights[indices].clone()
            for prev_idx in range(layer_idx - 1, -1, -1):
                prev_layer = model.layers[prev_idx]
                projected_weights = projected_weights @ prev_layer.weights
            
            w_np = projected_weights.cpu().numpy()
            
            for col in range(num_samples):
                ax = axes[layer_idx, col]
                if col < len(w_np):
                    img = w_np[col].reshape(28, 28)
                    
                    max_val = np.max(np.abs(img))
                    if max_val < 1e-9: max_val = 1.0
                    
                    ax.imshow(img, cmap='seismic', vmin=-max_val, vmax=max_val)
                
                ax.axis('off')
            
            axes[layer_idx, 0].text(-5, 14, f"L{layer_idx}", fontsize=12, fontweight='bold', ha='right')

    plt.suptitle(f"Deep Network Features (Epoch {epoch})", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved deep visualization to {filename}")


bio = KHDeep(28 * 28).to(device)
for layer_idx, layer in enumerate(bio.layers):
    print(f"\n=== Training Layer {layer_idx} ===")
    
    epochs = 30
    for epoch in range(epochs):
        current_lr = LR_UNSUPERVISED * (1 - epoch / epochs)
        pbar = tqdm(train_loader, desc=f"L{layer_idx} Epoch {epoch+1}")
        for data, _ in pbar:
            data = data.to(device)
            with torch.no_grad():
                stream = data
                for prev_i in range(layer_idx):
                    prev_out = bio.layers[prev_i](stream)
                    if prev_i == 0:
                        stream = prev_out
                    else:
                        stream = stream + prev_out
            
            layer.unsupervised_update(stream, current_lr, DELTA)
        if (epoch+1) % 10 == 0:
            visualize_deep_network(
                bio, 
                epoch=epoch+1, 
                num_samples=8, 
                filename=f"deep_features_epoch.png"
            )

print("Unsupervised training complete.")

model = FullNetworkDeep(bio).to(device)
optimizer = optim.Adam(model.parameters(), lr=.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS_SUPERVISED):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Supervised Epoch {epoch+1}/{EPOCHS_SUPERVISED}")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        preds = model(data)
        optimizer.zero_grad()
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(preds, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Acc: {100 * correct / total:.2f}%")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"\nFinal Test Accuracy: {100 * correct / total:.2f}%")
