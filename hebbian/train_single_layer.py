import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import FullNetwork, KHDeep, KHLayer

EPOCHS_UNSUPERVISED = 100 # 100
EPOCHS_SUPERVISED = 300 # 300
BATCH_SIZE = 100
HIDDEN_UNITS = 2000
P_NORM = 2
DELTA = 0
LR_UNSUPERVISED = 0.005
ACTIVATION_POW = 4.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def flatten(x):
    return x.view(-1)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(flatten),
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)



def visualize_weights(weights, title="Learned Weights", filename="weights.png"):
    weights = weights.cpu().detach().numpy()
    max_val = np.max(np.abs(weights)) 
    if max_val < 1e-9: max_val = 1.0 
    
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    
    for i, ax in enumerate(axes.flat):
        if i < len(weights):
            img = weights[i].reshape(28, 28)
            ax.imshow(img, cmap='seismic', vmin=-max_val, vmax=max_val)
            ax.axis('off')
            
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig) 
    print(f"Saved visualization to {filename} (Range: +/- {max_val:.4f})")


bio_layer = KHLayer(
    28 * 28, 
    HIDDEN_UNITS, 
    p_norm=P_NORM, 
).to(device)
# bio = KHDeep(28 * 28).to(device)
for epoch in range(EPOCHS_UNSUPERVISED):
    current_lr = LR_UNSUPERVISED * (1 - epoch / EPOCHS_UNSUPERVISED)
    pbar = tqdm(train_loader, desc=f"Unsupervised Epoch {epoch+1}/{EPOCHS_UNSUPERVISED}")
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        bio_layer.unsupervised_update(data, current_lr, DELTA)

    if epoch and epoch % 10 == 0:
        indices = torch.randperm(bio_layer.weights.size(0))[:20]
        random_weights = bio_layer.weights[indices]
        visualize_weights(random_weights, title="Bio-Learning Weights", filename=f"bio_weights_final.png")

print("Unsupervised training complete.")


indices = torch.randperm(bio_layer.weights.size(0))[:20]
random_weights = bio_layer.weights[indices]
visualize_weights(random_weights, title="Bio-Learning Weights", filename="bio_weights_final.png")


model = FullNetwork(bio_layer).to(device)
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
