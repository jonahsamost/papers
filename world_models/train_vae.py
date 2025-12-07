import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import GradScaler, autocast

from model import VAE
from build_dataset import RolloutsDataset

DATA_DIR = "data/rollouts"
NUM_ROLLOUTS = 10_000

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100
SAVE_DIR = 'checkpoints/vae'
os.makedirs(SAVE_DIR, exist_ok=True)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

vae = VAE().to(device)
vae = torch.compile(vae)

optimizer = optim.Adam(vae.parameters(), lr=LR)
scaler = GradScaler()
autocast_ctx = autocast(device_type=device_type , dtype=torch.bfloat16)

dataset = RolloutsDataset()
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
    num_workers=2,
    pin_memory=True 
)
for epoch in range(EPOCHS):
    print(f'Training epoch: {epoch}')
    vae.train()
    total_loss = 0
    batch_count = 0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        with autocast_ctx:
            results = vae(data)
            loss_dict = vae.loss_fn(*results, M_N=.005)
            loss = loss_dict['loss']
            recon_loss = loss_dict['reconstruction_loss']

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        batch_count += 1

        if batch_idx and batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | batch: {batch_idx} | loss: {loss.item():.4f} | recon loss: {recon_loss.item():.4f}')

    avg_loss = total_loss / batch_count
    torch.save(vae.state_dict(), f'{SAVE_DIR}/vae_{epoch}.pth')