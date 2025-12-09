import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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
TOTAL_STEPS = 300_000
WARMUP_STEPS = 10_000
SAVE_FREQ = 10_000
LOG_FREQ = 100

TARGET_KLD_WEIGHT = 0.005
ANNEAL_START = 5_000
ANNEAL_END = 25_000

os.makedirs(SAVE_DIR, exist_ok=True)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

vae = VAE().to(device)
vae = torch.compile(vae)

optimizer = optim.Adam(vae.parameters(), lr=LR)
scaler = GradScaler()
autocast_ctx = autocast(device_type=device_type , dtype=torch.bfloat16)

scheduler1 = LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS
)
scheduler2 = CosineAnnealingLR(
    optimizer, T_max=TOTAL_STEPS - WARMUP_STEPS, eta_min=1e-6
)
scheduler = SequentialLR(
    optimizer, schedulers=[scheduler1, scheduler2], milestones=[WARMUP_STEPS]
)

dataset = RolloutsDataset()
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
    num_workers=2,
    pin_memory=True 
)
data_iter = iter(dataloader)

vae.train()
current_step = 0

while current_step < TOTAL_STEPS:
    try:
        data = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        data = next(data_iter)

    data = data.to(device)
    optimizer.zero_grad()

    if current_step < ANNEAL_START:
        kld_weight = 0.0
    elif current_step < ANNEAL_END:
        # Linear interpolation
        progress = (current_step - ANNEAL_START) / (ANNEAL_END - ANNEAL_START)
        kld_weight = progress * TARGET_KLD_WEIGHT
    else:
        kld_weight = TARGET_KLD_WEIGHT
    
    with autocast_ctx:
        results = vae(data)
        loss_dict = vae.loss_fn(*results, M_N=kld_weight)
        loss = loss_dict['loss']
        recon_loss = loss_dict['reconstruction_loss']

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    scheduler.step()
    current_step += 1

    if current_step % LOG_FREQ == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Step {current_step}/{TOTAL_STEPS} | LR: {current_lr:.2e} | Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f}")

    if current_step % SAVE_FREQ == 0:
        print(f"Saving checkpoint at step {current_step}...")
        torch.save(vae.state_dict(), f'{SAVE_DIR}/vae_{current_step}.pth')