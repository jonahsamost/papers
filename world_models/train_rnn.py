import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import GradScaler, autocast

from model import VAE, MDN_RNN
from build_dataset import SequenceDataset
from load_model import load_vae

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
VAE_DIR = 'checkpoints/vae'
RNN_DIR = 'checkpoints/rnn'

os.makedirs(RNN_DIR, exist_ok=True)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

vae = load_vae(device=device)
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

rnn = MDN_RNN().to(device)
rnn = torch.compile(rnn)

optimizer = optim.Adam(rnn.parameters(), lr=LR)
scaler = GradScaler()
autocast_ctx = autocast(device_type=device_type, dtype=torch.bfloat16)

dataset = SequenceDataset()
dataloader = DataLoader( dataset, batch_size=BATCH_SIZE)

for epoch in range(EPOCHS):
    rnn.train()
    total_loss = 0
    batch_count = 0
    for batch_idx, (obs_seq, action_seq) in enumerate(dataloader):
        obs_seq = obs_seq.to(device)
        action_seq = action_seq.to(device)

        optimizer.zero_grad()

        with autocast_ctx:
            b, s, c, h, w = obs_seq.shape
            flat_obs = obs_seq.view(-1, c, h, w)

            with torch.no_grad():
                mu, logvar = vae.encode(flat_obs)
                z = vae.reparameterize(mu, logvar)
                z_seq = z.view(b, s, -1)

            z_input = z_seq[:, :-1, :]
            action_input = action_seq[:, :-1, :]

            z_target = z_seq[:, 1:, :]
            pi, sigma, mu_pred, _ = rnn(z_input, action_input)
            loss = rnn.mdn_loss_fn(pi, sigma, mu_pred, z_target)
        
        scaler(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)

        total_loss += loss.item()
        batch_count += 1

        if batch_idx and batch_idx % 10 == 0:
            print(f'Epoch: {epoch} | batch: {batch_idx} | loss: {loss.item():.4f}')

    avg_loss = total_loss / batch_count
    torch.save(rnn.state_dict(), f'{RNN_DIR}/rnn_{epoch}.pth')
