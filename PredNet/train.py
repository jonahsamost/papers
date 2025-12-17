import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from torch.amp import autocast

from model import PredNet
from dataloader import PredNetDataset


pn_dataset = PredNetDataset('kitti_data/X_train.hkl', timesteps=10)
dataloader = DataLoader(pn_dataset, batch_size=16)

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
model = PredNet().to(device)
LR = .001
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(.9, .999))
def lambda_func(step, warmup_steps=20_000):
    return LR if step < warmup_steps else LR / 10.0
scheduler = LambdaLR(optimizer, lambda_func)
autocast_ctx = autocast(device_type=device_type , dtype=torch.bfloat16)

def dl_iter(loader):
    while True:
        for batch in loader:
            yield batch

model.train()
train_iter = dl_iter(dataloader)
for step in range(pn_dataset.steps):
    data = next(train_iter).to(device)
    with autocast_ctx:
        loss = model(data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 100 == 0:
        print(f'Step: {step} | loss: {loss.item():.4f}')


