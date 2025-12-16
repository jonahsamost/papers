import torch
import torch.nn as nn
from dataclasses import dataclass
from torch import optim
from huggingface_hub import login
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from autoencoder import AutoEncoder
from dataloader import CalmDataset
from compute_loss import compute_loss
import os

login(os.environ['HF_TOKEN'])

@dataclass
class Config:
    layers: int = 12
    hidden_size: int = 768
    int_size: int = 2048
    ae_k: int = 4
    ae_hidden_size: int = 512
    ae_dropout: float = 0.0

config = Config()
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
dataset = CalmDataset(tokenizer, k=config.ae_k, steps=2048)

tokens_per_seq = config.ae_k * 2048
target_batch_tokens = 512_000
micro_batch_size = 1
tokens_per_micro_batch = micro_batch_size * tokens_per_seq
grad_accum_steps = max(1, target_batch_tokens // tokens_per_micro_batch)

dataloader = DataLoader(dataset, batch_size=micro_batch_size)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
model = AutoEncoder(tokenizer, config).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), eps=1e-8, weight_decay=.1)
def get_scheduler(optimizer, warmup_steps=2_000):
    def fn(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / warmup_steps
        return 1.0
    return LambdaLR(optimizer, fn)


def infinite_iter(loader):
    while True:
        for batch in loader:
            yield batch

autocast_ctx = autocast(device_type=device_type , dtype=torch.bfloat16)
scheduler = get_scheduler(optimizer)
train_iterator = infinite_iter(dataloader)

steps = 30_000
for step in range(steps):
    step_loss = 0.0
    for _ in range(grad_accum_steps):
        input_ids = next(train_iterator)
        input_ids = input_ids.to(device)
        with autocast_ctx:
            logits, mu, logvar = model(input_ids)
            loss, rec, kl = compute_loss(logits, input_ids, mu, logvar)
            loss = loss / grad_accum_steps
        loss.backward()
        step_loss += loss.item()
        del logits
        del mu
        del logvar

    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % 50 == 0:
        print(f"Step {step}/{steps} | Loss: {step_loss:.4f} | KL: {kl.item():.4f}")

