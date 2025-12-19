import math
from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from typing import Any, Sequence
from dataclasses import dataclass


@dataclass
class TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    carry: Any


def get_optimizer(model, dataset_type="sudoku"):
    # Hyperparameters from Section 6
    if dataset_type == "arc":
        lr_body = 1e-4      # 
        lr_embed = 1e-2     # 
        weight_decay = 0.1  # 
    else:
        # Sudoku-Extreme / Maze-Hard
        lr_body = 1e-4      # 
        lr_embed = 1e-4     # Same as body
        weight_decay = 1.0  # 
    
    embed_params = []
    body_params = []
    
    for name, param in model.named_parameters():
        if "embedding" in name or "embed" in name:
            embed_params.append(param)
        else:
            body_params.append(param)

    optim_groups = [
        {"params": body_params, "lr": lr_body, "weight_decay": weight_decay},
        {"params": embed_params, "lr": lr_embed, "weight_decay": weight_decay},
    ]

    optimizer = AdamW(
        optim_groups, 
        betas=(0.9, 0.95)
    )
    
    return optimizer


def get_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


def get_iterator(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def train_batch(config, train_state, batch, global_batch_size: int):
    batch = {k: v.cuda() for k, v in batch.items()}
    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    train_state.optimizer.zero_grad()
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()
    train_state.optimizer.step()

