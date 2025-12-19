import math
from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from typing import Any, Sequence
from dataclasses import dataclass
from torch.amp import GradScaler, autocast



@dataclass
class TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    autocast_ctx: torch.amp.autocast_mode.autocast
    carry: Any
    step: int
    total_steps: int
    epoch: int
    total_epochs: int


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
            train_state.carry = train_state.model.initial_carry(batch)

    # Memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**2  # MB

    train_state.optimizer.zero_grad()
    with train_state.autocast_ctx:
        train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()
    train_state.optimizer.step()
    train_state.scheduler.step()
    train_state.step += 1

    # Memory stats after training step
    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        mem_peak_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB

    # Print memory stats (change condition to see every step: train_state.step % 1 == 0)
    if train_state.step % 100 == 0 or train_state.step == 1:
        current_lr = train_state.scheduler.get_last_lr()[0]
        total = train_state.total_steps
        epoch = train_state.epoch
        total_epochs = train_state.total_epochs
        print(f"Step {train_state.step}/{total}, Epoch {epoch}/{total_epochs}, LR: {current_lr:.2e}")
        
        # Print metrics
        count = metrics.get("count", torch.tensor(1.0)).item()
        m_str = []
        for k, v in metrics.items():
            if k == "count": continue
            val = v.item() if isinstance(v, torch.Tensor) else v
            if "accuracy" in k:
                m_str.append(f"{k}: {val/max(1, count):.4f}")
            elif "loss" in k:
                # Loss is already scaled by loss_divisor in some cases or just summed.
                # Here we just show the raw metric value from the dictionary.
                m_str.append(f"{k}: {val:.4f}")
            else:
                m_str.append(f"{k}: {val:.2f}")
        print(f"  Metrics: {', '.join(m_str)}")

        if torch.cuda.is_available():
            peak_pct = (mem_peak / mem_total) * 100
            print(f"  Memory: allocated={mem_after:.1f}MB, peak={mem_peak:.1f}MB ({peak_pct:.1f}%), reserved={mem_reserved:.1f}MB, peak_reserved={mem_peak_reserved:.1f}MB, total={mem_total:.1f}MB")

