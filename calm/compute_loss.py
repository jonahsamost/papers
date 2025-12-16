import torch.nn as nn
import torch

def compute_loss(logits, targets, mu, log_var, kl_weight=0.001, kl_threshold=0.5):
    # Logits: [Batch * Seq_Len, Vocab]
    # Targets: [Batch, Seq_Len] -> Flatten to match logits
    loss_fct = nn.CrossEntropyLoss()
    rec_loss = loss_fct(logits, targets.view(-1))
    
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_clipped = torch.maximum(kl, torch.tensor(kl_threshold, device=kl.device))
    kl_loss = torch.sum(kl_clipped) / mu.size(0)
    total_loss = rec_loss + (kl_weight * kl_loss)
    return total_loss, rec_loss, kl_loss