from transformers import LlamaConfig, LlamaModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, model_samples, target_samples):
        b, s, n, d = model_samples.shape
        m = target_samples.shape[2]

        preds = model_samples.view(-1, n, d)   # [BS, N, D]
        targets = target_samples.view(-1, m, d) # [BS, M, D]

        dists_fidelity = torch.cdist(preds, targets, p=2)
        if self.alpha != 1.0:
            dists_fidelity = dists_fidelity.pow(self.alpha)
        term1 = dists_fidelity.mean(dim=(1, 2)) # [BS]

        dists_diversity = torch.cdist(preds, preds, p=2)
        
        if self.alpha != 1.0:
            dists_diversity = dists_diversity.pow(self.alpha)
            
        sum_diversity = dists_diversity.sum(dim=(1, 2))
        term2 = sum_diversity / (n * (n - 1)) # [BS]
        
        loss = 2 * term1 - term2
        return loss.mean()


class EnergyBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ff1 = nn.Linear(hidden_size, hidden_size)
        self.ff2 = nn.Linear(hidden_size, hidden_size)
        self.ff3 = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, noise, lhs):
        out = self.ff1(lhs) + self.ff2(noise)
        out = self.ff3(out)
        gate, val = out.chunk(2, dim=-1)
        out = val * F.silu(gate)
        return noise + out


class CalmModel(nn.Module):
    def __init__(self, autoencoder, tokenizer, config):
        super().__init__()
        self.autoencoder = autoencoder
        self.tokenizer = tokenizer
        self.config = config
        ll_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.int_size,
            num_hidden_layers=config.layers,
            num_attention_heads=config.hidden_size // 64,
            max_position_embeddings=2048,
            use_cache=False,
        )
        self.backbone = LlamaModel(ll_config)

        sz = self.config.hidden_size * self.config.ae_k
        self.ff = nn.Sequential(
            nn.Linear(sz, sz // 2),
            nn.GELU(),
            nn.Linear(sz // 2, self.config.hidden_size),
        )
        hs = self.config.hidden_size
        self.noise_proj = nn.Linear(hs, hs)
        num_blocks = self.config.layers // 4
        self.energy_blocks = nn.ModuleList([
            EnergyBlock(hs) for _ in range(num_blocks)
        ])
        self.down_proj = nn.Linear(hs, 32 * self.config.ae_k )
    
    def forward(self, input_ids, num_model_samples=8):
        batch_size, seq_len = input_ids.shape
        k = self.config.ae_k
        token_embeds = self.backbone.embed_tokens(input_ids)
        compressed_input = token_embeds.view(batch_size, -1, k * self.config.hidden_size)
        input_embeds = self.ff(compressed_input)
        outputs = self.backbone(inputs_embeds=input_embeds)
        lhs = outputs.last_hidden_state # [batch, seq_len, hidden_size]

        out = self.generate_model_samples(lhs, num_samples=num_model_samples)
        return out
    
    def generate(self, input_ids, max_new_tokens=50):
        self.eval()

        b, seq_len = input_ids.shape
        k = self.config.ae_k
        remainder = seq_len % k
        
        if remainder != 0:
            pad_len = k - remainder
            pad_id = self.tokenizer.eos_token_id or 0
            padding = torch.full((b, pad_len), pad_id, device=input_ids.device, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=1)

        steps = max_new_tokens // k
        for _ in range(steps):
            out = self.forward(input_ids, num_model_samples=1)
            out = out[:, -1, 0, :].unsqueeze(1)
            out = self.autoencoder.decode(out)
            out = out.argmax(dim=-1).view(b, k)
            input_ids = torch.cat([input_ids, out], dim=1)

        tokens = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return tokens
    
    def generate_model_samples(self, h, num_samples=8):
        b, s, d = h.shape
        
        h_expanded = h.unsqueeze(2).expand(b, s, num_samples, d)
        epsilon = torch.rand_like(h_expanded) - 0.5
        
        for block in self.energy_blocks:
            epsilon = block(epsilon, h_expanded)
            
        z_preds = self.down_proj(epsilon)
        return z_preds

    def sample_targets(self, mu, log_var, num_samples=100):
        b, s, d = mu.shape
        
        mu = mu.unsqueeze(2).expand(b, s, num_samples, d)
        log_var = log_var.unsqueeze(2).expand(b, s, num_samples, d)
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        z_targets = mu + eps * std
        return z_targets


