import torch
import torch.nn.functional as F
from torch import nn


class FFN(nn.Module):
    def __init__(self, a, b, dropout=.1):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(a, b),
            nn.GELU(),
            nn.Linear(b, a),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fnn(x)


class AutoEncoder(nn.Module):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.config = config
        self.vocab_size = tokenizer.vocab_size

        self.emb = nn.Embedding(self.vocab_size, config.ae_hidden_size)
        d = config.ae_hidden_size
        kd = config.ae_hidden_size * config.ae_k
        l = 32 * config.ae_k

        # encoder
        self.e_ffn1 = FFN(d, d * 4, dropout=config.ae_dropout)
        self.e_l1 = nn.Linear(kd, d)
        self.e_ffn2 = FFN(d, d * 4, dropout=config.ae_dropout)
        self.e_l2 = nn.Linear(d, l * 2)

        # decoder
        self.d_l1 = nn.Linear(l, d)
        self.d_ffn1 = FFN(d, d * 4, dropout=config.ae_dropout)
        self.d_l2 = nn.Linear(d, kd)
        self.d_ffn2 = FFN(d, d * 4, dropout=config.ae_dropout)

        self.o_proj = nn.Parameter(torch.zeros(self.vocab_size))
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        bs, chunks, _ = z.size()
        k = self.config.ae_k
        d = self.config.ae_hidden_size

        out = self.d_l1(z)
        out = self.d_ffn1(out)
        out = self.d_l2(out)
        out = out.view(bs, chunks, k, d)
        out = self.d_ffn2(out)
        logits = F.linear(out, self.emb.weight, self.o_proj)
        return logits

    def encode(self, input_ids):
        k = self.config.ae_k
        d = self.config.ae_hidden_size

        b, seq_len = input_ids.size()
        out = self.emb(input_ids)
        out = out.view(b, -1, k, d)

        out = self.e_ffn1(out)
        out = out.view(b, -1, k * d)
        out = self.e_l1(out)
        out = self.e_ffn2(out)
        mu_logvar = self.e_l2(out)

        mu, logvar = mu_logvar.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, input_ids):
        z, mu, logvar = self.encode(input_ids) # [batch, hidden_size * k, 32 * k]
        logits = self.decode(z) # [batch, hidden_size * k, hidden_size * k]
        logits = logits.view(-1, self.vocab_size)
        return logits, mu, logvar



