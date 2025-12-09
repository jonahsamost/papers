import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class VAE(nn.Module):
    def __init__(self, in_channels: int = 3, in_height_width: int = 96, latent_dim: int = 32, hidden_dims: list = None):
        super().__init__()
        self.expand = 4
        self.padding = 1
        self.kernel_size = 3
        self.stride = 2
        self.latent_dim = latent_dim
        self.in_height_width = in_height_width
        self.in_channels = in_channels

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256 + 32, 512 + 64]
        
        self.hidden_dims = hidden_dims

        len_hd = len(hidden_dims) - 1
        encoder_modules = []
        decoder_modules = []
        enc_in_channels = in_channels
        for i, enc_hd in enumerate(hidden_dims):
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        enc_in_channels, enc_hd,
                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
                    ),
                    nn.BatchNorm2d(enc_hd),
                    nn.LeakyReLU()
                )
            )
            enc_in_channels = enc_hd

            if i < len_hd:
                decoder_modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            hidden_dims[len_hd - i], hidden_dims[len_hd - i - 1],
                            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                            output_padding=self.padding
                        ),
                        nn.BatchNorm2d(hidden_dims[len_hd - i - 1]),
                        nn.LeakyReLU()
                    )
                )
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[0], hidden_dims[0],
                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                output_padding=self.padding
            ),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.Conv2d(
                hidden_dims[0], in_channels,
                kernel_size=self.kernel_size, stride=1, padding=self.padding
            ),
            nn.Sigmoid()
        )

        hw = math.ceil(self.in_height_width / (2 ** len(hidden_dims)))
        self.twidth = hw
        self.mult = self.twidth * self.twidth
        
        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.mult, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.mult, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.mult)
    
    def encode(self, x):
        enc = self.encoder(x)
        enc = enc.flatten(start_dim=1)
        mu = self.fc_mu(enc)
        logvar = self.fc_var(enc)
        return [mu, logvar]
    
    def decode(self, x):
        dec = self.decoder_input(x)
        dec = dec.view(-1, self.hidden_dims[-1], self.twidth, self.twidth)
        dec = self.decoder(dec)
        dec = self.final_layer(dec)
        return dec
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return [decoded, x, mu, logvar, z]
    
    def loss_fn(self, *args, **kwargs):
        recons, input, mu, logvar, _ = args
        recon_loss = F.mse_loss(recons, input)
        kld_weight = kwargs['M_N']
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = recon_loss + kld_loss * kld_weight
        return dict(
            loss=loss,
            reconstruction_loss=recon_loss.detach(),
            kld=-kld_loss.detach(),
        )

class MDN_RNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_size=256, num_guassians=8, num_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_guassians = num_guassians
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers
        )
        self.fc_pi = nn.Linear(hidden_size, num_guassians)
        self.fc_sigma = nn.Linear(hidden_size, num_guassians * latent_dim)
        self.fc_mu = nn.Linear(hidden_size, num_guassians * latent_dim)
        nn.init.constant_(self.fc_sigma.bias, -0.5)
        nn.init.constant_(self.fc_pi.bias, 0.0)
    
    def get_initial_state(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )
    
    def forward(self, z, action, hidden_state=None):
        batch_size, seq_len, _ = z.size()
        x = torch.cat([z, action], dim=2)
        lstm_out, next_hidden_state = self.lstm(x, hidden_state)
        flat_out = lstm_out.reshape(-1, self.hidden_size)

        # mdn
        pi = self.fc_pi(flat_out)
        pi = F.softmax(pi, dim=1)
        pi = pi.view(batch_size, seq_len, self.num_guassians)

        sigma = self.fc_sigma(flat_out)
        sigma = torch.exp(sigma)
        sigma = sigma.view(batch_size, seq_len, self.num_guassians, self.latent_dim)

        mu = self.fc_mu(flat_out)
        mu = mu.view(batch_size, seq_len, self.num_guassians, self.latent_dim)

        return pi, sigma, mu, next_hidden_state

    def mdn_loss_fn(self, pi, sigma, mu, target):
        target = target.unsqueeze(2)
        m_dist = torch.distributions.Normal(loc=mu, scale=sigma)
        log_probs = m_dist.log_prob(target)
        log_probs = torch.sum(log_probs, dim=3)
        log_pi = torch.log(torch.clamp(pi, min=1e-8))
        loss = -torch.logsumexp(log_pi + log_probs, dim=2)
        return torch.mean(loss)


class Controller(nn.Module):
    def __init__(self, in_features=256 + 32, out_features=3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def load_flat_params(self, flat_params):
        if isinstance(flat_params, np.ndarray):
            flat_params = torch.from_numpy(flat_params).float()
        
        flat_params = flat_params.to(next(self.parameters()).device)
        
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = flat_params[idx : idx + numel].view(p.shape)
            idx += numel
    
    def forward(self, x):
        output = self.linear(x)
        steer = torch.tanh(output[:, 0].unsqueeze(1))
        gas_brake = torch.sigmoid(output[:, 1:])
        return torch.cat([steer, gas_brake], dim=1)
