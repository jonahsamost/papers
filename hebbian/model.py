import torch
import torch.nn as nn
import torch.nn.functional as F


class KHLayer(nn.Module):
    def __init__(self, input_dim, output_dim, p_norm=2, data_sample=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p_norm = p_norm
        
        if data_sample is not None:
            self.weights = nn.Parameter(data_sample.clone() + 0.01 * torch.randn_like(data_sample), requires_grad=False)
        else:
            self.weights = nn.Parameter(torch.randn(output_dim, input_dim), requires_grad=False)

        self.normalize_weights()
        
        self.register_buffer('win_counts', torch.ones(output_dim) / output_dim)
        self.bias_strength = 4.0 * output_dim 
        self.k = 3
        self.delta = 0.0

    def normalize_weights(self):
        with torch.no_grad():
            norms = torch.norm(self.weights, p=self.p_norm, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            self.weights.data = self.weights.data / norms

    def _inner_forward(self, x):
        # Subtract mean activation to force sparsity
        threshold = x.mean(dim=1, keepdim=True)
        out = x - threshold
        
        out = F.relu(out)
        out = torch.tanh(out * 5.0) 
        return out
        
    def forward(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        out = x @ self.weights.t()
        return self._inner_forward(out)
    
    def unsupervised_update(self, inputs, lr, delta):
        bs = inputs.shape[0]
        
        inputs_centered = inputs - inputs.mean(dim=1, keepdim=True)
        inn_prod = inputs_centered @ self.weights.t()
        
        # Scale to the current batch's activity level
        current_max_values, _ = torch.max(inn_prod, dim=1) 
        adaptive_strength = current_max_values.unsqueeze(1) 
        
        # Penalty calculation
        penalty = self.bias_strength * self.win_counts.unsqueeze(0) * adaptive_strength
        biased_scores = inn_prod - penalty
        _, indices = torch.topk(biased_scores, k=self.k, dim=1)
        
        self.win_counts = 0.99 * self.win_counts
        wins_one_hot = torch.zeros_like(inn_prod)
        wins_one_hot.scatter_(1, indices, 1.0)
        self.win_counts += 0.01 * (wins_one_hot.sum(dim=0) / bs)

        g = torch.full_like(inn_prod, -delta)
        g.scatter_(1, indices, 1.0)
        
        term1 = g.t() @ inputs_centered
        term2 = torch.sum(g * inn_prod, dim=0).unsqueeze(1) * self.weights
        dw = term1 - term2
        
        self.weights.data += (lr / bs) * dw
        self.normalize_weights()
        
        with torch.no_grad():
            return self._inner_forward(inn_prod)


class KHDeep(nn.Module):
    def __init__(self, input_dim, hidden_dims: list = None):
        super().__init__()
        if not hidden_dims:
            hidden_dims = [1024, 512, 256, 128]
        hidden_dims.insert(0, input_dim)
        self.hidden_dims = hidden_dims
        self.layers = nn.Sequential(
            *[
                KHLayer(i, o)
                for i, o in zip(hidden_dims[:-1], hidden_dims[1:])
            ]
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def unsupervised_update(self, inputs, lr, delta):
        x = inputs
        for layer in self.layers:
            x = layer.unsupervised_update(x, lr, delta)


class FullNetworkDeep(nn.Module):
    def __init__(self, bio_model, n_classes=10):
        super().__init__()
        self.bio_model = bio_model
        for layer in self.bio_model.layers:
            for param in layer.parameters():
                param.requires_grad = False

        self.fc = nn.Linear(bio_model.layers[-1].output_dim, n_classes)

    def forward(self, x):
        x = self.bio_model(x)
        x = self.fc(x)
        return x


class FullNetwork(nn.Module):
    def __init__(self, bio_layer, n_classes=10):
        super().__init__()
        self.bio_layer = bio_layer
        for param in bio_layer.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(bio_layer.output_dim, n_classes)

    def forward(self, x):
        x = self.bio_layer(x)
        x = self.fc(x)
        return x