import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math



class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, min_epochs=350):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, current_loss, epoch):
        if epoch < self.min_epochs:
            if current_loss + self.min_delta < self.best_loss:
                self.best_loss = current_loss
                self.counter = 0
            return False

        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



class SinusoidalNoiseEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 64, max_freq: float = 1e4):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        # Logarithmically spaced frequencies
        self.freqs = nn.Parameter(
            torch.exp(torch.linspace(math.log(1.0), math.log(max_freq), half)),requires_grad=False
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, noise_level: torch.Tensor):
        # noise_level: [batch_size, 1]
        args = noise_level * self.freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embed_dim]
        return self.proj(emb)



class PCDAE(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        x_embed_dim: int = 16,
        hidden_dim: int = 104,
        noise_embed_dim: int = 10,
    ):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, x_embed_dim),
            nn.GELU(),
            nn.Linear(x_embed_dim, x_embed_dim)
        )
        # Sinusoidal noise embedding
        self.noise_emb = SinusoidalNoiseEmbedding(noise_embed_dim)

        # Initial projection of y_noisy
        self.y_proj = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim + x_embed_dim + noise_embed_dim),
            nn.Linear(hidden_dim + x_embed_dim + noise_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, y_dim + x_dim)
        )


    def forward(self, x: torch.Tensor, y_noisy: torch.Tensor, noise_level: torch.Tensor):
        # Embeddings
        x_emb = self.x_proj(x)                        # [B, x_embed_dim]
        
        noise_emb = self.noise_emb(noise_level)       # [B, noise_embed_dim]
        h = self.y_proj(y_noisy)                      # [B, hidden_dim]
        
        # Decoder
        dec_in = torch.cat([h, x_emb, noise_emb], dim=-1)
        dec_out = self.decoder(dec_in)                # [B, x_dim + y_dim]
        
        return dec_out, h




class PCDAE_EBM(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        x_embed_dim: int = 16,
        hidden_dim: int = 104
    ):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, x_embed_dim),
            nn.GELU(),
            nn.Linear(x_embed_dim, x_embed_dim)
        )
        # Initial projection of y_noisy
        self.y_proj = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim + x_embed_dim),
            nn.Linear(hidden_dim + x_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, y_dim + x_dim)
        )


    def forward(self, x: torch.Tensor, y_noisy: torch.Tensor):
        # Embeddings
        x_emb = self.x_proj(x)                        # [B, x_embed_dim]
        h = self.y_proj(y_noisy)                      # [B, hidden_dim]
        
        # Decoder
        dec_in = torch.cat([h, x_emb], dim=-1)
        dec_out = self.decoder(dec_in)                # [B, x_dim + y_dim]
        
        return dec_out, h






class Regressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        activation_fn = nn.LeakyReLU()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(activation_fn)
        
        self.init_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)


