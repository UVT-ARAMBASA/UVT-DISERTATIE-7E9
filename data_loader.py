import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  # ENCODER H1
            nn.ReLU(),
            nn.Linear(256, 128),        # ENCODER H2
            nn.ReLU(),
            nn.Linear(128, latent_dim)  # LATENT OUTPUT
        )

    def forward(self, x):
        return self.net(x)
