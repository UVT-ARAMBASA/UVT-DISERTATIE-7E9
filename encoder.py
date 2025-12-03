import torch  # TORCH IMPORT
import torch.nn as nn  # NN MODULE

class Encoder(nn.Module):  # ENCODER CLASS
    def __init__(self, input_dim, latent_dim):  # INIT ENCODER
        super().__init__()  # CALL SUPER
        self.net = nn.Sequential(  # SEQ MODULE
            nn.Linear(input_dim, 256),  # LINEAR LAYER
            nn.ReLU(),  # ACTIVATION
            nn.Linear(256, 128),  # LINEAR LAYER
            nn.ReLU(),  # ACTIVATION
            nn.Linear(128, latent_dim)  # LATENT OUTPUT
        )

    def forward(self, x):  # FORWARD PASS
        return self.net(x)  # RETURN LATENT
