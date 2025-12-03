import torch  # TORCH IMPORT
import torch.nn as nn  # NN MODULE

class Decoder(nn.Module):  # DECODER CLASS
    def __init__(self, latent_dim, output_dim):  # INIT DECODER
        super().__init__()  # CALL SUPER
        self.net = nn.Sequential(  # SEQ MODULE
            nn.Linear(latent_dim, 128),  # LINEAR LAYER
            nn.ReLU(),  # ACTIVATION
            nn.Linear(128, 256),  # LINEAR LAYER
            nn.ReLU(),  # ACTIVATION
            nn.Linear(256, output_dim)  # OUTPUT LAYER
        )

    def forward(self, z):  # FORWARD PASS
        return self.net(z)  # RETURN RECONSTRUCTION
