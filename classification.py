import torch  # TORCH IMPORT
import torch.nn as nn  # NN MODULE

class StabilityClassifier(nn.Module):  # BINARY CLASSIFIER
    def __init__(self, latent_dim):  # INIT MODEL
        super().__init__()  # SUPER
        self.net = nn.Sequential(  # SEQ
            nn.Linear(latent_dim, 32),  # LINEAR
            nn.ReLU(),  # ACT
            nn.Linear(32, 1),  # OUTPUT
            nn.Sigmoid()  # PROBABILITY
        )

    def forward(self, z):  # FORWARD
        return self.net(z)  # RETURN CLASS

def classify_orbit(orbit):  # ORBIT CLASS FN
    # ORBIT SHAPE (STEPS, LATENT_DIM)
    radius = torch.norm(orbit[-1])  # FINAL NORM
    return 1 if radius < 2 else 0  # 1=STABLE, 0=UNSTABLE
