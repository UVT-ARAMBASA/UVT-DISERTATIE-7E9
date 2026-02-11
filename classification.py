import torch  # TORCH IMPORT
import torch.nn as nn  # NN MODULE

class StabilityClassifier(nn.Module):  # CLASSIFIER MODEL
    def __init__(self, latent_dim):  # INIT
        super().__init__()  # SUPER
        self.net = nn.Sequential(  # SEQ
            nn.Linear(latent_dim, 32),  # FC
            nn.ReLU(),  # RELU
            nn.Linear(32, 1),  # FC
            nn.Sigmoid()  # SIGMOID
        )  # END SEQ

    def forward(self, z):  # FORWARD
        return self.net(z)  # PROB OUT

def classify_orbit(orbit, radius=2.0):  # ORBIT CHECK
    orbit = orbit.detach()  # NO GRAD
    norms = torch.norm(orbit, dim=-1)  # NORM PER STEP
    mx = torch.max(norms)  # MAX NORM
    return 1 if float(mx) < float(radius) else 0  # 1 STABLE, 0 UNSTABLE
