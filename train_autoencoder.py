# ============================= train_autoencoder.py ==========================
from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================= IMPORTS ==================================
import numpy as np  # NUMPY
import torch  # TORCH
from torch.utils.data import DataLoader, TensorDataset  # DATALOADER

from encoder import Encoder  # YOUR ENCODER
from decoder import Decoder  # YOUR DECODER
from losses import make_reconstruction_loss  # YOUR LOSS
from utils import to_tensor  # YOUR TENSOR HELPER

# ================================ TRAIN LOOP =================================
def train_autoencoder(  # TRAIN AE
    X: np.ndarray,  # TRAIN ARRAY
    *,
    latent_dim: int,  # LATENT SIZE
    epochs: int,  # EPOCHS
    batch_size: int,  # BATCH
    lr: float,  # LR
    device: torch.device,  # DEVICE
) -> tuple[Encoder, Decoder, list[float]]:  # RETURNS MODELS + LOSSES
    X_t = to_tensor(X, device)  # TO TENSOR
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True, drop_last=True)  # LOADER

    in_dim = int(X.shape[1])  # INPUT DIM
    enc = Encoder(in_dim, latent_dim).to(device)  # BUILD ENCODER
    dec = Decoder(latent_dim, in_dim).to(device)  # BUILD DECODER

    loss_fn = make_reconstruction_loss(0, beta=0.01, w_pow=1.0)  # MSE MODE
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)  # OPTIM

    losses: list[float] = []  # LOSS LIST

    for e in range(epochs):  # EPOCH LOOP
        s = 0.0  # SUM
        for (b,) in loader:  # BATCH LOOP
            z = enc(b)  # ENCODE
            xh = dec(z)  # DECODE
            loss = loss_fn(xh, b)  # LOSS
            opt.zero_grad()  # ZERO
            loss.backward()  # BACKPROP
            opt.step()  # STEP
            s += float(loss.detach().cpu().item())  # ACCUM

        s /= max(1, len(loader))  # MEAN
        losses.append(s)  # STORE
        print(f"EPOCH {e+1}/{epochs} LOSS={s:.6f}")  # PRINT

    return enc, dec, losses  # RETURN
