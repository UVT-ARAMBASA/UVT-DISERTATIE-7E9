# ============================= train_autoencoder.py ==========================
from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================= IMPORTS ==================================
import numpy as np  # NUMPY
import torch  # TORCH
from torch.utils.data import DataLoader, TensorDataset  # DATALOADER

from encoder import Encoder  # YOUR ENCODER
from decoder import Decoder  # YOUR DECODER
from losses import make_reconstruction_loss  # YOUR LOSS

# ================================ TRAIN LOOP =================================
def train_autoencoder(  # TRAIN AE
    X: np.ndarray | list[np.ndarray],  # TRAIN ARRAY OR LIST OF ARRAYS
    *,
    latent_dim: int,  # LATENT SIZE
    epochs: int,  # EPOCHS
    batch_size: int,  # BATCH
    lr: float,  # LR
    device: torch.device,  # DEVICE
) -> tuple[Encoder, Decoder, list[float]]:  # RETURNS MODELS + LOSSES
    if isinstance(X, np.ndarray):  # SINGLE ARRAY
        X_list = [X]  # WRAP
    else:  # MANY ARRAYS
        X_list = list(X)  # COPY

    if len(X_list) == 0:  # EMPTY
        raise ValueError("X_list IS EMPTY")  # ERROR

    in_dim = int(X_list[0].shape[1])  # INPUT DIM
    enc = Encoder(in_dim, latent_dim).to(device)  # BUILD ENCODER
    dec = Decoder(latent_dim, in_dim).to(device)  # BUILD DECODER

    loss_fn = make_reconstruction_loss(0, beta=0.01, w_pow=1.0)  # MSE MODE
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)  # OPTIM

    losses: list[float] = []  # LOSS LIST

    for e in range(epochs):  # EPOCH LOOP
        rng = np.random.default_rng(e)  # RNG
        order = rng.permutation(len(X_list))  # SHUFFLE DATA BLOCKS

        s = 0.0  # LOSS SUM
        n_batches = 0  # BATCH COUNT

        for block_id in order:  # LOOP BLOCKS
            X_block = np.asarray(X_list[int(block_id)], dtype=np.float32)  # ONE BLOCK
            X_cpu = torch.tensor(X_block, dtype=torch.float32)  # KEEP ON CPU
            loader = DataLoader(  # LOADER
                TensorDataset(X_cpu),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )

            for (b_cpu,) in loader:  # BATCH LOOP
                b = b_cpu.to(device)  # MOVE BATCH ONLY
                z = enc(b)  # ENCODE
                xh = dec(z)  # DECODE
                loss = loss_fn(xh, b)  # LOSS

                opt.zero_grad()  # ZERO
                loss.backward()  # BACKPROP
                opt.step()  # STEP

                s += float(loss.detach().cpu().item())  # ACCUM
                n_batches += 1  # COUNT

        s /= max(1, n_batches)  # MEAN
        losses.append(s)  # STORE
        print(f"EPOCH {e+1}/{epochs} LOSS={s:.6e}  ({s:.8f})")  # LOG

    return enc, dec, losses  # RETURN