# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# train_autoencoder.py #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# IMPORTS #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
import numpy as np  # NUMPY
import torch  # TORCH
import torch.nn as nn  # NEURAL MODULES
from torch.utils.data import DataLoader, TensorDataset  # DATALOADER

from losses import make_reconstruction_loss, koopman_ae_loss  # LOSS FUNCTIONS

import defines as D # FORGOT DEFINES

from encoder import Encoder  # ENCODER
from decoder import Decoder  # DECODER
#from losses import make_reconstruction_loss  # LOSS

# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# TRAIN LOOP #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
def train_autoencoder(  # TRAIN AE + DMD-LIKE PIPELINE
    X1: np.ndarray | list[np.ndarray],  # LEFT STATE OR LIST
    X2: np.ndarray | list[np.ndarray],  # RIGHT STATE OR LIST
    *,
    latent_dim: int,  # LATENT SIZE
    epochs: int,  # EPOCHS
    batch_size: int,  # BATCH
    lr: float,  # LR
    device: torch.device,  # DEVICE
) -> tuple[Encoder, Decoder, list[float]]:  # RETURNS MODELS + LOSSES
    if isinstance(X1, np.ndarray):  # SINGLE ARRAY
        X1_list = [X1]  # WRAP
    else:  # MANY ARRAYS
        X1_list = list(X1)  # COPY

    if isinstance(X2, np.ndarray):  # SINGLE ARRAY
        X2_list = [X2]  # WRAP
    else:  # MANY ARRAYS
        X2_list = list(X2)  # COPY

    if len(X1_list) == 0 or len(X2_list) == 0:  # EMPTY
        raise ValueError("X1_list OR X2_list IS EMPTY")  # ERROR
    if len(X1_list) != len(X2_list):  # SIZE MISMATCH
        raise ValueError("X1_list AND X2_list MUST HAVE SAME LENGTH")  # ERROR

    in_dim = int(X1_list[0].shape[1])  # INPUT DIM
    enc = Encoder(in_dim, latent_dim).to(device)  # BUILD ENCODER
    dec = Decoder(latent_dim, in_dim).to(device)  # BUILD DECODER

    loss_fn = make_reconstruction_loss(0, beta=0.01, w_pow=1.0)  # MSE MODE
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)  # OPTIM

    losses: list[float] = []  # LOSS LIST

    for e in range(epochs):  # EPOCH LOOP
        rng = np.random.default_rng(e)  # RNG
        order = rng.permutation(len(X1_list))  # SHUFFLE DATA BLOCKS

        s = 0.0  # LOSS SUM
        n_batches = 0  # BATCH COUNT

        for block_id in order:  # LOOP BLOCKS
            X1_block = np.asarray(X1_list[int(block_id)], dtype=np.float32)  # ONE BLOCK
            X2_block = np.asarray(X2_list[int(block_id)], dtype=np.float32)  # ONE BLOCK

            X1_cpu = torch.tensor(X1_block, dtype=torch.float32)  # KEEP ON CPU
            X2_cpu = torch.tensor(X2_block, dtype=torch.float32)  # KEEP ON CPU

            loader = DataLoader(  # LOADER
                TensorDataset(X1_cpu, X2_cpu),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )

            for b1_cpu, b2_cpu in loader:  # BATCH LOOP
                b1 = b1_cpu.to(device)  # MOVE BATCH ONLY
                b2 = b2_cpu.to(device)  # MOVE BATCH ONLY

                #base_loss = make_reconstruction_loss(loss_mode=0)  # MSE

                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# AE ENCODE - BEGIN #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
                z1 = enc(b1)  # ENCODE X1
                z2 = enc(b2)  # ENCODE X2
                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# AE ENCODE - END #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#

                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# AE RECONSTRUCT - BEGIN #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
                x1_rec = dec(z1)  # RECONSTRUCT X1
                x2_rec = dec(z2)  # RECONSTRUCT X2
                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# AE RECONSTRUCT - END #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#

                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# COMPUTE KOOPMAN AE LOSS - BEGIN #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
                loss, loss_info = koopman_ae_loss(  # COMPUTE FULL AE + DMD LOSS
                    x1=b1,  # TRUE X1
                    x2=b2,  # TRUE X2
                    z1=z1,  # LATENT X1
                    z2=z2,  # LATENT X2
                    x1_rec=x1_rec,  # RECONSTRUCTED X1
                    x2_rec=x2_rec,  # RECONSTRUCTED X2
                    decoder=dec,  # DECODER
                    base_loss=loss_fn,  # BASIC LOSS FUNCTION
                    alpha_rec=D.AE_REC_WEIGHT,  # RECONSTRUCTION WEIGHT
                    alpha_lin=D.AE_LATENT_WEIGHT,  # LATENT DMD WEIGHT
                    alpha_pred=D.AE_PRED_WEIGHT,  # DECODED PREDICTION WEIGHT
                    ridge=D.DMD_RIDGE,  # SAFE DMD FIT
                    # VALUES WERE HARDCODED IN V30. 'TWAS A MISTAKE ON MY PART
                )
                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# COMPUTE KOOPMAN AE LOSS - END #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
                
                # OPTIMIZER STEP
                # BEGIN #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
                opt.zero_grad()  # ZERO
                loss.backward()  # BACKPROP
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()),
                                               max_norm=1.0) # FIXES LOSS SPIKE AT SYMPTOM LEVEL (MAYBE REMOVE) - UNSURE IF THE RIGHT APPROACH
                opt.step()  # STEP
                
                # OPTIMIZER STEP
                # END #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#

                s += float(loss.detach().cpu().item())  # ACCUM
                n_batches += 1  # COUNT

        s /= max(1, n_batches)  # MEAN
        losses.append(s)  # STORE
        print(f"EPOCH {e+1}/{epochs} LOSS={s:.6e}  ({s:.8f})")  # LOG

    return enc, dec, losses  # RETURN