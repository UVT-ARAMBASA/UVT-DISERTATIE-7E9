# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# losses.py #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# IMPORTS #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
import torch  # TORCH
import torch.nn.functional as F  # TORCH FUNCTIONAL LOSSES
from tensor_diagnostics import check_tensor
import defines


# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# BASIC LOSS FACTORY #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
def make_reconstruction_loss(  # MAKE BASIC RECONSTRUCTION LOSS
    loss_mode: int = 0,  # 0=MSE, 1=MAE, 2=HUBER, 3=WEIGHTED_MSE
    beta: float = 0.05,  # HUBER BETA
    eps: float = 1e-6,  # EPSILON FOR SAFE DIVISION
    w_pow: float = 1.0,  # WEIGHT POWER
):
    if loss_mode == 0:  # MSE LOSS
        def loss_fn(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # LOSS FUNCTION
            return torch.mean((x_hat - x) ** 2)  # RETURN MSE

        return loss_fn  # RETURN FUNCTION

    if loss_mode == 1:  # MAE LOSS
        def loss_fn(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # LOSS FUNCTION
            return torch.mean(torch.abs(x_hat - x))  # RETURN MAE

        return loss_fn  # RETURN FUNCTION

    if loss_mode == 2:  # HUBER LOSS
        def loss_fn(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # LOSS FUNCTION
            return F.smooth_l1_loss(x_hat, x, beta=beta)  # RETURN HUBER

        return loss_fn  # RETURN FUNCTION

    if loss_mode == 3:  # WEIGHTED MSE LOSS
        def loss_fn(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # LOSS FUNCTION
            weights = 1.0 / (torch.abs(x) ** w_pow + eps)  # SMALLER WEIGHT FOR HUGE VALUES
            return torch.mean(weights * (x_hat - x) ** 2)  # RETURN WEIGHTED MSE

        return loss_fn  # RETURN FUNCTION

    raise ValueError(f"UNKNOWN LOSS MODE: {loss_mode}")  # INVALID LOSS MODE


# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# DMD FIT INSIDE LOSS #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
def fit_batch_dmd_matrix_OLD(  # FIT SMALL DMD MATRIX ON CURRENT BATCH
    z1: torch.Tensor,  # LATENT STATES AT TIME T
    z2: torch.Tensor,  # LATENT STATES AT TIME T+1
    ridge: float = 1e-6,  # RIDGE REGULARISATION
) -> torch.Tensor:
    latent_dim = z1.shape[1]  # LATENT DIMENSION

    eye = torch.eye(  # IDENTITY MATRIX
        latent_dim,  # SIZE
        dtype=z1.dtype,  # SAME TYPE
        device=z1.device,  # SAME DEVICE
    )

    G = z1.T @ z1  # GRAM MATRIX
    H = z1.T @ z2  # CROSS MATRIX

    A_t = torch.linalg.solve(G + ridge * eye, H)  # SOLVE Z1 @ A_T = Z2
    A = A_t.T  # TRANSPOSE SO WE CAN USE z @ A.T

    return A  # RETURN DMD MATRIX

def fit_batch_dmd_matrix_NEW(
        z1: torch.Tensor, # LATENT STATES AT TIME T
        z2: torch.Tensor, # LATENT STATES AT TIME T+1
        ridge: float = defines.DMD_RIDGE, #RIDGE REGULARISATION FROM DEFINES
) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(z1, full_matrices=False) # SVD OF SNAPSHOTS
    S_inv = S / (S * S + ridge) #DAMPED 1/sigma (REGULARISED PINV)
    z1_pinv = (Vh.transpose(-1,-2) * S_inv) @ U.transpose(-1,-2)  # (dim, N)
    A_t = z1_pinv @ z2 # (dimension, dimension) = z1hat + z2
    return A_t.transpose(-1,-2)

def fit_batch_dmd_matrix__CU_CHECKER(
        z1: torch.Tensor, # LATENT STATES AT TIME T
        z2: torch.Tensor, # LATENT STATES AT TIME T+1
        ridge: float = defines.DMD_RIDGE, #RIDGE REGULARISATION FROM DEFINES
) -> torch.Tensor:
    check_tensor(z1, name="z1")  # <-- one line, prints right before the crash poin
    U, S, Vh = torch.linalg.svd(z1, full_matrices=False) # SVD OF SNAPSHOTS
    S_inv = S / (S * S + ridge) #DAMPED 1/sigma (REGULARISED PINV)
    z1_pinv = (Vh.transpose(-1,-2) * S_inv) @ U.transpose(-1,-2)  # (dim, N)
    A_t = z1_pinv @ z2 # (dimension, dimension) = z1hat + z2
    return A_t.transpose(-1,-2)

def fit_batch_dmd_matrix(
        z1: torch.Tensor,
        z2: torch.Tensor,
        ridge: float = defines.DMD_RIDGE,
) -> torch.Tensor:
    latent_dim = z1.shape[1]
    eye = torch.eye(latent_dim, dtype=z1.dtype, device=z1.device)
    G = z1.T @ z1
    H = z1.T @ z2
    A_t = torch.linalg.solve(G + ridge * eye, H)
    return A_t.T
# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# KOOPMAN AE LOSS #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
def koopman_ae_loss(  # AUTOENCODER + LATENT DMD LOSS
    x1: torch.Tensor,  # TRUE STATE AT TIME T
    x2: torch.Tensor,  # TRUE STATE AT TIME T+1
    z1: torch.Tensor,  # ENCODED STATE AT TIME T
    z2: torch.Tensor,  # ENCODED STATE AT TIME T+1
    x1_rec: torch.Tensor,  # RECONSTRUCTED X1
    x2_rec: torch.Tensor,  # RECONSTRUCTED X2
    decoder,  # DECODER MODEL
    base_loss,  # MSE / MAE / HUBER / WEIGHTED MSE
    alpha_rec: float = 0.5,  # RECONSTRUCTION WEIGHT
    alpha_lin: float = 0.5,  # LATENT DMD WEIGHT
    alpha_pred: float = 2.0,  # DECODED PREDICTION WEIGHT
    ridge: float = 1e-6,  # DMD RIDGE
):
    A = fit_batch_dmd_matrix(z1, z2, ridge=ridge)  # FIT BATCH DMD MATRIX

    z2_pred = z1 @ A.T  # PREDICT NEXT LATENT STATE
    x2_pred = decoder(z2_pred)  # DECODE PREDICTED NEXT STATE

    loss_rec = base_loss(x1_rec, x1) + base_loss(x2_rec, x2)  # AE RECONSTRUCTION LOSS
    loss_lin = base_loss(z2_pred, z2)  # LATENT LINEAR DMD LOSS
    loss_pred = base_loss(x2_pred, x2)  # FINAL PREDICTION LOSS

    loss_total = (  # TOTAL LOSS
        alpha_rec * loss_rec  # ADD RECONSTRUCTION
        + alpha_lin * loss_lin  # ADD LATENT LINEARITY
        + alpha_pred * loss_pred  # ADD FINAL PREDICTION
    )

    loss_info = {  # DICTIONARY FOR PRINTING
        "total": float(loss_total.detach().cpu()),  # TOTAL LOSS
        "rec": float(loss_rec.detach().cpu()),  # RECONSTRUCTION LOSS
        "lin": float(loss_lin.detach().cpu()),  # LATENT LINEAR LOSS
        "pred": float(loss_pred.detach().cpu()),  # PREDICTION LOSS
    }

    return loss_total, loss_info  # RETURN LOSS AND INFO