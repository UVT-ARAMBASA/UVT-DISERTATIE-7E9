# =========================== eval_matrix_dmd_ae.py ===========================
from __future__ import annotations  # TYPE HINTS

# ================================= IMPORTS ==================================
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH
from PIL import Image  # PIL
import matplotlib.pyplot as plt  # PLOT

from utils import to_tensor  # HELPER
from mandelbrot_reconstruct import reconstruct_final_snapshot  # MODEL ROLLOUT

# ============================== LOSS CURVE ===================================
def save_loss_curve(losses: list[float], out_png: str | Path, title: str) -> str:  # SAVE LOSS
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    plt.figure()  # FIG
    plt.plot(np.arange(1, len(losses) + 1), np.asarray(losses, dtype=np.float32))  # CURVE
    plt.xlabel("Epoch")  # X
    plt.ylabel("Loss")  # Y
    plt.title(title)  # TITLE
    plt.tight_layout()  # TIGHT
    plt.savefig(out_png, dpi=200)  # SAVE
    plt.close()  # CLOSE
    return str(out_png)  # RETURN

# =============================== METRICS =====================================
@torch.no_grad()  # NO GRAD
def autoencoder_reconstruction_metrics(  # AE METRICS
    encoder,  # ENC
    decoder,  # DEC
    X: np.ndarray,  # DATA
    device: torch.device,  # DEVICE
) -> dict:
    x = to_tensor(X, device)  # TO TENSOR
    xh = decoder(encoder(x))  # RECON
    xh_np = xh.detach().cpu().numpy().astype(np.float32)  # CPU

    err = xh_np - X.astype(np.float32)  # ERR
    mse = float(np.mean(err * err))  # MSE
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(X), 1e-12))  # REL
    fit = float(1.0 - rel_l2)  # FIT
    return {"ae_mse": mse, "ae_rel_l2": rel_l2, "ae_fit": fit}  # RETURN

@torch.no_grad()  # NO GRAD
def dmd_one_step_metrics(  # DMD ONE STEP
    encoder,  # ENC
    decoder,  # DEC
    dmd,  # DMD
    X1: np.ndarray,  # LEFT
    X2: np.ndarray,  # RIGHT
    device: torch.device,  # DEVICE
) -> dict:
    x1 = to_tensor(X1, device)  # LEFT
    z1 = encoder(x1)  # ENC
    z2h = dmd.predict(z1, steps=1)[-1]  # ONE STEP
    x2h = decoder(z2h)  # DEC

    x2h_np = x2h.detach().cpu().numpy().astype(np.float32)  # CPU
    err = x2h_np - X2.astype(np.float32)  # ERR
    mse = float(np.mean(err * err))  # MSE
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(X2), 1e-12))  # REL
    fit = float(1.0 - rel_l2)  # FIT
    return {"dmd_mse": mse, "dmd_rel_l2": rel_l2, "dmd_fit": fit}  # RETURN

# ============================== MASK SAVE ====================================
def save_ground_truth_final_mask(  # GT FINAL MASK
    td,  # TRAINING DATA
    escape_r: float,  # ESCAPE
    out_png: str | Path,  # PATH
    scale: int = 64,  # UPSCALE FACTOR
) -> str:
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    feat_dim = int(td.X_grid.shape[-1])  # FEAT DIM
    d = (feat_dim - 2) // 2  # STATE DIM
    r2 = float(escape_r) * float(escape_r)  # R2

    zr = td.X_grid[-1, :, :, 0:d]  # FINAL RE
    zi = td.X_grid[-1, :, :, d:2 * d]  # FINAL IM
    comp_mag2 = zr * zr + zi * zi  # COMP MAG2
    max_mag2 = np.max(comp_mag2, axis=-1)  # MAX MAG2
    mask = max_mag2 < r2  # BOUNDED

    img = (mask.astype(np.uint8) * 255)  # BW
    pil_img = Image.fromarray(img, mode="L")  # MAKE IMAGE
    pil_img = pil_img.resize(  # UPSCALE
        (img.shape[1] * int(scale), img.shape[0] * int(scale)),  # NEW SIZE
        resample=Image.NEAREST,  # KEEP PIXELS SHARP
    )
    pil_img.save(out_png)  # SAVE
    return str(out_png)  # RETURN

@torch.no_grad()  # NO GRAD
def save_predicted_final_mask(  # MODEL FINAL MASK
    td,  # TRAINING DATA
    encoder,  # ENC
    decoder,  # DEC
    dmd,  # DMD
    device: torch.device,  # DEVICE
    escape_r: float,  # ESCAPE
    out_png: str | Path,  # PATH
    scale: int = 64,  # UPSCALE FACTOR
) -> str:
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    feat_dim = int(td.X_grid.shape[-1])  # FEAT DIM
    d = (feat_dim - 2) // 2  # STATE DIM
    T = int(td.X_grid.shape[0])  # STEPS

    C = td.X_grid[0, :, :, 2 * d:2 * d + 2].reshape(-1, 2).astype(np.float32)  # C GRID
    grid_n_y = int(td.X_grid.shape[1])  # H
    grid_n_x = int(td.X_grid.shape[2])  # W

    if grid_n_y != grid_n_x:  # SAFE
        raise ValueError("EXPECTED SQUARE GRID FOR VISUAL COMPARISON")  # ERROR

    Z_final = reconstruct_final_snapshot(  # ROLLOUT
        encoder=encoder,  # ENC
        decoder=decoder,  # DEC
        dmd=dmd,  # DMD
        C=C,  # GRID
        grid_n=grid_n_x,  # RES
        steps=T,  # SAME STEPS
        escape_r=escape_r,  # CLAMP
        device=device,  # DEVICE
        batch_size=50000,  # BATCH
        state_dim=d,  # STATE DIM
        feat_dim=feat_dim,  # FEAT DIM
    )

    zr = Z_final[..., 0:d].astype(np.float32, copy=False)  # ALL RE
    zi = Z_final[..., d:2 * d].astype(np.float32, copy=False)  # ALL IM
    comp_mag2 = zr * zr + zi * zi  # ALL MAG2
    max_mag2 = np.max(comp_mag2, axis=-1)  # MAX OVER COMPONENTS
    mask = max_mag2 < float(escape_r) * float(escape_r)  # BOUNDED

    img = (mask.astype(np.uint8) * 255)  # BW
    pil_img = Image.fromarray(img, mode="L")  # MAKE IMAGE
    pil_img = pil_img.resize(  # UPSCALE
        (img.shape[1] * int(scale), img.shape[0] * int(scale)),  # NEW SIZE
        resample=Image.NEAREST,  # KEEP PIXELS SHARP
    )
    pil_img.save(out_png)  # SAVE
    return str(out_png)  # RETURN


def save_ground_truth_escape_iters(td, escape_r: float, out_png: str | Path) -> str:  # TRUE ESCAPE IMAGE
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    feat_dim = int(td.X_grid.shape[-1])  # FEAT DIM
    d = (feat_dim - 2) // 2  # STATE DIM
    T = int(td.X_grid.shape[0])  # STEPS
    r2 = float(escape_r) * float(escape_r)  # R2

    zr = td.X_grid[:, :, :, 0:d]  # ALL RE
    zi = td.X_grid[:, :, :, d:2 * d]  # ALL IM
    comp_mag2 = zr * zr + zi * zi  # MAG2
    max_mag2 = np.max(comp_mag2, axis=-1)  # MAX OVER COMPONENTS
    escaped = max_mag2 >= r2  # ESCAPED AFTER CLAMP TOO

    first_escape = np.argmax(escaped, axis=0).astype(np.int32) + 1  # FIRST ESC
    never_escaped = ~np.any(escaped, axis=0)  # NEVER ESC
    first_escape[never_escaped] = T  # STABLE

    img = (255.0 * (1.0 - first_escape.astype(np.float32) / float(T))).clip(0, 255).astype(np.uint8)  # GRAY
    Image.fromarray(img, mode="L").save(out_png)  # SAVE
    return str(out_png)  # RETURN