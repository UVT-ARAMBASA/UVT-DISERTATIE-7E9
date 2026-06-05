# =========================== eval_matrix_dmd_ae.py ===========================
from __future__ import annotations  # TYPE HINTS

# ================================= IMPORTS ==================================
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH
from PIL import Image  # PIL
import matplotlib.pyplot as plt  # PLOT

from utils import to_tensor  # HELPER

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
def autoencoder_reconstruction_metrics(encoder, decoder, X: np.ndarray, device: torch.device) -> dict:  # AE METRICS
    x = to_tensor(X, device)  # TO TENSOR
    xh = decoder(encoder(x))  # RECON
    xh_np = xh.detach().cpu().numpy().astype(np.float32)  # CPU

    err = xh_np - X.astype(np.float32)  # ERR
    mse = float(np.mean(err * err))  # MSE
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(X), 1e-12))  # REL
    fit = float(1.0 - rel_l2)  # FIT
    return {"ae_mse": mse, "ae_rel_l2": rel_l2, "ae_fit": fit}  # RETURN

@torch.no_grad()  # NO GRAD
def dmd_one_step_metrics(encoder, decoder, dmd, X1: np.ndarray, X2: np.ndarray, device: torch.device) -> dict:  # DMD ONE STEP
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
def save_ground_truth_final_mask(td, escape_r: float, out_png: str | Path, scale: int = 64) -> str:  # GT FINAL MASK
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    feat_dim = int(td.X_grid.shape[-1])  # FEAT DIM
    d = (feat_dim - 2) // 2  # STATE DIM
    r2 = float(escape_r) * float(escape_r)  # R2

    zr = td.X_grid[-1, :, :, 0:d].astype(np.float32, copy=False)  # FINAL RE
    zi = td.X_grid[-1, :, :, d:2 * d].astype(np.float32, copy=False)  # FINAL IM
    comp_mag2 = zr * zr + zi * zi  # COMP MAG2

    max_mag2 = np.max(comp_mag2, axis=-1)  # MAX COMPONENT MAGNITUDE
    mask = max_mag2 < r2  # WHITE ONLY IF ALL COMPONENTS ARE BOUNDED

    img = (mask.astype(np.uint8) * 255)  # BW
    pil_img = Image.fromarray(img, mode="L")  # MAKE IMAGE
    pil_img = pil_img.resize(  # UPSCALE
        (img.shape[1] * int(scale), img.shape[0] * int(scale)),  # NEW SIZE
        resample=Image.NEAREST,  # KEEP PIXELS SHARP
    )
    pil_img.save(out_png)  # SAVE
    return str(out_png)  # RETURN

@torch.no_grad()  # NO GRAD
def reconstruct_true_final_snapshot(td, encoder, decoder, device: torch.device, batch_size: int = 50000) -> np.ndarray:  # AE ENCODE-DECODE TRUE xT
    if td.X_grid is None:  # NEED GRID
        raise ValueError("RECON NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEAT DIM
    d = (feat_dim - 2) // 2  # STATE DIM
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH

    X_final = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)  # TRUE FINAL STATE xT
    C = X_final[:, 2 * d:2 * d + 2].copy().astype(np.float32)  # C VALUES
    X_out = np.zeros((X_final.shape[0], 2 * d), dtype=np.float32)  # OUTPUT

    for i0 in range(0, X_final.shape[0], int(batch_size)):  # BATCH LOOP
        i1 = min(X_final.shape[0], i0 + int(batch_size))  # END
        x = to_tensor(X_final[i0:i1], device)  # TO DEVICE
        x_rec = decoder(encoder(x))  # ENCODE THEN DECODE
        x_rec[:, 2 * d] = to_tensor(C[i0:i1, 0], device)  # KEEP CR EXACT
        x_rec[:, 2 * d + 1] = to_tensor(C[i0:i1, 1], device)  # KEEP CI EXACT
        x_np = x_rec.detach().cpu().numpy().astype(np.float32)  # CPU
        X_out[i0:i1, 0:d] = x_np[:, 0:d]  # REAL
        X_out[i0:i1, d:2 * d] = x_np[:, d:2 * d]  # IMAG

    return X_out.reshape(H, W, 2 * d)  # IMAGE SHAPE

# ===================== TRUE NEXT-STEP GROUND TRUTH ==========================
def iterate_true_next_snapshot(td, A: np.ndarray, *, steps: int, escape_r: float) -> np.ndarray:  # TRUE x_{T+steps}
    # CONTINUE THE EXACT SYSTEM z<-(A z)^2+c FROM THE TRUE FINAL STATE xT
    if td.X_grid is None:  # NEED GRID
        raise ValueError("TRUE NEXT NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEAT DIM
    d = (feat_dim - 2) // 2  # STATE DIM
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH
    r = float(escape_r)  # R
    r2 = r * r  # R2

    A = np.asarray(A, dtype=np.float32)  # FP32 MATRIX
    X_T = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)  # TRUE FINAL STATE
    z = (X_T[:, 0:d] + 1j * X_T[:, d:2 * d]).astype(np.complex64)  # COMPLEX zT
    c = (X_T[:, 2 * d] + 1j * X_T[:, 2 * d + 1]).astype(np.complex64)  # COMPLEX c

    for _ in range(int(steps)):  # ADVANCE EXACT SYSTEM
        Az = (z @ A.T).astype(np.complex64)  # APPLY A
        z = (Az * Az).astype(np.complex64)  # NONLINEAR SQUARE
        z = (z + c[:, None]).astype(np.complex64)  # ADD C TO ALL COMPONENTS

        mag2 = (z.real * z.real + z.imag * z.imag).astype(np.float32)  # |z|^2 PER COMP
        bad = (~np.isfinite(mag2)) | (mag2 > r2)  # BAD
        if np.any(bad):  # CLAMP SAME AS BUILDER
            mag = np.sqrt(np.maximum(mag2, 1e-30)).astype(np.float32)  # |z|
            scale = (r / mag).astype(np.float32)  # SCALE
            z_real = np.where(bad, z.real * scale, z.real).astype(np.float32)  # CLAMP RE
            z_imag = np.where(bad, z.imag * scale, z.imag).astype(np.float32)  # CLAMP IM
            z = (z_real + 1j * z_imag).astype(np.complex64)  # WRITE BACK

    X_out = np.zeros((z.shape[0], 2 * d), dtype=np.float32)  # OUT
    X_out[:, 0:d] = z.real.astype(np.float32)  # RE
    X_out[:, d:2 * d] = z.imag.astype(np.float32)  # IM
    return X_out.reshape(H, W, 2 * d)  # IMAGE SHAPE

# ===================== MODEL NEXT-STEP PREDICTION ===========================
@torch.no_grad()  # NO GRAD
def predict_next_snapshot(td, encoder, decoder, dmd, device: torch.device, *, steps: int, escape_r: float, batch_size: int = 50000) -> np.ndarray:  # MODEL x_{T+steps}
    # START FROM THE TRUE FINAL STATE xT, THEN PREDICT steps STEPS AHEAD
    if td.X_grid is None:  # NEED GRID
        raise ValueError("PREDICT NEXT NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEAT DIM
    d = (feat_dim - 2) // 2  # STATE DIM
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH
    r = float(escape_r)  # R

    X_T = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)  # TRUE FINAL STATE
    C = X_T[:, 2 * d:2 * d + 2].copy().astype(np.float32)  # C VALUES
    X_out = np.zeros((X_T.shape[0], 2 * d), dtype=np.float32)  # OUTPUT
    n_roll = max(int(steps), 0)  # PREDICT_EXTRA_STEPS

    for i0 in range(0, X_T.shape[0], int(batch_size)):  # BATCH LOOP
        i1 = min(X_T.shape[0], i0 + int(batch_size))  # END
        X_t = to_tensor(X_T[i0:i1], device)  # START FROM TRUE xT
        C_t = to_tensor(C[i0:i1], device)  # C

        for _ in range(n_roll):  # PREDICT NEXT STEP(S)
            zk = encoder(X_t)  # ENC
            zk1 = dmd.predict(zk, steps=1)[-1]  # ONE DMD STEP
            X_t = decoder(zk1)  # DEC
            X_t[:, 2 * d] = C_t[:, 0]  # KEEP CR EXACT
            X_t[:, 2 * d + 1] = C_t[:, 1]  # KEEP CI EXACT

            x_np = X_t.detach().cpu().numpy().astype(np.float32)  # CPU
            zr = x_np[:, 0:d]  # RE
            zi = x_np[:, d:2 * d]  # IM
            mag = np.sqrt(np.maximum(zr * zr + zi * zi, 1e-30)).astype(np.float32)  # |z|
            bad = (~np.isfinite(mag)) | (mag > r)  # BAD
            if np.any(bad):  # CLAMP
                safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)  # SAFE
                scale = (r / safe).astype(np.float32)  # SCALE
                x_np[:, 0:d] = np.where(bad, zr * scale, zr).astype(np.float32)  # CLAMP RE
                x_np[:, d:2 * d] = np.where(bad, zi * scale, zi).astype(np.float32)  # CLAMP IM
            X_t = to_tensor(x_np, device)  # BACK TO DEVICE

        x_final = X_t.detach().cpu().numpy().astype(np.float32)  # FINAL
        X_out[i0:i1, 0:d] = x_final[:, 0:d]  # RE
        X_out[i0:i1, d:2 * d] = x_final[:, d:2 * d]  # IM

    return X_out.reshape(H, W, 2 * d)  # IMAGE SHAPE

def next_step_prediction_metrics(pred_grid: np.ndarray, true_grid: np.ndarray) -> dict:  # PRED vs TRUE NEXT
    p = pred_grid.astype(np.float32).reshape(-1)  # FLAT PRED
    t = true_grid.astype(np.float32).reshape(-1)  # FLAT TRUE
    err = p - t  # ERR
    mse = float(np.mean(err * err))  # MSE
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(t), 1e-12))  # REL
    fit = float(1.0 - rel_l2)  # FIT
    return {"pred_mse": mse, "pred_rel_l2": rel_l2, "pred_fit": fit}  # RETURN

# ===================== TEACHER-FORCED PREDICTED FRACTAL =====================
@torch.no_grad()  # NO GRAD
def teacher_forced_escape_iters(td, encoder, decoder, dmd, device: torch.device, *, escape_r: float, batch_size: int = 50000) -> np.ndarray:  # ONE-STEP PRED FRACTAL
    # AT EACH TRUE STATE x_t PREDICT x_{t+1}; RECORD FIRST PREDICTED ESCAPE (NO COMPOUNDING)
    if td.X_grid is None:  # NEED GRID
        raise ValueError("TEACHER FORCED FRACTAL NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEAT DIM
    d = (feat_dim - 2) // 2  # STATE DIM
    T = int(td.X_grid.shape[0])  # STEPS
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH
    r2 = float(escape_r) * float(escape_r)  # R2

    P = H * W  # POINT COUNT
    iters = np.full((P,), int(T), dtype=np.int32)  # DEFAULT NEVER ESCAPED

    for t in range(0, T - 1):  # LOOP TRUE STATES x_1..x_{T-1}
        X_t = td.X_grid[t].reshape(-1, feat_dim).astype(np.float32)  # TRUE STATE
        C = X_t[:, 2 * d:2 * d + 2]  # C VALUES

        esc_pred = np.zeros((P,), dtype=bool)  # ESCAPED AT THIS STEP

        for i0 in range(0, P, int(batch_size)):  # BATCH LOOP
            i1 = min(P, i0 + int(batch_size))  # END
            xb = to_tensor(X_t[i0:i1], device)  # TRUE STATE BATCH
            zk1 = dmd.predict(encoder(xb), steps=1)[-1]  # PREDICT NEXT LATENT
            xk1 = decoder(zk1)  # DECODE PREDICTED NEXT
            xk1[:, 2 * d] = to_tensor(C[i0:i1, 0], device)  # KEEP CR
            xk1[:, 2 * d + 1] = to_tensor(C[i0:i1, 1], device)  # KEEP CI

            x_np = xk1.detach().cpu().numpy().astype(np.float32)  # CPU
            zr = x_np[:, 0:d]  # RE
            zi = x_np[:, d:2 * d]  # IM
            comp_mag2 = zr * zr + zi * zi  # MAG2
            max_mag2 = np.max(np.where(np.isfinite(comp_mag2), comp_mag2, np.inf), axis=1)  # MAX COMP
            esc_pred[i0:i1] = (~np.isfinite(max_mag2)) | (max_mag2 >= r2)  # ESCAPED

        new_esc = esc_pred & (iters == int(T))  # FIRST ESCAPE ONLY
        iters[new_esc] = int(t + 2)  # PREDICTED x_{t+1} IS ITER t+2

    return iters.reshape(H, W)  # IMAGE SHAPE

# ============================== TRUE FRACTAL =================================
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