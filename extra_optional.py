# ============================ extra_optional.py ==============================
from __future__ import annotations  # MODERN TYPE HINTS

# ================================= IMPORTS ==================================
import numpy as np  # NUMPY
import torch  # TORCH
import torch.nn as nn  # NN MODULE
from PIL import Image  # IMAGE SAVE
from pathlib import Path  # PATHS
from torch.utils.data import DataLoader, TensorDataset  # TRAIN LOADER

import defines as D  # PROJECT DEFINES

from encoder import Encoder  # ENCODER MODEL
from decoder import Decoder  # DECODER MODEL
from quadratic_predictor import make_quadratic_predictor_pair, plot_learned_vs_true_matrix_spectrum  # QUADRATIC PREDICTOR
from losses import make_reconstruction_loss  # BASIC AE LOSS
from utils import to_tensor  # TENSOR HELPER

from data_loader import load_all_A_matrices, split_explicit_matrix_indices, load_one_A_matrix  # MATRIX SPLIT
from prepare_training_data import (  # TRAINING DATA BUILDERS
    build_matrix_c_grid_training_data,  # SINGLE MATRIX DATA
    build_matrix_c_grid_training_data_many_matrices,  # MULTI MATRIX DATA
    save_training_npz,  # SAVE NPZ
)
from apply_dmd import fit_dmd_from_latent_covariances  # STREAMED DMD FIT
from eval_matrix_dmd_ae import (  # EXISTING VISUAL HELPERS
    save_loss_curve,  # LOSS PLOT
    save_ground_truth_final_mask,  # GT MASK
    save_ground_truth_escape_iters,  # GT ESCAPE ITERS
    iterate_true_next_snapshot,  # TRUE x_{T+k}
    next_step_prediction_metrics,  # PRED vs TRUE NEXT
)
from mandelbrot_reconstruct import save_final_snapshot_image, save_escape_image  # IMAGE HELPERS
from experiment_common import (  # COMMON HELPERS
    make_out_dirs,  # OUTPUT DIRS
    print_metric_block,  # PRINT
    mean_metric_dict,  # MEAN
    write_metrics_txt,  # SAVE METRICS
)


# =============================== METRICS =====================================
def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:  # RELATIVE L2
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))  # SAFE


def _mse(pred: np.ndarray, true: np.ndarray) -> float:  # MSE
    err = pred - true  # ERROR
    return float(np.mean(err * err))  # MEAN SQUARED ERROR


# ============================= IMAGE HELPERS =================================
def _save_final_mask_95(Z_final: np.ndarray, escape_r: float, out_png, scale: int = 64) -> str:  # SAVE BOUNDED MASK
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # DIR

    d = int(Z_final.shape[-1] // 2)  # STATE DIM
    zr = Z_final[..., 0:d].astype(np.float32, copy=False)  # REAL
    zi = Z_final[..., d:2 * d].astype(np.float32, copy=False)  # IMAG
    r2 = float(escape_r) * float(escape_r)  # RADIUS SQUARED

    comp_mag2 = zr * zr + zi * zi  # COMPONENT MAGNITUDES
    max_mag2 = np.max(comp_mag2, axis=-1)  # MAX COMPONENT MAGNITUDE
    mask = max_mag2 < r2  # WHITE ONLY IF ALL COMPONENTS ARE BOUNDED

    img = (mask.astype(np.uint8) * 255)  # BLACK WHITE
    pil = Image.fromarray(img, mode="L")  # IMAGE
    pil = pil.resize(  # UPSCALE
        (img.shape[1] * int(scale), img.shape[0] * int(scale)),  # SIZE
        resample=Image.NEAREST,  # SHARP PIXELS
    )
    pil.save(out_png)  # SAVE
    return str(out_png)  # RETURN

def _save_mask_from_escape_iters(escape_iters: np.ndarray, max_iters: int, out_png, scale: int = 1) -> str:  # SAVE BOUNDED MASK
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    mask = escape_iters.astype(np.int32) >= int(max_iters)  # NEVER ESCAPED
    img = (mask.astype(np.uint8) * 255)  # WHITE BOUNDED

    pil = Image.fromarray(img, mode="L")  # IMAGE
    if int(scale) > 1:  # UPSCALE
        pil = pil.resize(
            (img.shape[1] * int(scale), img.shape[0] * int(scale)),
            resample=Image.NEAREST,
        )

    pil.save(out_png)  # SAVE
    return str(out_png)  # RETURN

def save_escape_image_contrast(escape_iters: np.ndarray, *, out_png) -> str:  # CONTRAST-STRETCHED PNG
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    esc = escape_iters.astype(np.float32)  # FLOAT
    lo = float(np.min(esc))  # MIN
    hi = float(np.max(esc))  # MAX

    if hi <= lo + 1e-6:  # FLAT IMAGE
        img = np.zeros_like(esc, dtype=np.uint8)  # BLACK
    else:
        norm = (esc - lo) / (hi - lo)  # LOCAL NORMALISE
        img = (255.0 * (1.0 - norm)).clip(0, 255).astype(np.uint8)  # INVERT

    Image.fromarray(img, mode="L").save(out_png)  # SAVE
    return str(out_png)  # RETURN


#============HELPER FOR PREDICT - ONLY ALIVE ROWS - SE f CEVA LA EA ==========
def _alive_rows(X: np.ndarray, escape_r: float) -> np.ndarray:  # ALIVE MASK
    X = np.asarray(X, dtype=np.float32)  # ARRAY
    feat_dim = int(X.shape[1])  # FEAT
    d = int((feat_dim - 2) // 2)  # STATE DIM
    r2 = float(escape_r) * float(escape_r)  # R2

    zr = X[:, 0:d]  # REAL
    zi = X[:, d:2 * d]  # IMAG

    mag2 = zr * zr + zi * zi  # MAG2
    max_mag2 = np.max(np.where(np.isfinite(mag2), mag2, np.inf), axis=1)  # MAX

    finite = np.isfinite(zr).all(axis=1) & np.isfinite(zi).all(axis=1)  # FINITE
    alive = finite & (max_mag2 < 0.999 * r2)  # STRICTLY INSIDE

    return alive  # RETURN
# ========================== AE PREDICTOR TRAINER ============================
def train_autoencoder_predict_next_only(  # TRAIN AE AS X_N -> X_N+1
    X1, X2, *, latent_dim: int, epochs: int, batch_size: int, lr: float, device: torch.device,
    encoder: nn.Module | None = None,  # OPTIONAL: SWAP IN A DIFFERENT ARCHITECTURE
    decoder: nn.Module | None = None,  # OPTIONAL: SWAP IN A DIFFERENT ARCHITECTURE
) -> tuple[nn.Module, nn.Module, list[float]]:
    X1_list = [X1] if isinstance(X1, np.ndarray) else list(X1)  # WRAP
    X2_list = [X2] if isinstance(X2, np.ndarray) else list(X2)  # WRAP

    if len(X1_list) == 0 or len(X2_list) == 0:  # EMPTY CHECK
        raise ValueError("AE PREDICTOR RECEIVED EMPTY X1 OR X2 LIST")  # ERROR
    if len(X1_list) != len(X2_list):  # LENGTH CHECK
        raise ValueError("AE PREDICTOR X1 AND X2 LISTS MUST HAVE SAME LENGTH")  # ERROR

    in_dim = int(X1_list[0].shape[1])  # INPUT DIM
    enc = (encoder if encoder is not None else Encoder(in_dim, latent_dim)).to(device)  # ENCODER
    dec = (decoder if decoder is not None else Decoder(latent_dim, in_dim)).to(device)  # DECODER

    loss_fn = make_reconstruction_loss(0, beta=0.01, w_pow=1.0)  # MSE
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)  # OPTIMIZER

    losses: list[float] = []  # STORE LOSSES

    for e in range(int(epochs)):  # EPOCH LOOP
        rng = np.random.default_rng(e)  # RNG
        order = rng.permutation(len(X1_list))  # SHUFFLE BLOCKS

        s = 0.0  # LOSS SUM
        n_batches = 0  # BATCH COUNT

        for block_id in order:  # BLOCK LOOP
            X1_block = np.asarray(X1_list[int(block_id)], dtype=np.float32)  # BLOCK
            X2_block = np.asarray(X2_list[int(block_id)], dtype=np.float32)  # BLOCK
            if bool(getattr(D, "AE_PRED_USE_ALIVE_ONLY", True)):  # FILTER DEAD DATA
                keep = _alive_rows(X1_block, float(D.ESCAPE_R))  # PRE-ESCAPE ONLY

                if int(np.count_nonzero(keep)) > 0:  # HAS DATA
                    X1_block = X1_block[keep].astype(np.float32)  # FILTER X1
                    X2_block = X2_block[keep].astype(np.float32)  # FILTER X2
            X1_cpu = torch.tensor(X1_block, dtype=torch.float32)  # CPU TENSOR
            X2_cpu = torch.tensor(X2_block, dtype=torch.float32)  # CPU TENSOR

            loader = DataLoader(  # LOADER
                TensorDataset(X1_cpu, X2_cpu),  # X_n, X_n+1
                batch_size=int(batch_size),  # BATCH
                shuffle=True,  # SHUFFLE
                drop_last=False,  # KEEP ALL
            )

            for b1_cpu, b2_cpu in loader:  # BATCH LOOP
                b1 = b1_cpu.to(device)  # X_n
                b2 = b2_cpu.to(device)  # X_n+1

                z1 = enc(b1)  # ENCODE X_n
                x2_pred = dec(z1)  # PREDICT X_n+1
                loss = loss_fn(x2_pred, b2)  # COMPARE WITH NEXT STATE

                opt.zero_grad()  # ZERO
                loss.backward()  # BACKPROP
                opt.step()  # UPDATE

                s += float(loss.detach().cpu().item())  # ACCUMULATE
                n_batches += 1  # COUNT

        s /= max(1, n_batches)  # MEAN LOSS
        losses.append(s)  # STORE
        print(f"[AE PREDICTOR] EPOCH {e + 1}/{epochs} LOSS={s:.6e}")  # LOG

    return enc, dec, losses  # RETURN


@torch.no_grad()  # NO GRAD
def ae_predictor_one_step_metrics(encoder, decoder, X1: np.ndarray, X2: np.ndarray, device) -> dict:  # AE X_N -> X_N+1 METRICS
    x1 = to_tensor(X1.astype(np.float32), device)  # LEFT
    pred = decoder(encoder(x1))  # PREDICT NEXT
    pred_np = pred.detach().cpu().numpy().astype(np.float32)  # CPU

    if pred_np.shape[1] >= 4 and pred_np.shape[1] == X2.shape[1]:  # SAFE CHECK
        pred_np[:, -2:] = X2[:, -2:].astype(np.float32)  # DO NOT PENALISE C DRIFT

    err = pred_np - X2.astype(np.float32)  # ERR
    mse = float(np.mean(err * err))  # MSE
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(X2), 1e-12))  # REL
    fit = float(1.0 - rel_l2)  # FIT
    return {"ae_pred_mse": mse, "ae_pred_rel_l2": rel_l2, "ae_pred_fit": fit}  # RETURN


# ============================ AE OUTPUT SNAPSHOTS ============================
@torch.no_grad()  # NO GRAD
def reconstruct_final_snapshot_ae_only(td, encoder, decoder, device, batch_size: int = 50000) -> np.ndarray:  # ENCODE-DECODE TRUE xT
    if td.X_grid is None:  # NEED GRID
        raise ValueError("AE ONLY FINAL SNAPSHOT NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEATURE DIM
    d = int((feat_dim - 2) // 2)  # STATE DIM
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH

    X_final = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)  # TRUE FINAL X_T
    C = X_final[:, 2 * d:2 * d + 2].copy().astype(np.float32)  # C VALUES
    X_out = np.zeros((X_final.shape[0], 2 * d), dtype=np.float32)  # OUTPUT

    for i0 in range(0, X_final.shape[0], int(batch_size)):  # BATCH LOOP
        i1 = min(X_final.shape[0], i0 + int(batch_size))  # END
        x = to_tensor(X_final[i0:i1], device)  # TO DEVICE
        x_rec = decoder(encoder(x))  # AE ONLY: ENCODE -> DECODE
        x_rec[:, 2 * d] = to_tensor(C[i0:i1, 0], device)  # KEEP CR EXACT
        x_rec[:, 2 * d + 1] = to_tensor(C[i0:i1, 1], device)  # KEEP CI EXACT
        x_np = x_rec.detach().cpu().numpy().astype(np.float32)  # CPU
        X_out[i0:i1, 0:d] = x_np[:, 0:d]  # REAL
        X_out[i0:i1, d:2 * d] = x_np[:, d:2 * d]  # IMAG

    return X_out.reshape(H, W, 2 * d)  # IMAGE SHAPE


@torch.no_grad()  # NO GRAD
def predict_next_snapshot_ae_predictor(td, encoder, decoder, device, *, steps: int, escape_r: float, batch_size: int = 50000) -> np.ndarray:  # AE x_{T+steps} FROM TRUE xT
    # START FROM THE TRUE FINAL STATE xT, THEN PREDICT steps STEP(S) WITH THE AE PREDICTOR
    if td.X_grid is None:  # NEED GRID
        raise ValueError("AE PREDICT NEXT NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEATURE DIM
    d = int((feat_dim - 2) // 2)  # STATE DIM
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH
    r = float(escape_r)  # R

    X_T = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)  # TRUE FINAL STATE xT
    C = X_T[:, 2 * d:2 * d + 2].copy().astype(np.float32)  # C VALUES
    X_out = np.zeros((X_T.shape[0], 2 * d), dtype=np.float32)  # OUTPUT
    n_roll = max(int(steps), 0)  # PREDICT_EXTRA_STEPS

    for i0 in range(0, X_T.shape[0], int(batch_size)):  # BATCH LOOP
        i1 = min(X_T.shape[0], i0 + int(batch_size))  # END
        X_t = to_tensor(X_T[i0:i1], device)  # START FROM TRUE xT
        C_t = to_tensor(C[i0:i1], device)  # C

        for _ in range(n_roll):  # PREDICT NEXT STEP(S)
            X_t = decoder(encoder(X_t))  # AE PREDICTS NEXT STATE
            X_t[:, 2 * d] = C_t[:, 0]  # KEEP CR EXACT
            X_t[:, 2 * d + 1] = C_t[:, 1]  # KEEP CI EXACT

            x_np = X_t.detach().cpu().numpy().astype(np.float32)  # CPU
            zr = x_np[:, 0:d]  # REAL
            zi = x_np[:, d:2 * d]  # IMAG
            mag = np.sqrt(np.maximum(zr * zr + zi * zi, 1e-30)).astype(np.float32)  # MAG
            bad = (~np.isfinite(mag)) | (mag > r)  # ESCAPED
            if np.any(bad):  # CLAMP
                safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)  # SAFE
                scale = (r / safe).astype(np.float32)  # SCALE
                x_np[:, 0:d] = np.where(bad, zr * scale, zr).astype(np.float32)  # CLAMP RE
                x_np[:, d:2 * d] = np.where(bad, zi * scale, zi).astype(np.float32)  # CLAMP IM
            X_t = to_tensor(x_np, device)  # BACK TO DEVICE

        x_final = X_t.detach().cpu().numpy().astype(np.float32)  # FINAL
        X_out[i0:i1, 0:d] = x_final[:, 0:d]  # REAL
        X_out[i0:i1, d:2 * d] = x_final[:, d:2 * d]  # IMAG

    return X_out.reshape(H, W, 2 * d)  # IMAGE SHAPE

@torch.no_grad()  # NO GRAD
def predict_future_snapshots_ae_predictor(
    td,
    encoder,
    decoder,
    device,
    *,
    steps: int,
    escape_r: float,
    batch_size: int = 50000,
) -> np.ndarray:  # AE x_{N+1} ... x_{N+T}
    # START FROM TRUE xN, THEN ROLL AE FORWARD
    if td.X_grid is None:  # NEED GRID
        raise ValueError("AE FUTURE PREDICT NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEATURE DIM
    d = int((feat_dim - 2) // 2)  # STATE DIM
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH
    r = float(escape_r)  # R
    n_roll = max(int(steps), 0)  # FUTURE STEPS

    X_N = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)  # TRUE xN
    C = X_N[:, 2 * d:2 * d + 2].copy().astype(np.float32)  # C VALUES

    P = int(X_N.shape[0])  # PIXELS
    future = np.zeros((n_roll, P, 2 * d), dtype=np.float32)  # OUTPUT

    for i0 in range(0, P, int(batch_size)):  # BATCH LOOP
        i1 = min(P, i0 + int(batch_size))  # END

        X_t = to_tensor(X_N[i0:i1], device)  # START FROM TRUE xN
        C_t = to_tensor(C[i0:i1], device)  # C

        for s in range(n_roll):  # FUTURE LOOP
            X_t = decoder(encoder(X_t))  # AE PREDICTS NEXT
            X_t[:, 2 * d] = C_t[:, 0]  # KEEP CR EXACT
            X_t[:, 2 * d + 1] = C_t[:, 1]  # KEEP CI EXACT

            x_np = X_t.detach().cpu().numpy().astype(np.float32)  # CPU

            zr = x_np[:, 0:d]  # REAL
            zi = x_np[:, d:2 * d]  # IMAG
            mag = np.sqrt(np.maximum(zr * zr + zi * zi, 1e-30)).astype(np.float32)  # MAG
            bad = (~np.isfinite(mag)) | (mag > r)  # BAD

            if np.any(bad):  # CLAMP
                safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)  # SAFE
                scale = (r / safe).astype(np.float32)  # SCALE
                x_np[:, 0:d] = np.where(bad, zr * scale, zr).astype(np.float32)  # CLAMP RE
                x_np[:, d:2 * d] = np.where(bad, zi * scale, zi).astype(np.float32)  # CLAMP IM

            x_np[:, 2 * d:2 * d + 2] = C[i0:i1]  # KEEP C EXACT

            future[s, i0:i1, 0:d] = x_np[:, 0:d]  # SAVE RE
            future[s, i0:i1, d:2 * d] = x_np[:, d:2 * d]  # SAVE IM

            X_t = to_tensor(x_np, device)  # FEED PREDICTION BACK

    return future.reshape(n_roll, H, W, 2 * d)  # RETURN xN+1...xN+T
# ==================== AE ROLLOUT FROM START (VERIFICATION) ====================
@torch.no_grad()  # NO GRAD
def predict_rollout_from_start_ae_predictor(  # AE ROLLOUT FROM TRUE x1
    td,
    encoder,
    decoder,
    device,
    *,
    steps: int,
    escape_r: float,
    batch_size: int = 50000,
) -> np.ndarray:  # AE x_2 ... x_{steps} FROM THE TRUE ANCHOR x_1
    # START FROM x1 (TRUE FIRST STORED STATE -- z0=0 SO x1 = c IN ALL COMPONENTS)
    # SAME CONVENTION AS mandelbrot_reconstruct.reconstruct_final_snapshot
    if td.X_grid is None:  # NEED GRID
        raise ValueError("AE ROLLOUT-FROM-START NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEATURE DIM
    d = int((feat_dim - 2) // 2)  # STATE DIM
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH
    r = float(escape_r)  # R

    n_roll = max(int(steps) - 1, 0)  # T-1 STEPS FROM x1 TO REACH x_{steps}

    X_1 = td.X_grid[0].reshape(-1, feat_dim).astype(np.float32)  # TRUE x1
    C = X_1[:, 2 * d:2 * d + 2].copy().astype(np.float32)  # C VALUES

    P = int(X_1.shape[0])  # PIXELS
    future = np.zeros((n_roll, P, 2 * d), dtype=np.float32)  # OUTPUT: x_2 .. x_{steps}

    for i0 in range(0, P, int(batch_size)):  # BATCH LOOP
        i1 = min(P, i0 + int(batch_size))  # END

        X_t = to_tensor(X_1[i0:i1], device)  # START FROM TRUE x1
        C_t = to_tensor(C[i0:i1], device)  # C

        for s in range(n_roll):  # ROLL x1 -> x_{steps}
            X_t = decoder(encoder(X_t))  # AE PREDICTS NEXT STATE
            X_t[:, 2 * d] = C_t[:, 0]  # KEEP CR EXACT
            X_t[:, 2 * d + 1] = C_t[:, 1]  # KEEP CI EXACT

            x_np = X_t.detach().cpu().numpy().astype(np.float32)  # CPU
            zr = x_np[:, 0:d]  # REAL
            zi = x_np[:, d:2 * d]  # IMAG
            mag = np.sqrt(np.maximum(zr * zr + zi * zi, 1e-30)).astype(np.float32)  # MAG
            bad = (~np.isfinite(mag)) | (mag > r)  # ESCAPED

            if np.any(bad):  # CLAMP, SAME AS THE DATA BUILDER
                safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)  # SAFE
                scale = (r / safe).astype(np.float32)  # SCALE
                x_np[:, 0:d] = np.where(bad, zr * scale, zr).astype(np.float32)  # CLAMP RE
                x_np[:, d:2 * d] = np.where(bad, zi * scale, zi).astype(np.float32)  # CLAMP IM

            x_np[:, 2 * d:2 * d + 2] = C[i0:i1]  # KEEP C EXACT

            future[s, i0:i1, 0:d] = x_np[:, 0:d]  # SAVE RE
            future[s, i0:i1, d:2 * d] = x_np[:, d:2 * d]  # SAVE IM

            X_t = to_tensor(x_np, device)  # FEED PREDICTION BACK

    return future.reshape(n_roll, H, W, 2 * d)  # RETURN x_2 ... x_{steps}
@torch.no_grad()  # NO GRAD
def teacher_forced_escape_iters_ae_predictor(td, encoder, decoder, device, escape_r: float, batch_size: int = 50000) -> np.ndarray:  # ONE-STEP PRED FRACTAL
    # AT EACH TRUE STATE x_t PREDICT x_{t+1} WITH THE AE; RECORD FIRST PREDICTED ESCAPE (NO COMPOUNDING)
    if td.X_grid is None:  # NEED GRID
        raise ValueError("AE TEACHER FORCED FRACTAL NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEATURE DIM
    d = int((feat_dim - 2) // 2)  # STATE DIM
    T = int(td.X_grid.shape[0])  # STORED STEPS
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
            xk1 = decoder(encoder(xb))  # AE PREDICTS NEXT STATE
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


# ============================= DMD ONLY HELPERS =============================
def fit_full_state_dmd_streamed(X1, X2, *, device: torch.device):  # FIT RAW DMD WITHOUT AE
    X1_list = [X1] if isinstance(X1, np.ndarray) else list(X1)  # WRAP
    X2_list = [X2] if isinstance(X2, np.ndarray) else list(X2)  # WRAP

    if len(X1_list) != len(X2_list):  # MISMATCH
        raise ValueError("DMD ONLY X1 AND X2 LISTS MUST HAVE SAME LENGTH")  # ERROR

    feat_dim = int(X1_list[0].shape[1])  # FEATURE DIM
    G = np.zeros((feat_dim, feat_dim), dtype=np.float64)  # X1^T X1
    H = np.zeros((feat_dim, feat_dim), dtype=np.float64)  # X1^T X2

    for a, b in zip(X1_list, X2_list):  # BLOCK LOOP
        A = np.asarray(a, dtype=np.float64)  # LEFT
        B = np.asarray(b, dtype=np.float64)  # RIGHT
        G += A.T @ A  # ACCUMULATE
        H += A.T @ B  # ACCUMULATE

    return fit_dmd_from_latent_covariances(G, H, device=device)  # SAME FORMULA, FULL STATE


@torch.no_grad()  # NO GRAD
def dmd_only_one_step_metrics(dmd, X1: np.ndarray, X2: np.ndarray, device) -> dict:  # RAW DMD METRICS
    x1 = to_tensor(X1.astype(np.float32), device)  # LEFT
    x2_hat = dmd.predict(x1, steps=1)[-1]  # ONE STEP
    pred = x2_hat.detach().cpu().numpy().astype(np.float32)  # CPU

    return {  # METRICS
        "dmd_only_mse": _mse(pred, X2.astype(np.float32)),  # MSE
        "dmd_only_rel_l2": _rel_l2(pred, X2.astype(np.float32)),  # REL L2
        "dmd_only_fit": float(1.0 - _rel_l2(pred, X2.astype(np.float32))),  # FIT
    }


@torch.no_grad()  # NO GRAD
def predict_next_snapshot_dmd_only(td, dmd, device, *, steps: int, escape_r: float, batch_size: int = 50000) -> np.ndarray:  # RAW DMD x_{T+steps} FROM TRUE xT
    # START FROM THE TRUE FINAL STATE xT, THEN PREDICT steps STEP(S) WITH THE RAW DMD
    if td.X_grid is None:  # NEED GRID
        raise ValueError("DMD ONLY PREDICT NEXT NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEATURE DIM
    d = int((feat_dim - 2) // 2)  # STATE DIM
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH
    r = float(escape_r)  # R

    X_T = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)  # TRUE FINAL STATE xT
    C = X_T[:, 2 * d:2 * d + 2].copy().astype(np.float32)  # C VALUES
    X_out = np.zeros((X_T.shape[0], 2 * d), dtype=np.float32)  # OUTPUT
    n_roll = max(int(steps), 0)  # PREDICT_EXTRA_STEPS

    for i0 in range(0, X_T.shape[0], int(batch_size)):  # BATCH LOOP
        i1 = min(X_T.shape[0], i0 + int(batch_size))  # END
        X_t = to_tensor(X_T[i0:i1], device)  # START FROM TRUE xT
        C_t = to_tensor(C[i0:i1], device)  # C

        for _ in range(n_roll):  # PREDICT NEXT STEP(S)
            X_t = dmd.predict(X_t, steps=1)[-1]  # RAW DMD STEP
            X_t[:, 2 * d] = C_t[:, 0]  # KEEP CR EXACT
            X_t[:, 2 * d + 1] = C_t[:, 1]  # KEEP CI EXACT

            x_np = X_t.detach().cpu().numpy().astype(np.float32)  # CPU
            zr = x_np[:, 0:d]  # REAL
            zi = x_np[:, d:2 * d]  # IMAG
            mag = np.sqrt(np.maximum(zr * zr + zi * zi, 1e-30)).astype(np.float32)  # MAG
            bad = (~np.isfinite(mag)) | (mag > r)  # EXPLODED
            if np.any(bad):  # CLAMP
                safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)  # SAFE
                scale = (r / safe).astype(np.float32)  # SCALE
                x_np[:, 0:d] = np.where(bad, zr * scale, zr).astype(np.float32)  # CLAMP REAL
                x_np[:, d:2 * d] = np.where(bad, zi * scale, zi).astype(np.float32)  # CLAMP IMAG
            X_t = to_tensor(x_np, device)  # BACK TO DEVICE

        x_final = X_t.detach().cpu().numpy().astype(np.float32)  # FINAL
        X_out[i0:i1, 0:d] = x_final[:, 0:d]  # REAL
        X_out[i0:i1, d:2 * d] = x_final[:, d:2 * d]  # IMAG

    return X_out.reshape(H, W, 2 * d)  # IMAGE SHAPE


# =========================== DATA BUILDERS USED HERE =========================
def _build_single_td(train_clip_r: float | None = None):  # BUILD SINGLE MATRIX DATA
    clip_r = float(D.ESCAPE_R if train_clip_r is None else train_clip_r)  # TRAIN CLIP

    return build_matrix_c_grid_training_data(  # SAME AS MAIN
        data_dir=D.A_DATA_DIR,  # DATA DIR
        source=D.SINGLE_MATRIX_SOURCE,  # SOURCE
        index=D.SINGLE_MATRIX_INDEX,  # MATRIX INDEX
        c_re_min=D.C_RE_MIN,  # RE MIN
        c_re_max=D.C_RE_MAX,  # RE MAX
        c_im_min=D.C_IM_MIN,  # IM MIN
        c_im_max=D.C_IM_MAX,  # IM MAX
        c_re_n=D.SINGLE_MATRIX_C_RE_N,  # RE RES
        c_im_n=D.SINGLE_MATRIX_C_IM_N,  # IM RES
        max_iters=D.TRAIN_MAX_ITERS,  # ITERS
        escape_r=clip_r,  # TRAIN CLIP, NOT FINAL ESCAPE THRESHOLD
    )


def _build_multi_train_test():  # BUILD MULTI MATRIX DATA
    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)  # LOAD ALL
    total = int(A_all.shape[0])  # COUNT

    train_idx, test_idx = split_explicit_matrix_indices(  # SPLIT
        total_count=total,  # TOTAL
        train_count=D.MULTI_MATRIX_TRAIN_COUNT,  # TRAIN COUNT
        test_count=D.MULTI_MATRIX_TEST_COUNT,  # TEST COUNT
        seed=D.MULTI_MATRIX_SPLIT_SEED,  # SEED
    )

    td_train_list = build_matrix_c_grid_training_data_many_matrices(  # TRAIN DATA
        data_dir=D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE, indices=train_idx,
        c_re_min=D.C_RE_MIN, c_re_max=D.C_RE_MAX, c_im_min=D.C_IM_MIN, c_im_max=D.C_IM_MAX,
        c_re_n=D.MULTI_MATRIX_C_RE_N, c_im_n=D.MULTI_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS, escape_r=D.ESCAPE_R,
    )

    td_test_list = build_matrix_c_grid_training_data_many_matrices(  # TEST DATA
        data_dir=D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE, indices=test_idx,
        c_re_min=D.C_RE_MIN, c_re_max=D.C_RE_MAX, c_im_min=D.C_IM_MIN, c_im_max=D.C_IM_MAX,
        c_re_n=D.MULTI_MATRIX_C_RE_N, c_im_n=D.MULTI_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS, escape_r=D.ESCAPE_R,
    )

    return train_idx, test_idx, td_train_list, td_test_list  # RETURN


# =============================== AE-ONLY-SINGLE =============================
def run_ae_only_single(device: torch.device) -> None:  # RUN AE-ONLY ON ONE MATRIX
    print("\n================ AE ONLY / SINGLE MATRIX ================\n")  # HEADER
    dirs = make_out_dirs("ae-only-single")  # OUT DIRS

    A = load_one_A_matrix(D.A_DATA_DIR, source=D.SINGLE_MATRIX_SOURCE, index=D.SINGLE_MATRIX_INDEX)  # TRUE MATRIX
    td = _build_single_td(float(getattr(D, "AE_PRED_TRAIN_CLIP_R", D.ESCAPE_R)))  # BUILD DATA

    save_training_npz(dirs["td"] / "training_single_matrix.npz", td)  # SAVE NPZ
    save_ground_truth_escape_iters(td, D.ESCAPE_R, dirs["td"] / "gt_escape_iters.png")  # GT FRACTAL
    save_ground_truth_final_mask(td, D.ESCAPE_R, dirs["td"] / "gt_final_mask.png", scale=D.IMAGE_SCALE)  # GT MASK

    train_kwargs = dict(  # SHARED TRAIN KWARGS
        latent_dim=int(getattr(D, "AE_PRED_LATENT_DIM", D.LATENT_DIM)),  # BIGGER LATENT
        epochs=int(getattr(D, "AE_PRED_EPOCHS", getattr(D, "AE_ONLY_EPOCHS", D.AE_EPOCHS))),  # EPOCHS
        batch_size=int(getattr(D, "AE_ONLY_BATCH_SIZE", D.AE_BATCH_SIZE)),  # BATCH
        lr=float(getattr(D, "AE_ONLY_LR", D.AE_LR)),  # LR
        device=device,
    )
    if bool(getattr(D, "AE_USE_QUADRATIC_PREDICTOR", False)):  # TOGGLE IN defines.py
        state_dim = int(td.meta["state_dim"])  # d, ALREADY SAVED BY THE DATA BUILDER
        q_enc, q_dec = make_quadratic_predictor_pair(  # BUILD (encoder, decoder) PAIR
            state_dim, rank=getattr(D, "AE_QUADRATIC_RANK", None),
        )
        enc, dec, losses = train_autoencoder_predict_next_only(  # TRAIN AE PREDICTOR
            td.X1, td.X2, encoder=q_enc, decoder=q_dec, **train_kwargs,
        )

        plot_learned_vs_true_matrix_spectrum(  # CHECK LEARNED A vs TRUE A
            A, q_enc.A.weight, dirs["res"] / "quadratic_A_spectrum.png",
        )
    else:  # DEFAULT: EXISTING Encoder/Decoder MLP
        enc, dec, losses = train_autoencoder_predict_next_only(td.X1, td.X2, **train_kwargs)  # TRAIN AE PREDICTOR

    save_loss_curve(losses, dirs["res"] / "loss_curve.png", "AE Predictor Single Matrix Loss")  # LOSS

    # FUTURE PREDICTION xN+1 ... xN+T
    k = int(D.PREDICT_EXTRA_STEPS)  # FUTURE STEPS
    future_pred = predict_future_snapshots_ae_predictor(td, enc, dec, device, steps=k, escape_r=D.ESCAPE_R)  # AE FUTURE

    m = ae_predictor_one_step_metrics(enc, dec, td.X1, td.X2, device)  # ONE-STEP METRICS
    print_metric_block("AE PREDICTOR SINGLE (ONE-STEP)", m)  # PRINT

    all_metrics = {**m, "predict_extra_steps": float(k)}  # METRICS

    for s in range(k):  # SAVE EACH FUTURE STEP
        step = s + 1  # HUMAN STEP

        Z_pred = future_pred[s]  # AE xN+step
        Z_true = iterate_true_next_snapshot(td, A, steps=step, escape_r=D.ESCAPE_R)  # TRUE xN+step

        save_final_snapshot_image(
            Z_pred,
            escape_r=D.ESCAPE_R,
            out_png=dirs["res"] / f"pred_xn_plus_{step:03d}_mask.png",
            mode="mask",
        )  # SAVE PRED MASK

        save_final_snapshot_image(
            Z_pred,
            escape_r=D.ESCAPE_R,
            out_png=dirs["res"] / f"pred_xn_plus_{step:03d}_mag.png",
            mode="mag",
        )  # SAVE PRED MAG

        save_final_snapshot_image(
            Z_true,
            escape_r=D.ESCAPE_R,
            out_png=dirs["res"] / f"true_xn_plus_{step:03d}_mask.png",
            mode="mask",
        )  # SAVE TRUE MASK

        save_final_snapshot_image(
            Z_true,
            escape_r=D.ESCAPE_R,
            out_png=dirs["res"] / f"true_xn_plus_{step:03d}_mag.png",
            mode="mag",
        )  # SAVE TRUE MAG

        pred_m = next_step_prediction_metrics(Z_pred, Z_true)  # STEP METRICS
        print_metric_block(f"AE PREDICTOR SINGLE xN+{step}", pred_m)  # PRINT

        all_metrics[f"pred_mse_xn_plus_{step:03d}"] = float(pred_m["pred_mse"])  # SAVE MSE
        all_metrics[f"pred_rel_l2_xn_plus_{step:03d}"] = float(pred_m["pred_rel_l2"])  # SAVE REL
        all_metrics[f"pred_fit_xn_plus_{step:03d}"] = float(pred_m["pred_fit"])  # SAVE FIT

        # ============================================================
        # NEW, SERIOUS TESTS: ROLL THE AE FORWARD FROM x1 (TRUE z0=0 ANCHOR)
        # THROUGH THE KNOWN TRAJECTORY, SO EVERY STEP HAS A REAL x_t TO CHECK
        # AGAINST -- NO ORACLE NEEDED.
        # ============================================================
        maxit = int(D.TRAIN_MAX_ITERS)  # T
        rollout = predict_rollout_from_start_ae_predictor(  # ONE ROLLOUT SERVES BOTH TESTS BELOW
            td, enc, dec, device, steps=maxit, escape_r=D.ESCAPE_R,
        )  # rollout[s] = AE PREDICTION OF x_{s+2}, s = 0 .. maxit-2

        # ---- TEST 1: MACRO / EYEBALL -----------------------------------
        d_state = int((int(td.X_grid.shape[-1]) - 2) // 2)  # td.X_grid HAS +2 C CHANNELS, rollout DOES NOT

        Z_pred_final = rollout[-1]  # AE's PREDICTED x_{maxit}
        Z_true_final = td.X_grid[-1][..., :2 * d_state]  # DROP TRAILING C SO SHAPES MATCH
        save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                                  out_png=dirs["res"] / "rollout_from_start_final_mask.png",
                                  mode="mask")  # COMPARE VS gt_final_mask.png
        save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                                  out_png=dirs["res"] / "rollout_from_start_final_mag.png", mode="mag")

        final_m = next_step_prediction_metrics(Z_pred_final, Z_true_final)  # QUANTIFY THE EYEBALL TEST TOO
        print_metric_block(f"AE ROLLOUT FROM x1 -> x{maxit} (MACRO)", final_m)  # PRINT
        all_metrics.update({f"rollout_final_{key}": val for key, val in final_m.items()})  # SAVE

        # ---- TEST 2: QUANTITATIVE ---------------------------------------
        n_check = min(int(getattr(D, "PREDICT_ROLLOUT_CHECK_STEPS", 10)), rollout.shape[0])  # HOW MANY TO CHECK
        rollout_rel_l2: list[float] = []  # FOR THE CURVE PLOT

        for s in range(n_check):  # s AE STEPS BEYOND x1 -> COMPARES TO x_{s+2}
            true_iter = s + 2  # WHICH TRUE ITERATE THIS IS
            Z_pred_s = rollout[s]  # AE PREDICTION OF x_{true_iter}
            Z_true_s = td.X_grid[s + 1][..., :2 * d_state]  # DROP TRAILING C SO SHAPES MATCH

            m_s = next_step_prediction_metrics(Z_pred_s, Z_true_s)  # rel_l2, mse, fit
            print_metric_block(f"AE ROLLOUT FROM x1, {s + 1} STEP(S) IN (x{true_iter})", m_s)  # PRINT

            all_metrics[f"rollout_rel_l2_step_{s + 1:03d}"] = float(m_s["pred_rel_l2"])  # SAVE
            rollout_rel_l2.append(float(m_s["pred_rel_l2"]))  # COLLECT

        save_loss_curve(  # REUSE THE LOSS-CURVE PLOTTER FOR A rel_l2-VS-STEP PLOT
            rollout_rel_l2, dirs["res"] / "rollout_rel_l2_vs_step.png",
            "AE Rollout Relative L2 Error vs Steps Beyond x1",
        )

        write_metrics_txt(dirs["res"] / "metrics.txt", all_metrics)  # SAVE


# =============================== AE-ONLY-MULTI =============================
def run_ae_only_multi(device: torch.device) -> None:  # RUN AE-ONLY ON MANY MATRICES
    print("\n================ AE ONLY / MULTIPLE MATRICES ================\n")  # HEADER
    dirs = make_out_dirs("ae-only-multi")  # OUT DIRS

    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)  # ALL MATRICES
    train_idx, test_idx, td_train_list, td_test_list = _build_multi_train_test()  # BUILD (40/8)
    print("TRAIN COUNT:", int(train_idx.size), "TEST COUNT:", int(test_idx.size))  # LOG

    save_ground_truth_final_mask(td_train_list[0], D.ESCAPE_R, dirs["td"] / "train_example_gt_final_mask.png", scale=D.IMAGE_SCALE)  # TRAIN GT
    save_training_npz(dirs["td"] / "training_train_example.npz", td_train_list[0])  # SAVE EXAMPLE

    enc, dec, losses = train_autoencoder_predict_next_only(  # TRAIN MULTI AE PREDICTOR
        [td.X1 for td in td_train_list], [td.X2 for td in td_train_list],
        latent_dim=int(getattr(D, "AE_PRED_LATENT_DIM", D.LATENT_DIM)),  # LATENT
        epochs=int(getattr(D, "AE_PRED_EPOCHS", getattr(D, "AE_ONLY_EPOCHS", D.AE_EPOCHS))),  # EPOCHS
        batch_size=int(getattr(D, "AE_ONLY_BATCH_SIZE", D.AE_BATCH_SIZE)),  # BATCH
        lr=float(getattr(D, "AE_ONLY_LR", D.AE_LR)),  # LR
        device=device,
    )
    save_loss_curve(losses, dirs["res"] / "loss_curve.png", "AE Predictor Multiple Matrices Loss")  # LOSS

    k = int(D.PREDICT_EXTRA_STEPS)  # STEPS AHEAD
    metric_list = []  # STORE ONE-STEP METRICS
    pred_metric_list = []  # STORE NEXT-STEP METRICS

    for j, td_test in enumerate(td_test_list):  # TEST LOOP (HELD-OUT)
        A_test = A_all[int(test_idx[j])]  # TRUE MATRIX FOR THIS TEST CASE
        mm = ae_predictor_one_step_metrics(enc, dec, td_test.X1, td_test.X2, device)  # ONE-STEP METRICS

        Z_pred = predict_next_snapshot_ae_predictor(td_test, enc, dec, device, steps=k, escape_r=D.ESCAPE_R)  # MODEL x_{T+k}
        Z_true_next = iterate_true_next_snapshot(td_test, A_test, steps=k, escape_r=D.ESCAPE_R)  # TRUE x_{T+k}
        pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)  # PRED vs TRUE NEXT

        metric_list.append(mm)  # STORE
        pred_metric_list.append(pred_m)  # STORE
        print_metric_block(f"AE PREDICTOR TEST MATRIX {int(test_idx[j])} (ONE-STEP)", mm)  # PRINT
        print_metric_block(f"AE PREDICTOR TEST MATRIX {int(test_idx[j])} PREDICT (+{k})", pred_m)  # PRINT

        if j == 0:  # SAVE FIRST TEST VISUALS (GT + RECON + PRED + TRUE NEXT)
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_final_mask.png", scale=D.IMAGE_SCALE)  # GT MASK
            save_ground_truth_escape_iters(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_escape_iters.png")  # GT FRACTAL

            Z_recon = reconstruct_final_snapshot_ae_only(td_test, enc, dec, device)  # RECON
            save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_recon_final_mask.png", mode="mask")  # RECON MASK

            save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_pred_final_mask.png", mode="mask")  # PRED MASK
            save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_true_next_final_mask.png", mode="mask")  # TRUE NEXT MASK

            iters_test = teacher_forced_escape_iters_ae_predictor(td_test, enc, dec, device, D.ESCAPE_R)  # PRED FRACTAL
            save_escape_image(iters_test, max_iters=int(td_test.X_grid.shape[0]), out_png=dirs["res"] / "test_pred_escape_iters.png")  # SAVE FRACTAL

    mean_m = mean_metric_dict(metric_list)  # MEAN ONE-STEP
    mean_pred = mean_metric_dict(pred_metric_list)  # MEAN PREDICTION
    print_metric_block("AE PREDICTOR MEAN TEST (ONE-STEP)", mean_m)  # PRINT
    print_metric_block(f"AE PREDICTOR MEAN TEST PREDICT (+{k})", mean_pred)  # PRINT
    write_metrics_txt(dirs["res"] / "mean_test_metrics.txt", {**mean_m, **mean_pred, "predict_extra_steps": float(k)})  # SAVE


# =============================== DMD-ONLY-SINGLE ===========================
def run_dmd_only_single(device: torch.device) -> None:  # RUN RAW DMD ON ONE MATRIX
    print("\n================ DMD ONLY / SINGLE MATRIX ================\n")  # HEADER
    dirs = make_out_dirs("dmd-only-single")  # OUT DIRS

    A = load_one_A_matrix(D.A_DATA_DIR, source=D.SINGLE_MATRIX_SOURCE, index=D.SINGLE_MATRIX_INDEX)  # TRUE MATRIX
    td = _build_single_td()  # BUILD DATA

    save_training_npz(dirs["td"] / "training_single_matrix.npz", td)  # SAVE NPZ
    save_ground_truth_escape_iters(td, D.ESCAPE_R, dirs["td"] / "gt_escape_iters.png")  # GT FRACTAL
    save_ground_truth_final_mask(td, D.ESCAPE_R, dirs["td"] / "gt_final_mask.png", scale=D.IMAGE_SCALE)  # GT MASK

    dmd = fit_full_state_dmd_streamed(td.X1, td.X2, device=device)  # FIT RAW DMD
    rho = float(np.max(np.abs(np.linalg.eigvals(dmd.A.detach().cpu().numpy()))))  # SPECTRAL RADIUS
    print("DMD ONLY SINGLE SPECTRAL RADIUS:", rho)  # PRINT

    k = int(D.PREDICT_EXTRA_STEPS)  # STEPS AHEAD
    Z_pred = predict_next_snapshot_dmd_only(td, dmd, device, steps=k, escape_r=D.ESCAPE_R)  # MODEL x_{T+k}
    Z_true_next = iterate_true_next_snapshot(td, A, steps=k, escape_r=D.ESCAPE_R)  # TRUE x_{T+k}

    _save_final_mask_95(Z_pred, D.ESCAPE_R, dirs["res"] / "pred_final_mask.png", scale=D.IMAGE_SCALE)  # PRED MASK
    save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "pred_final_snapshot_mag.png", mode="mag")  # PRED MAG
    _save_final_mask_95(Z_true_next, D.ESCAPE_R, dirs["res"] / "true_next_final_mask.png", scale=D.IMAGE_SCALE)  # TRUE NEXT MASK

    m = dmd_only_one_step_metrics(dmd, td.X1, td.X2, device)  # ONE-STEP METRICS
    pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)  # PRED vs TRUE NEXT
    m["dmd_only_spectral_radius"] = rho  # ADD RHO
    print_metric_block("DMD ONLY SINGLE (ONE-STEP)", m)  # PRINT
    print_metric_block(f"DMD ONLY SINGLE PREDICT (+{k})", pred_m)  # PRINT
    write_metrics_txt(dirs["res"] / "metrics.txt", {**m, **pred_m, "predict_extra_steps": float(k)})  # SAVE


# =============================== DMD-ONLY-MULTI ============================
def run_dmd_only_multi(device: torch.device) -> None:  # RUN RAW DMD ON MANY MATRICES
    print("\n================ DMD ONLY / MULTIPLE MATRICES ================\n")  # HEADER
    dirs = make_out_dirs("dmd-only-multi")  # OUT DIRS

    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)  # ALL MATRICES
    train_idx, test_idx, td_train_list, td_test_list = _build_multi_train_test()  # BUILD (40/8)
    print("TRAIN COUNT:", int(train_idx.size), "TEST COUNT:", int(test_idx.size))  # LOG

    save_ground_truth_final_mask(td_train_list[0], D.ESCAPE_R, dirs["td"] / "train_example_gt_final_mask.png", scale=D.IMAGE_SCALE)  # TRAIN GT
    save_training_npz(dirs["td"] / "training_train_example.npz", td_train_list[0])  # SAVE EXAMPLE

    dmd = fit_full_state_dmd_streamed(  # FIT SHARED RAW DMD
        [td.X1 for td in td_train_list], [td.X2 for td in td_train_list], device=device,
    )
    rho = float(np.max(np.abs(np.linalg.eigvals(dmd.A.detach().cpu().numpy()))))  # SPECTRAL RADIUS
    print("DMD ONLY MULTI SPECTRAL RADIUS:", rho)  # PRINT

    k = int(D.PREDICT_EXTRA_STEPS)  # STEPS AHEAD
    metric_list = []  # STORE ONE-STEP METRICS
    pred_metric_list = []  # STORE NEXT-STEP METRICS

    for j, td_test in enumerate(td_test_list):  # TEST LOOP (HELD-OUT)
        A_test = A_all[int(test_idx[j])]  # TRUE MATRIX FOR THIS TEST CASE
        mm = dmd_only_one_step_metrics(dmd, td_test.X1, td_test.X2, device)  # ONE-STEP METRICS
        mm["dmd_only_spectral_radius"] = rho  # ADD RHO

        Z_pred = predict_next_snapshot_dmd_only(td_test, dmd, device, steps=k, escape_r=D.ESCAPE_R)  # MODEL x_{T+k}
        Z_true_next = iterate_true_next_snapshot(td_test, A_test, steps=k, escape_r=D.ESCAPE_R)  # TRUE x_{T+k}
        pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)  # PRED vs TRUE NEXT

        metric_list.append(mm)  # STORE
        pred_metric_list.append(pred_m)  # STORE
        print_metric_block(f"DMD ONLY TEST MATRIX {int(test_idx[j])} (ONE-STEP)", mm)  # PRINT
        print_metric_block(f"DMD ONLY TEST MATRIX {int(test_idx[j])} PREDICT (+{k})", pred_m)  # PRINT

        if j == 0:  # SAVE FIRST TEST VISUALS
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_final_mask.png", scale=D.IMAGE_SCALE)  # GT
            _save_final_mask_95(Z_pred, D.ESCAPE_R, dirs["res"] / "test_pred_final_mask.png", scale=D.IMAGE_SCALE)  # PRED MASK
            save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_pred_final_snapshot_mag.png", mode="mag")  # PRED MAG
            _save_final_mask_95(Z_true_next, D.ESCAPE_R, dirs["res"] / "test_true_next_final_mask.png", scale=D.IMAGE_SCALE)  # TRUE NEXT MASK

    mean_m = mean_metric_dict(metric_list)  # MEAN ONE-STEP
    mean_pred = mean_metric_dict(pred_metric_list)  # MEAN PREDICTION
    print_metric_block("DMD ONLY MEAN TEST (ONE-STEP)", mean_m)  # PRINT
    print_metric_block(f"DMD ONLY MEAN TEST PREDICT (+{k})", mean_pred)  # PRINT
    write_metrics_txt(dirs["res"] / "mean_test_metrics.txt", {**mean_m, **mean_pred, "predict_extra_steps": float(k)})  # SAVE