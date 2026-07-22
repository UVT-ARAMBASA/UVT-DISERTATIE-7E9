# =========================== eval_matrix_dmd_ae.py ===========================
from __future__ import annotations  # TYPE HINTS

# ================================= IMPORTS ==================================
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH
from PIL import Image  # PIL
import matplotlib.pyplot as plt  # PLOT

from utils import to_tensor  # HELPER
import defines as D  # DEFAULT FOR log_scale

# ============================== LOSS CURVE ===================================
def save_loss_curve(  # SAVE LOSS
    losses: list[float],
    out_png: str | Path,
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    *,
    log_scale: bool | None = None,  # None = USE defines.LOSS_CURVES_LOG_SCALE
    extra_series: dict[str, list[float]] | None = None,  # NEW: E.G. {"Validation": val_losses}
    primary_label: str | None = None,  # NEW: LEGEND LABEL FOR `losses` (ONLY SHOWN IF extra_series GIVEN)
) -> str:
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    if log_scale is None:  # USE PROJECT DEFAULT
        log_scale = bool(getattr(D, "LOSS_CURVES_LOG_SCALE", True))  # DEFAULT

    xs = np.arange(1, len(losses) + 1)  # EPOCH AXIS
    ys = np.asarray(losses, dtype=np.float32)  # CURVE

    plt.figure()  # FIG
    plot_fn = plt.semilogy if log_scale else plt.plot  # LOG OR LINEAR

    label = primary_label if extra_series else None  # ONLY LABEL IF THERE'S A LEGEND TO SHOW
    if log_scale:  # semilogy SILENTLY DROPS <= 0 VALUES -- CLIP INSTEAD OF HIDING THEM
        ys_plot = np.where(ys > 0, ys, np.nan)  # NON-POSITIVE -> GAP, NOT A CRASH
        plot_fn(xs, ys_plot, label=label)  # CURVE
    else:
        plot_fn(xs, ys, label=label)  # CURVE

    if extra_series:  # NEW: OVERLAY EXTRA CURVES (E.G. VALIDATION LOSS)
        for name, series in extra_series.items():  # EACH EXTRA CURVE
            series = np.asarray(series, dtype=np.float32)  # ARRAY
            xs_e = np.arange(1, len(series) + 1)  # ITS OWN X AXIS (SAME LENGTH, USUALLY)
            if log_scale:  # SAME NaN-GAP TREATMENT
                series = np.where(series > 0, series, np.nan)  # SKIP NON-POSITIVE
            plot_fn(xs_e, series, label=name)  # OVERLAY
        plt.legend()  # SHOW LEGEND ONLY WHEN THERE'S >1 SERIES

    plt.xlabel(xlabel)  # X
    plt.ylabel(ylabel)  # Y
    plt.title(title)  # TITLE
    plt.tight_layout()  # TIGHT
    plt.savefig(out_png, dpi=200)  # SAVE
    plt.close()  # CLOSE
    return str(out_png)  # RETURN

# =============================== METRICS =====================================
@torch.no_grad()  # NO GRAD
def autoencoder_reconstruction_metrics(encoder, decoder, X: np.ndarray, device: torch.device, batch_size: int = 50000) -> dict:  # AE METRICS
    N = int(X.shape[0])  # ROWS
    sse = 0.0  # SUM SQ ERROR
    sq_true = 0.0  # SUM SQ TRUE

    for i0 in range(0, N, int(batch_size)):  # BATCH LOOP
        i1 = min(N, i0 + int(batch_size))  # END
        xb = to_tensor(X[i0:i1], device)  # TO TENSOR
        xh_np = decoder(encoder(xb)).detach().cpu().numpy().astype(np.float32)  # RECON
        err = xh_np - X[i0:i1].astype(np.float32)  # ERR
        sse += float(np.sum(err * err))  # ACCUM
        sq_true += float(np.sum(X[i0:i1].astype(np.float32) ** 2))  # ACCUM

    mse = sse / max(N * X.shape[1], 1)  # MSE
    rel_l2 = float(np.sqrt(sse) / max(np.sqrt(sq_true), 1e-12))  # REL
    fit = float(1.0 - rel_l2)  # FIT
    return {"ae_mse": mse, "ae_rel_l2": rel_l2, "ae_fit": fit}  # RETURN

# ========================= NEW: ALIVE-ROW MASK / ALIVE-ONLY AE METRICS =======
def _alive_row_mask(X: np.ndarray, escape_r: float) -> np.ndarray:  # PER-ROW HEURISTIC (FALLBACK ONLY, SEE ABOVE)
    X = np.asarray(X, dtype=np.float32)  # ARRAY
    feat_dim = int(X.shape[1])  # FEAT
    d = int((feat_dim - 2) // 2)  # STATE DIM

    zr = X[:, 0:d]  # REAL
    zi = X[:, d:2 * d]  # IMAG

    mag2 = zr * zr + zi * zi  # MAG2
    max_mag2 = np.max(np.where(np.isfinite(mag2), mag2, np.inf), axis=1)  # MAX

    finite = np.isfinite(zr).all(axis=1) & np.isfinite(zi).all(axis=1)  # FINITE
    alive = finite & (max_mag2 < 0.999 * float(escape_r) * float(escape_r))  # STRICTLY INSIDE

    return alive  # RETURN


def _exact_trained_row_mask(td) -> np.ndarray:  # EXACT ROW MASK OVER td.X, MATCHING X1/X2 MEMBERSHIP
    meta = getattr(td, "meta", None) or {}  # META DICT
    alive_grid = meta.get("alive_mask_grid", None)  # (H,W) BOOL, PER GRID POINT
    T = meta.get("max_iters", None)  # HOW MANY STORED TIME STEPS

    if alive_grid is None or T is None:  # CAN'T BUILD THE EXACT MASK
        classify_r = float(meta.get("classify_r", 2.0))  # BEST-EFFORT FALLBACK
        return _alive_row_mask(td.X, classify_r)  # APPROXIMATION

    alive_flat = np.asarray(alive_grid, dtype=bool).reshape(-1)  # (P,)
    P = int(alive_flat.shape[0])  # GRID POINT COUNT
    T = int(T)  # STEPS

    if int(td.X.shape[0]) != T * P:  # SHAPE GUARD -- td.X MUST BE (T*P, feat), t-MAJOR
        classify_r = float(meta.get("classify_r", 2.0))  # BEST-EFFORT FALLBACK
        return _alive_row_mask(td.X, classify_r)  # APPROXIMATION

    return np.tile(alive_flat, T)  # (T*P,) EXACT MASK


@torch.no_grad()  # NO GRAD
def autoencoder_reconstruction_metrics_alive(  # AE METRICS, EXACTLY-TRAINED-ON ROWS ONLY
    encoder, decoder, td, device: torch.device, batch_size: int = 50000,
) -> dict:
    X = np.asarray(td.X)  # FULL GRID ARRAY
    mask = _exact_trained_row_mask(td)  # EXACT (OR BEST-EFFORT) MEMBERSHIP
    n_alive = int(np.count_nonzero(mask))  # HOW MANY

    if n_alive == 0:  # NOTHING TO SCORE (SHOULDN'T HAPPEN IF TRAINING SUCCEEDED)
        return {  # NaN RATHER THAN A MISLEADING 0/1
            "ae_alive_mse": float("nan"), "ae_alive_rel_l2": float("nan"), "ae_alive_fit": float("nan"),
            "ae_alive_n": 0.0, "ae_alive_n_total": float(X.shape[0]),
        }

    base = autoencoder_reconstruction_metrics(encoder, decoder, X[mask], device, batch_size=batch_size)  # REUSE
    return {  # RENAME + ADD COUNTS
        "ae_alive_mse": base["ae_mse"],
        "ae_alive_rel_l2": base["ae_rel_l2"],
        "ae_alive_fit": base["ae_fit"],
        "ae_alive_n": float(n_alive),
        "ae_alive_n_total": float(X.shape[0]),
    }

@torch.no_grad()  # NO GRAD
def dmd_one_step_metrics(encoder, decoder, dmd, X1: np.ndarray, X2: np.ndarray, device: torch.device, batch_size: int = 50000) -> dict:  # DMD ONE STEP
    N = int(X1.shape[0])  # ROWS
    sse = 0.0  # SUM SQ ERROR
    sq_true = 0.0  # SUM SQ TRUE

    for i0 in range(0, N, int(batch_size)):  # BATCH LOOP
        i1 = min(N, i0 + int(batch_size))  # END
        x1b = to_tensor(X1[i0:i1], device)  # LEFT
        z1 = encoder(x1b)  # ENC
        z2h = dmd.predict(z1, steps=1)[-1]  # ONE STEP
        x2h_np = decoder(z2h).detach().cpu().numpy().astype(np.float32)  # DEC
        err = x2h_np - X2[i0:i1].astype(np.float32)  # ERR
        sse += float(np.sum(err * err))  # ACCUM
        sq_true += float(np.sum(X2[i0:i1].astype(np.float32) ** 2))  # ACCUM

    mse = sse / max(N * X1.shape[1], 1)  # MSE
    rel_l2 = float(np.sqrt(sse) / max(np.sqrt(sq_true), 1e-12))  # REL
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

# ============= ADDED in V33 ===============
@torch.no_grad()  # NO GRAD
def predict_rollout_from_start_ae_dmd(  # AE+DMD ROLLOUT FROM TRUE x1
    td,
    encoder,
    decoder,
    dmd,
    device,
    *,
    steps: int,
    escape_r: float,
    batch_size: int = 50000,
) -> np.ndarray:  # AE+DMD x_2 ... x_{steps} FROM THE TRUE ANCHOR x_1
    # START FROM x1 (TRUE FIRST STORED STATE -- z0=0 SO x1 = c IN ALL COMPONENTS)
    # SAME CONVENTION AS predict_rollout_from_start_ae_predictor, BUT STEPS
    # THROUGH THE FITTED LATENT DMD INSTEAD OF USING THE DECODER DIRECTLY.
    if td.X_grid is None:  # NEED GRID
        raise ValueError("AE+DMD ROLLOUT-FROM-START NEEDS td.X_grid")  # ERROR

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

        for s in range(n_roll):  # ROLL x1 -> x_{steps} THROUGH LATENT DMD
            zk = encoder(X_t)  # ENC
            zk1 = dmd.predict(zk, steps=1)[-1]  # ONE DMD STEP
            X_t = decoder(zk1)  # DEC
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
    zi = td.X_grid[:, :, :, d:2 * d]
    comp_mag2 = zr * zr + zi * zi
    max_mag2 = np.max(comp_mag2, axis=-1)
    escaped = max_mag2 >= r2

    first_escape = np.argmax(escaped, axis=0).astype(np.int32) + 1
    never_escaped = ~np.any(escaped, axis=0)
    first_escape[never_escaped] = T

    img = (255.0 * (1.0 - first_escape.astype(np.float32) / float(T))).clip(0, 255).astype(np.uint8)  # GRAY
    Image.fromarray(img, mode="L").save(out_png)
    return str(out_png)