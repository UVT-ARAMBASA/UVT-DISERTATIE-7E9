# ========================== mandelbrot_reconstruct.py =========================
from __future__ import annotations  # TYPE HINTS

# ================================= IMPORTS ==================================
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH
from PIL import Image  # PIL
import matplotlib.pyplot as plt  # PLOT

from utils import to_tensor  # YOUR HELPER

# =============================== GRID BUILDER ================================
def build_c_grid(  # MAKE GRID
    *,
    c_re_min: float,  # RE MIN
    c_re_max: float,  # RE MAX
    c_im_min: float,  # IM MIN
    c_im_max: float,  # IM MAX
    grid_n: int,  # RES
) -> np.ndarray:  # (P,2)
    xs = np.linspace(c_re_min, c_re_max, grid_n, dtype=np.float32)  # X
    ys = np.linspace(c_im_min, c_im_max, grid_n, dtype=np.float32)  # Y
    C = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)  # FLAT
    return C  # RETURN

# =============================== CLAMP =======================================
def _cap_to_escape(zr: np.ndarray, zi: np.ndarray, escape_r: float):  # CLAMP Z
    r = float(escape_r)  # R
    mag = np.sqrt(zr * zr + zi * zi).astype(np.float32)  # |Z|
    bad = (~np.isfinite(mag)) | (mag > r)  # BAD
    if not np.any(bad):  # OK
        return zr, zi  # RETURN
    mag_safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)  # SAFE
    s = (r / mag_safe).astype(np.float32)  # SCALE
    zr2 = np.where(bad, zr * s, zr).astype(np.float32)  # RE
    zi2 = np.where(bad, zi * s, zi).astype(np.float32)  # IM
    return zr2, zi2  # RETURN

# =========================== RECONSTRUCT MANDELBROT ==========================
@torch.no_grad()  # NO GRADS
def reconstruct_mandelbrot(  # RECON SET
    *,  # KWONLY
    encoder,  # ENC
    decoder,  # DEC
    dmd,  # DMD
    C: np.ndarray,  # (P,2)
    grid_n: int,  # RES
    max_iters: int,  # ITERS
    escape_r: float,  # ESCAPE R
    device: torch.device,  # DEVICE
    state_dim: int,  # STATE DIM
    feat_dim: int,  # FEATURE DIM
) -> np.ndarray:  # (grid_n,grid_n)
    P = int(C.shape[0])  # COUNT
    r2 = float(escape_r) * float(escape_r)  # R^2

    # INIT FULL FEATURE VECTOR  # MATCH TRAINING
    X = np.zeros((P, feat_dim), dtype=np.float32)  # INIT FULL
    X[:, 0:state_dim] = C[:, 0:1].astype(np.float32)  # SET Z0 RE IN ALL COMPONENTS
    X[:, state_dim:2 * state_dim] = C[:, 1:2].astype(np.float32)  # SET Z0 IM IN ALL COMPONENTS
    X[:, 2 * state_dim] = C[:, 0].astype(np.float32)  # SET CR
    X[:, 2 * state_dim + 1] = C[:, 1].astype(np.float32)  # SET CI

    alive = np.ones((P,), dtype=bool)  # ACTIVE
    iters = np.zeros((P,), dtype=np.int32)  # ESC ITER

    X_t = to_tensor(X, device)  # TO TENSOR
    C_t = to_tensor(C.astype(np.float32), device)  # CONST C

    for k in range(1, int(max_iters) + 1):  # LOOP
        if not alive.any():  # DONE
            break  # STOP

        idx = np.where(alive)[0]  # IDX
        xk = X_t[idx]  # ACTIVE

        zk = encoder(xk)  # ENC
        zk1 = dmd.predict(zk, steps=1)[-1]  # 1 STEP
        xk1 = decoder(zk1)  # DEC

        # FORCE C CONSTANT  # KEEP C
        xk1[:, 2 * state_dim] = C_t[idx, 0]  # FIX CR
        xk1[:, 2 * state_dim + 1] = C_t[idx, 1]  # FIX CI

        # CPU COPY (FOR ESCAPE TEST + SANITY)
        x_cpu = xk1.detach().cpu().numpy().astype(np.float32)

        zr_all = x_cpu[:, 0:state_dim]  # ALL REAL COMPONENTS
        zi_all = x_cpu[:, state_dim:2 * state_dim]  # ALL IMAG COMPONENTS

        # ESCAPE TEST ON MAX COMPONENT
        comp_mag2 = zr_all * zr_all + zi_all * zi_all  # COMPONENTWISE |z_i|^2
        max_mag2 = np.max(comp_mag2, axis=1)  # MAX COMPONENT |z_i|^2
        esc = max_mag2 > r2  # ESCAPED MASK

        # SANITISE ONLY NaN/Inf
        bad = (~np.isfinite(zr_all)) | (~np.isfinite(zi_all))  # BAD VALUES
        if np.any(bad):  # FIX IF NEEDED
            zr_all = np.where(np.isfinite(zr_all), zr_all, 0.0).astype(np.float32, copy=False)  # FIX RE
            zi_all = np.where(np.isfinite(zi_all), zi_all, 0.0).astype(np.float32, copy=False)  # FIX IM
            x_cpu[:, 0:state_dim] = zr_all  # WRITE ALL RE
            x_cpu[:, state_dim:2 * state_dim] = zi_all  # WRITE ALL IM

        # PUSH BACK
        xk1 = to_tensor(x_cpu, device)
        X_t[idx] = xk1

        # MARK ESCAPED POINTS
        escaped_idx = idx[esc]
        iters[escaped_idx] = k
        alive[escaped_idx] = False

    iters[alive] = int(max_iters)  # STABLE
    return iters.reshape(int(grid_n), int(grid_n))  # IMAGE


# =============================== IMAGE SAVE ==================================
def save_escape_image(  # SAVE PNG
    escape_iters: np.ndarray,  # (H,W)
    *,
    max_iters: int,  # MAX
    out_png: str | Path,  # PATH
) -> str:
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    esc = escape_iters.astype(np.float32)  # FLOAT
    norm = esc / float(max_iters)  # 0..1
    img = (255.0 * norm).clip(0, 255).astype(np.uint8)  # 8BIT
    img = 255 - img  # INVERT
    im = Image.fromarray(img, mode="L")  # GRAY
    im.save(out_png)  # SAVE
    return str(out_png)  # RETURN

# =============================== OPTIONAL PLOT ===============================
def save_and_show_plot(  # PLOT
    escape_iters: np.ndarray,  # DATA
    *,
    out_png: str | Path,  # PATH
    show: bool,  # SHOW
) -> str:
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    plt.figure()  # FIG
    plt.imshow(escape_iters, origin="lower")  # SHOW
    plt.colorbar()  # BAR
    plt.tight_layout()  # TIGHT
    plt.savefig(out_png, dpi=200)  # SAVE
    if show:  # SHOW
        plt.show()  # DISPLAY
    plt.close()  # CLOSE
    return str(out_png)  # RETURN
# ======================= RECONSTRUCT FINAL SNAPSHOT ==========================
@torch.no_grad()  # NO GRADS
def reconstruct_final_snapshot(  # FINAL STATE IMAGE
    *,  # KWONLY
    encoder,  # ENC
    decoder,  # DEC
    dmd,  # DMD
    C: np.ndarray,  # (P,2)
    grid_n: int,  # RES
    steps: int,  # STEPS
    escape_r: float,  # ESCAPE
    device: torch.device,  # DEVICE
    batch_size: int = 200000,  # BATCH
    state_dim: int,  # STATE DIM
    feat_dim: int,  # FEATURE DIM
) -> np.ndarray:  # (H,W,2)
    P = int(C.shape[0])  # COUNT

    X = np.zeros((P, feat_dim), dtype=np.float32)  # INIT FULL
    X[:, 2 * state_dim] = C[:, 0].astype(np.float32)  # SET CR
    X[:, 2 * state_dim + 1] = C[:, 1].astype(np.float32)  # SET CI

    C_t_all = to_tensor(C.astype(np.float32), device)  # CONST C
    X_out = np.zeros((P, 2 * state_dim), dtype=np.float32)  # FINAL OUT ALL COMPONENTS

    for i0 in range(0, P, int(batch_size)):  # BATCH LOOP
        i1 = min(P, i0 + int(batch_size))  # BATCH END
        idx = slice(i0, i1)  # SLICE

        X_t = to_tensor(X[idx], device)  # TO TENSOR
        C_t = C_t_all[idx]  # CONST C

        for _k in range(int(steps)):  # APPLY STEPS
            zk = encoder(X_t)  # ENC
            zk1 = dmd.predict(zk, steps=1)[-1]  # ONE STEP
            X_t = decoder(zk1)  # DEC

            X_t[:, 2 * state_dim] = C_t[:, 0]  # FIX CR
            X_t[:, 2 * state_dim + 1] = C_t[:, 1]  # FIX CI

            x_cpu = X_t.detach().cpu().numpy().astype(np.float32)  # CPU
            zr_all = x_cpu[:, 0:state_dim]  # ALL RE
            zi_all = x_cpu[:, state_dim:2 * state_dim]  # ALL IM

            comp_mag2 = zr_all * zr_all + zi_all * zi_all  # COMPONENTWISE |z_i|^2
            comp_mag = np.sqrt(np.maximum(comp_mag2, 1e-30)).astype(np.float32)  # COMPONENTWISE |z_i|
            bad = (~np.isfinite(comp_mag)) | (comp_mag > float(escape_r))  # BAD MASK

            if np.any(bad):  # FIX IF NEEDED
                safe_mag = np.where((comp_mag > 0.0) & np.isfinite(comp_mag), comp_mag, 1.0).astype(
                    np.float32)  # SAFE MAG
                scale = (float(escape_r) / safe_mag).astype(np.float32)  # SCALE
                zr_all = np.where(bad, zr_all * scale, zr_all).astype(np.float32)  # CLAMP RE
                zi_all = np.where(bad, zi_all * scale, zi_all).astype(np.float32)  # CLAMP IM
                x_cpu[:, 0:state_dim] = zr_all  # WRITE ALL RE
                x_cpu[:, state_dim:2 * state_dim] = zi_all  # WRITE ALL IM

            X_t = to_tensor(x_cpu, device)  # BACK

            x_final = X_t.detach().cpu().numpy().astype(np.float32)  # CPU FINAL
            X_out[idx, 0:state_dim] = x_final[:, 0:state_dim]  # SAVE ALL RE
            X_out[idx, state_dim:2 * state_dim] = x_final[:, state_dim:2 * state_dim]  # SAVE ALL IM

    return X_out.reshape(int(grid_n), int(grid_n), 2 * state_dim)  # RESHAPE

# ======================= SAVE FINAL SNAPSHOT IMAGE ===========================
def save_final_snapshot_image(  # SAVE FINAL PNG
    Z_final: np.ndarray,  # (H,W,2*D)
    *,
    escape_r: float,  # SCALE
    out_png: str | Path,  # PATH
    mode: str = "mag",  # "mag" OR "angle" OR "mask"
) -> str:
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    state_dim = int(Z_final.shape[-1] // 2)  # STATE DIM
    zr = Z_final[..., 0:state_dim].astype(np.float32, copy=False)  # ALL RE
    zi = Z_final[..., state_dim:2 * state_dim].astype(np.float32, copy=False)  # ALL IM
    comp_mag2 = zr * zr + zi * zi  # ALL MAG2
    max_mag = np.sqrt(np.max(comp_mag2, axis=-1)).astype(np.float32)  # MAX MAG OVER COMPONENTS

    if str(mode).lower() == "mask":  # BOUNDED MASK
        img = ((max_mag < float(escape_r)).astype(np.uint8) * 255)  # BW
    elif str(mode).lower() == "angle":  # PHASE OF FIRST COMPONENT
        ang = np.arctan2(zi[..., 0], zr[..., 0]).astype(np.float32)  # [-pi,pi]
        norm = (ang + np.pi) / (2.0 * np.pi)  # [0,1]
        img = (255.0 * norm).clip(0, 255).astype(np.uint8)  # 8BIT
    else:  # MAG IMAGE
        mag = np.minimum(max_mag, float(escape_r)).astype(np.float32)  # CLIP
        norm = mag / float(escape_r)  # [0,1]
        img = (255.0 * norm).clip(0, 255).astype(np.uint8)  # 8BIT

    Image.fromarray(img, mode="L").save(out_png)  # SAVE
    return str(out_png)  # RETURN