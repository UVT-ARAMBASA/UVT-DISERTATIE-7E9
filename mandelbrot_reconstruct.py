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
    *,
    encoder,  # ENC
    decoder,  # DEC
    dmd,  # DMD
    C: np.ndarray,  # (P,2)
    grid_n: int,  # RES
    max_iters: int,  # ITERS
    escape_r: float,  # ESCAPE R
    device: torch.device,  # DEVICE
) -> np.ndarray:  # (grid_n,grid_n)
    P = int(C.shape[0])  # COUNT
    r2 = float(escape_r) * float(escape_r)  # R^2

    # INIT x0=[zr,zi,cr,ci]  # AUG STATE
    X = np.zeros((P, 4), dtype=np.float32)  # INIT
    X[:, 2:4] = C.astype(np.float32)  # SET C

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
        xk1[:, 2:4] = C_t[idx]  # FIX

        # CPU COPY (FOR ESCAPE TEST + SANITY)
        x_cpu = xk1.detach().cpu().numpy().astype(np.float32)

        zr = x_cpu[:, 0]
        zi = x_cpu[:, 1]

        # ESCAPE TEST MUST USE RAW (BEFORE ANY CLAMP-TO-R)
        mag2 = zr * zr + zi * zi
        esc = mag2 > r2

        # SANITISE ONLY NaN/Inf (DO NOT CLAMP TO ESCAPE R HERE)
        bad = (~np.isfinite(zr)) | (~np.isfinite(zi))
        if np.any(bad):
            zr = np.where(bad, 0.0, zr).astype(np.float32, copy=False)
            zi = np.where(bad, 0.0, zi).astype(np.float32, copy=False)
            x_cpu[:, 0] = zr
            x_cpu[:, 1] = zi

        # PUSH BACK
        xk1 = to_tensor(x_cpu, device)
        X_t[idx] = xk1

        # MARK ESCAPED POINTS
        escaped_idx = idx[esc]
        iters[escaped_idx] = k
        alive[escaped_idx] = False

    iters[alive] = int(max_iters)  # STABLE
    return iters.reshape(int(grid_n), int(grid_n))  # IMAGE


@torch.no_grad()  # NO GRADS
def reconstruct_final_state(  # FINAL ITER ONLY
    *,
    encoder,
    decoder,
    dmd,
    C: np.ndarray,  # (P,2)
    max_iters: int,  # ITERS
    escape_r: float,  # CLAMP R
    device: torch.device,
) -> np.ndarray:  # (P,4)
    P = int(C.shape[0])  # COUNT

    # INIT x0=[zr,zi,cr,ci]
    X = np.zeros((P, 4), dtype=np.float32)
    X[:, 2:4] = C.astype(np.float32)

    X_t = to_tensor(X, device)
    C_t = to_tensor(C.astype(np.float32), device)

    for _k in range(int(max_iters)):  # FIXED STEPS (NO ESCAPE IMAGE)
        zk = encoder(X_t)
        zk1 = dmd.predict(zk, steps=1)[-1]
        X_t = decoder(zk1)

        # FORCE C CONSTANT
        X_t[:, 2:4] = C_t

        # CLAMP z TO ESCAPE R (UPPER BOUND)
        x_cpu = X_t.detach().cpu().numpy().astype(np.float32)
        zr = x_cpu[:, 0]
        zi = x_cpu[:, 1]
        zr, zi = _cap_to_escape(zr, zi, float(escape_r))
        x_cpu[:, 0] = zr
        x_cpu[:, 1] = zi
        X_t = to_tensor(x_cpu, device)

    return X_t.detach().cpu().numpy().astype(np.float32)  # (P,4)

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
    *,
    encoder,  # ENC
    decoder,  # DEC
    dmd,  # DMD (USES READ MATRIX INSIDE)
    C: np.ndarray,  # (P,2)
    grid_n: int,  # RES
    steps: int,  # HOW MANY ITERATIONS TO APPLY
    escape_r: float,  # CLIP FOR NUMERIC SAFETY
    device: torch.device,  # DEVICE
    batch_size: int = 200000,  # BATCH (GPU RAM)
) -> np.ndarray:  # (grid_n,grid_n,2)  # FINAL (zr,zi)
    P = int(C.shape[0])  # COUNT

    # INIT x0=[zr,zi,cr,ci]  # AUG STATE
    X = np.zeros((P, 4), dtype=np.float32)  # INIT
    X[:, 2:4] = C.astype(np.float32)  # SET C

    C_t_all = to_tensor(C.astype(np.float32), device)  # CONST C
    X_out = np.zeros((P, 2), dtype=np.float32)  # FINAL Z (CPU)

    # BATCH LOOP
    for i0 in range(0, P, int(batch_size)):  # LOOP BATCHES
        i1 = min(P, i0 + int(batch_size))  # END
        idx = slice(i0, i1)  # SLICE

        X_t = to_tensor(X[idx], device)  # TO TENSOR
        C_t = C_t_all[idx]  # CONST C

        for _k in range(int(steps)):  # APPLY LEARNED DYNAMICS MANY TIMES
            zk = encoder(X_t)  # ENC
            zk1 = dmd.predict(zk, steps=1)[-1]  # 1 STEP USING "READ MATRIX"
            X_t = decoder(zk1)  # DEC

            # FORCE C CONSTANT  # KEEP C
            X_t[:, 2:4] = C_t  # FIX

            # CLAMP Z (UPPER BOUND)  # MATCH YOUR "CAP TO ESCAPE R"
            x_cpu = X_t.detach().cpu().numpy().astype(np.float32)  # CPU
            zr = x_cpu[:, 0]  # RE
            zi = x_cpu[:, 1]  # IM
            zr, zi = _cap_to_escape(zr, zi, float(escape_r))  # CLAMP
            x_cpu[:, 0] = zr  # WRITE
            x_cpu[:, 1] = zi  # WRITE
            X_t = to_tensor(x_cpu, device)  # BACK

        # STORE FINAL Z
        x_final = X_t.detach().cpu().numpy().astype(np.float32)  # CPU
        X_out[idx, 0] = x_final[:, 0]  # zr
        X_out[idx, 1] = x_final[:, 1]  # zi

    return X_out.reshape(int(grid_n), int(grid_n), 2)  # (H,W,2)

# ======================= SAVE FINAL SNAPSHOT IMAGE ===========================
def save_final_snapshot_image(  # SAVE FINAL PNG
    Z_final: np.ndarray,  # (H,W,2)
    *,
    escape_r: float,  # SCALE
    out_png: str | Path,  # PATH
    mode: str = "mag",  # "mag" OR "angle"
) -> str:
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    zr = Z_final[..., 0].astype(np.float32, copy=False)  # RE
    zi = Z_final[..., 1].astype(np.float32, copy=False)  # IM

    if str(mode).lower() == "angle":  # PHASE IMAGE
        ang = np.arctan2(zi, zr).astype(np.float32)  # [-pi,pi]
        norm = (ang + np.pi) / (2.0 * np.pi)  # [0,1]
        img = (255.0 * norm).clip(0, 255).astype(np.uint8)  # 8BIT
    else:  # MAG IMAGE (DEFAULT)
        mag = np.sqrt(zr * zr + zi * zi).astype(np.float32)  # |Z|
        mag = np.where(mag > float(escape_r), float(escape_r), mag).astype(np.float32)  # CLIP
        norm = mag / float(escape_r)  # [0,1] FOR DISPLAY ONLY
        img = (255.0 * norm).clip(0, 255).astype(np.uint8)  # 8BIT

    im = Image.fromarray(img, mode="L")  # GRAY
    im.save(out_png)  # SAVE
    return str(out_png)  # RETURN