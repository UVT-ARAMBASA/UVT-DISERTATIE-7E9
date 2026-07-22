# ========================== mandelbrot_reconstruct.py =========================
from __future__ import annotations  # TYPE HINTS

# ================================= IMPORTS ==================================
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH
from PIL import Image  # PIL
import matplotlib.pyplot as plt  # PLOT

from utils import to_tensor  # YOUR HELPER

DEAD_PIXEL_TINT = (255, 205, 205)  # LIGHT RED/PINK, RGB

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

    X = np.zeros((P, feat_dim), dtype=np.float32)  # INIT FULL
    X[:, 0:state_dim] = C[:, 0:1].astype(np.float32)  # x1 RE IN ALL COMPONENTS
    X[:, state_dim:2 * state_dim] = C[:, 1:2].astype(np.float32)  # x1 IM IN ALL COMPONENTS
    X[:, 2 * state_dim] = C[:, 0].astype(np.float32)  # CR
    X[:, 2 * state_dim + 1] = C[:, 1].astype(np.float32)  # CI

    alive = np.ones((P,), dtype=bool)  # ACTIVE
    iters = np.zeros((P,), dtype=np.int32)  # FIRST ESCAPE ITER

    # TEST x1 FIRST, BECAUSE GROUND TRUTH COUNTS ESCAPE FROM THE FIRST STORED STATE
    zr0 = X[:, 0:state_dim]  # RE
    zi0 = X[:, state_dim:2 * state_dim]  # IM
    comp_mag2_0 = zr0 * zr0 + zi0 * zi0  # |z_i|^2
    max_mag2_0 = np.max(comp_mag2_0, axis=1)  # MAX OVER COMPONENTS
    esc0 = max_mag2_0 > r2  # ESCAPED AT ITER 1
    iters[esc0] = 1  # MARK ITER 1
    alive[esc0] = False  # DEACTIVATE

    X_t = to_tensor(X, device)  # TO DEVICE
    C_t = to_tensor(C.astype(np.float32), device)  # CONST C

    for k in range(2, int(max_iters) + 1):  # CONTINUE FROM x2..xT
        if not alive.any():  # DONE
            break  # STOP

        idx = np.where(alive)[0]  # ACTIVE IDX
        xk = X_t[idx]  # ACTIVE STATES

        zk = encoder(xk)  # ENC
        zk1 = dmd.predict(zk, steps=1)[-1]  # ONE STEP
        xk1 = decoder(zk1)  # DEC

        xk1[:, 2 * state_dim] = C_t[idx, 0]  # KEEP CR EXACT
        xk1[:, 2 * state_dim + 1] = C_t[idx, 1]  # KEEP CI EXACT

        x_cpu = xk1.detach().cpu().numpy().astype(np.float32)  # CPU
        zr_all = x_cpu[:, 0:state_dim]  # RE
        zi_all = x_cpu[:, state_dim:2 * state_dim]  # IM

        comp_mag2 = zr_all * zr_all + zi_all * zi_all  # |z_i|^2
        max_mag2 = np.max(comp_mag2, axis=1)  # MAX OVER COMPONENTS
        esc = max_mag2 > r2  # ESCAPED NOW

        bad = (~np.isfinite(zr_all)) | (~np.isfinite(zi_all))  # NaN/Inf
        if np.any(bad):  # FIX
            zr_all = np.where(np.isfinite(zr_all), zr_all, 0.0).astype(np.float32, copy=False)  # FIX RE
            zi_all = np.where(np.isfinite(zi_all), zi_all, 0.0).astype(np.float32, copy=False)  # FIX IM
            x_cpu[:, 0:state_dim] = zr_all  # WRITE RE
            x_cpu[:, state_dim:2 * state_dim] = zi_all  # WRITE IM

        xk1 = to_tensor(x_cpu, device)  # BACK
        X_t[idx] = xk1  # WRITE BACK

        escaped_idx = idx[esc]  # GLOBAL IDX
        iters[escaped_idx] = k  # FIRST ESCAPE ITER
        alive[escaped_idx] = False  # DEACTIVATE

    iters[alive] = int(max_iters)  # NEVER ESCAPED
    return iters.reshape(int(grid_n), int(grid_n))  # IMAGE

# =============================== IMAGE SAVE ==================================
def save_escape_image(  # SAVE PNG
    escape_iters: np.ndarray,  # (H,W)
    *,
    max_iters: int,  # MAX
    out_png: str | Path,  # PATH
    alive_mask: np.ndarray | None = None,  # NEW: (H,W) BOOL, SEE save_final_snapshot_image
) -> str:
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    esc = escape_iters.astype(np.float32)  # FLOAT
    norm = esc / float(max_iters)  # 0..1
    img = (255.0 * norm).clip(0, 255).astype(np.uint8)  # 8BIT
    img = 255 - img  # INVERT

    if alive_mask is None:  # OLD BEHAVIOUR, UNCHANGED
        im = Image.fromarray(img, mode="L")  # GRAY
    else:  # NEW: FLAG OUT-OF-DOMAIN PIXELS
        im = _tint_dead_pixels(img, alive_mask)  # RGB WITH DEAD_PIXEL_TINT

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
) -> np.ndarray:  # (H,W,2*D)
    P = int(C.shape[0])  # COUNT

    # IMPORTANT:
    # TRAINING STORES x1, x2, ..., xT  # NOT x0
    # x1 CORRESPONDS TO z1 = c (REPEATED IN ALL COMPONENTS) WHEN z0 = 0
    X = np.zeros((P, feat_dim), dtype=np.float32)  # INIT FULL
    X[:, 0:state_dim] = C[:, 0:1].astype(np.float32)  # x1 REAL PART IN ALL COMPONENTS
    X[:, state_dim:2 * state_dim] = C[:, 1:2].astype(np.float32)  # x1 IMAG PART IN ALL COMPONENTS
    X[:, 2 * state_dim] = C[:, 0].astype(np.float32)  # FIX CR
    X[:, 2 * state_dim + 1] = C[:, 1].astype(np.float32)  # FIX CI

    # IF TRAINING HAS T STORED STATES x1..xT, THEN FROM x1 WE ONLY NEED T-1 ROLLOUT STEPS
    n_roll = max(int(steps) - 1, 0)  # CORRECT STEP COUNT

    C_t_all = to_tensor(C.astype(np.float32), device)  # CONST C
    X_out = np.zeros((P, 2 * state_dim), dtype=np.float32)  # FINAL OUT

    for i0 in range(0, P, int(batch_size)):  # BATCH LOOP
        i1 = min(P, i0 + int(batch_size))  # BATCH END
        idx = slice(i0, i1)  # SLICE

        X_t = to_tensor(X[idx], device)  # START FROM x1
        C_t = C_t_all[idx]  # CONST C

        for _ in range(n_roll):  # ROLL x1 -> xT
            zk = encoder(X_t)  # ENC
            zk1 = dmd.predict(zk, steps=1)[-1]  # ONE STEP
            X_t = decoder(zk1)  # DEC

            X_t[:, 2 * state_dim] = C_t[:, 0]  # KEEP CR EXACT
            X_t[:, 2 * state_dim + 1] = C_t[:, 1]  # KEEP CI EXACT

            x_cpu = X_t.detach().cpu().numpy().astype(np.float32)  # CPU
            zr_all = x_cpu[:, 0:state_dim]  # RE
            zi_all = x_cpu[:, state_dim:2 * state_dim]  # IM

            comp_mag2 = zr_all * zr_all + zi_all * zi_all  # |z_i|^2
            comp_mag = np.sqrt(np.maximum(comp_mag2, 1e-30)).astype(np.float32)  # |z_i|
            bad = (~np.isfinite(comp_mag)) | (comp_mag > float(escape_r))  # BAD

            if np.any(bad):  # CLAMP
                safe_mag = np.where((comp_mag > 0.0) & np.isfinite(comp_mag), comp_mag, 1.0).astype(np.float32)  # SAFE
                scale = (float(escape_r) / safe_mag).astype(np.float32)  # SCALE
                zr_all = np.where(bad, zr_all * scale, zr_all).astype(np.float32)  # FIX RE
                zi_all = np.where(bad, zi_all * scale, zi_all).astype(np.float32)  # FIX IM
                x_cpu[:, 0:state_dim] = zr_all  # WRITE RE
                x_cpu[:, state_dim:2 * state_dim] = zi_all  # WRITE IM

            X_t = to_tensor(x_cpu, device)  # BACK TO DEVICE

        x_final = X_t.detach().cpu().numpy().astype(np.float32)  # FINAL
        X_out[idx, 0:state_dim] = x_final[:, 0:state_dim]  # SAVE RE
        X_out[idx, state_dim:2 * state_dim] = x_final[:, state_dim:2 * state_dim]  # SAVE IM

    return X_out.reshape(int(grid_n), int(grid_n), 2 * state_dim)  # RESHAPE

# ======================= NEW: FLAG OUT-OF-DOMAIN PIXELS =======================
def _tint_dead_pixels(img_l: np.ndarray, alive_mask: np.ndarray) -> Image.Image:  # GRAY -> RGB, DEAD PIXELS TINTED
    alive_mask = np.asarray(alive_mask, dtype=bool)  # BOOL
    if alive_mask.shape != img_l.shape:  # SANITY
        raise ValueError(f"alive_mask SHAPE {alive_mask.shape} != IMAGE SHAPE {img_l.shape}")  # ERROR

    rgb = np.stack([img_l, img_l, img_l], axis=-1).astype(np.uint8)  # GRAY -> RGB
    tint = np.array(DEAD_PIXEL_TINT, dtype=np.uint8)  # TINT COLOUR
    rgb[~alive_mask] = tint  # PAINT DEAD PIXELS
    return Image.fromarray(rgb, mode="RGB")  # RETURN

# ======================= SAVE FINAL SNAPSHOT IMAGE ===========================
def save_final_snapshot_image(  # SAVE FINAL PNG
    Z_final: np.ndarray,  # (H,W,2*D)
    *,
    escape_r: float,  # SCALE
    out_png: str | Path,  # PATH
    mode: str = "mag",  # "mag" OR "angle" OR "mask"
    alive_mask: np.ndarray | None = None,  # NEW: (H,W) BOOL -- WHERE THE MODEL IS ACTUALLY IN-DOMAIN.
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

    if alive_mask is None:  # OLD BEHAVIOUR, UNCHANGED (E.G. GROUND-TRUTH IMAGES)
        Image.fromarray(img, mode="L").save(out_png)  # SAVE
    else:  # NEW: FLAG OUT-OF-DOMAIN PIXELS INSTEAD OF LETTING THEM POLLUTE THE IMAGE
        _tint_dead_pixels(img, alive_mask).save(out_png)  # SAVE RGB

    return str(out_png)  # RETURN