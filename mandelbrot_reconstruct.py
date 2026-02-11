# ========================== mandelbrot_reconstruct.py =========================
from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================= IMPORTS ==================================
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH
from PIL import Image  # PIL IMAGE
import matplotlib.pyplot as plt  # PLOT

from utils import to_tensor  # YOUR HELPER

# =============================== GRID BUILDER ================================
def build_c_grid(  # MAKE GRID
    *,
    c_re_min: float,  # RE MIN
    c_re_max: float,  # RE MAX
    c_im_min: float,  # IM MIN
    c_im_max: float,  # IM MAX
    grid_n: int,  # RESOLUTION
) -> np.ndarray:  # RETURNS (grid_n*grid_n,2)
    xs = np.linspace(c_re_min, c_re_max, grid_n, dtype=np.float32)  # X AXIS
    ys = np.linspace(c_im_min, c_im_max, grid_n, dtype=np.float32)  # Y AXIS
    C = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)  # FLAT GRID
    return C  # RETURN

# ============================= NORMALISE HELPERS =============================
def _norm_forward(x: np.ndarray, mn: float, mx: float) -> np.ndarray:  # SCALE TO 0..1
    return ((x - mn) / (mx - mn + 1e-12)).astype(np.float32)  # SCALE

def _norm_inverse(x: np.ndarray, mn: float, mx: float) -> np.ndarray:  # SCALE BACK
    return (x * (mx - mn + 1e-12) + mn).astype(np.float32)  # UNSCALE

# =========================== RECONSTRUCT MANDELBROT ==========================
@torch.no_grad()  # NO GRADS
def reconstruct_mandelbrot(  # RECONSTRUCT SET
    *,
    encoder,  # ENCODER MODEL
    decoder,  # DECODER MODEL
    dmd,  # DMD MODEL
    C: np.ndarray,  # (P,2) GRID
    grid_n: int,  # RESOLUTION
    max_iters: int,  # ITER COUNT
    escape_r: float,  # ESCAPE RADIUS
    mn: float,  # NORMALISE MIN
    mx: float,  # NORMALISE MAX
    device: torch.device,  # DEVICE
) -> np.ndarray:  # RETURNS (grid_n,grid_n) ESCAPE ITERS
    P = int(C.shape[0])  # NUM POINTS
    r2 = float(escape_r) * float(escape_r)  # R^2

    # INIT STATE x0 = [Re(z),Im(z),Re(c),Im(c)]  # AUGMENTED STATE
    X = np.zeros((P, 4), dtype=np.float32)  # INIT X
    X[:, 2:4] = C  # SET C

    # NORMALISE INPUT LIKE TRAINING  # MATCH TRAINING
    Xn = _norm_forward(X, mn, mx)  # NORMALISE

    alive = np.ones((P,), dtype=bool)  # NOT ESCAPED
    iters = np.zeros((P,), dtype=np.int32)  # ESCAPE ITER

    X_t = to_tensor(Xn, device)  # TO TENSOR
    C_t = to_tensor(_norm_forward(np.concatenate([np.zeros((P, 2), np.float32), C], axis=1), mn, mx)[:, 2:4], device)  # CONST C (NORMALISED)

    for k in range(1, max_iters + 1):  # ITER LOOP
        if not alive.any():  # DONE
            break  # EXIT

        idx = np.where(alive)[0]  # ACTIVE IDX
        xk = X_t[idx]  # ACTIVE STATES

        zk = encoder(xk)  # ENCODE
        zk1 = dmd.predict(zk, steps=1)[-1]  # RETURN LAST STEP: (B,D) WITH B=BATCH SAMPLES, D=LATENT FEATURES
        xk1 = decoder(zk1)  # DECODE

        # FORCE C TO STAY CONSTANT  # KEEP PARAM FIXED
        xk1[:, 2:4] = C_t[idx]  # FIX C

        X_t[idx] = xk1  # WRITE BACK

        # CHECK ESCAPE USING UNNORMALISED Z  # ESCAPE TEST
        x_cpu = xk1.detach().cpu().numpy()  # TO NUMPY
        x_den = _norm_inverse(x_cpu, mn, mx)  # DENORM
        zr = x_den[:, 0]  # RE Z
        zi = x_den[:, 1]  # IM Z
        esc = (zr * zr + zi * zi) > r2  # ESCAPE MASK

        escaped_idx = idx[esc]  # GLOBAL IDX
        iters[escaped_idx] = k  # SAVE ITER
        alive[escaped_idx] = False  # MARK DEAD

    # STABLE POINTS GET max_iters  # INSIDE SET
    iters[alive] = max_iters  # MARK STABLE

    return iters.reshape(grid_n, grid_n)  # RESHAPE IMAGE

# =============================== IMAGE SAVE ==================================
def save_escape_image(  # SAVE PNG
    escape_iters: np.ndarray,  # (H,W)
    *,
    max_iters: int,  # MAX
    out_png: str | Path,  # OUTPUT
) -> str:  # RETURNS PATH
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # ENSURE DIR

    # MAP: STABLE (max) -> BLACK, FAST ESCAPE -> WHITE  # SIMPLE MAP
    esc = escape_iters.astype(np.float32)  # FLOAT
    norm = esc / float(max_iters)  # 0..1
    img = (255.0 * norm).clip(0, 255).astype(np.uint8)  # TO 8BIT
    img = 255 - img  # INVERT (INSIDE BLACK)
    im = Image.fromarray(img, mode="L")  # GRAYSCALE
    im.save(out_png)  # SAVE
    return str(out_png)  # RETURN

# =============================== OPTIONAL PLOT ===============================
def save_and_show_plot(  # OPTIONAL FIGURE
    escape_iters: np.ndarray,  # DATA
    *,
    out_png: str | Path,  # OUTPUT
    show: bool,  # SHOW FLAG
) -> str:  # RETURNS PATH
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # ENSURE DIR

    plt.figure()  # NEW FIG
    plt.imshow(escape_iters, origin="lower")  # SHOW
    plt.axis("off")  # NO AXIS
    plt.tight_layout()  # TIGHT
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0)  # SAVE
    if show:  # SHOW
        plt.show()  # DISPLAY
    else:  # NO SHOW
        plt.close()  # CLOSE
    return str(out_png)  # RETURN
