# =========================== PREPARE TRAINING DATA ===========================
from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================ IMPORTS ====================================
from dataclasses import dataclass  # DATACLASS SUPPORT  # (PREIMPLEMENTED CLASS OK)
from pathlib import Path  # PATH UTILITIES
import numpy as np  # NUMPY

from typing import Optional  # OPTIONAL TYPE

from data_loader import (  # LOCAL DATA UTILS
    load_task_npz_pair,  # LOAD TASK/REST NPZ
    generate_state_trajectories,  # TRAJ GEN
    load_one_A_matrix,  # LOAD ONE A
)  # END IMPORTS




# ============================== DATA STRUCTURES ==============================
@dataclass  # DATACLASS  # (PREIMPLEMENTED CLASS OK)
class TrainingData:  # DATA CONTAINER
    X: np.ndarray   # (N,D)  # FULL
    X1: np.ndarray  # (N,D)  # LEFT
    X2: np.ndarray  # (N,D)  # RIGHT
    meta: dict  # META
    X_grid: Optional[np.ndarray] = None  # (T,Ni,Nr,D)  # BIG ARRAY YOU ASKED FOR

# =============================== HELPERS =====================================
def _sanitize_finite(x: np.ndarray, name: str) -> np.ndarray:  # FINITE FIX
    x = np.asarray(x, dtype=np.float32)  # FP32
    if np.isfinite(x).all():  # OK
        return x  # RETURN
    print(f"[WARN] {name} HAD NaN/Inf -> FIXING")  # WARN
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)  # FIX

def _cap_complex_vec(zr: np.ndarray, zi: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:  # CLAMP |Z|
    r = float(r)  # R
    if r <= 0.0:  # BAD
        r = 2.0  # DEFAULT
    r2 = r * r  # R^2

    mag2 = zr * zr + zi * zi  # |Z|^2
    over = mag2 > r2  # OVER?
    if np.any(over):  # APPLY
        mag = np.sqrt(np.maximum(mag2, 1e-30)).astype(np.float32, copy=False)  # |Z|
        s = (r / mag).astype(np.float32, copy=False)  # SCALE
        zr = np.where(over, zr * s, zr).astype(np.float32, copy=False)  # RE
        zi = np.where(over, zi * s, zi).astype(np.float32, copy=False)  # IM
    return zr, zi  # OUT

# ========================= BUILD DATA FROM A-SEQUENCES =======================
def build_from_A_sequences(  # BUILD FROM MATLAB A
    data_dir: str | Path,  # FOLDER
    *,  # KWONLY
    n_traj: int = 500,  # COUNT
    x0_scale: float = 1.0,  # SCALE
    noise_std: float = 0.0,  # NOISE
    flatten: bool = False,  # FLATTEN
    escape_r: float = 2.0,  # CLAMP R
) -> TrainingData:
    data_dir = Path(data_dir)  # PATH

    A_emotion, A_rest = load_task_npz_pair(data_dir, key=None, flatten=flatten)  # LOAD
    A_seq = np.concatenate([A_emotion, A_rest], axis=0)  # MERGE

    X, X1, X2 = generate_state_trajectories(  # GEN
        A_seq=A_seq,  # A SEQ
        n_traj=n_traj,  # COUNT
        x0_mode="zeros",  # Z0=0
        c_scale=0.03,  # RANDOM C
        use_square=True,  # APPLY SQUARE
        x0_scale=x0_scale,  # KEEP
        noise_std=noise_std,  # KEEP
        escape_r=escape_r,  # CLAMP
    )  # END

    X = _sanitize_finite(X, "X")  # FIX
    X1 = _sanitize_finite(X1, "X1")  # FIX
    X2 = _sanitize_finite(X2, "X2")  # FIX

    return TrainingData(  # PACK
        X=X.astype(np.float32),  # STORE
        X1=X1.astype(np.float32),  # STORE
        X2=X2.astype(np.float32),  # STORE
        meta={  # META
            "mode": "A_sequences",  # TAG
            "data_dir": str(data_dir),  # SRC
            "n_traj": n_traj,  # SAVE
            "x0_scale": x0_scale,  # SAVE
            "noise_std": noise_std,  # SAVE
            "escape_r": float(escape_r),  # SAVE
        },
    )

# ===================== MANDELBROT TRAINING DATA (GRID, FIXED SIZE) =====================
def build_mandelbrot_training_data(  # BUILD MANDELBROT (GRID)
    *,
    c_re_min: float,  # RE MIN
    c_re_max: float,  # RE MAX
    c_im_min: float,  # IM MIN
    c_im_max: float,  # IM MAX
    c_re_n: int = 256,  # RESOLUTION ON REAL AXIS
    c_im_n: int = 256,  # RESOLUTION ON IMAG AXIS
    max_iters: int = 40,  # ITERS (T)
    escape_r: float = 2.0,  # ESCAPE R (UPPER BOUND)
    seed: int = 0,  # KEEP PARAM (NOT USED IN GRID MODE)  # COMPAT
) -> TrainingData:
    """
    YOU ASKED FOR:
      - IF |Z| > ESCAPE_R => SET |Z| = ESCAPE_R (CLAMP UPPER BOUND)
      - REMOVE NORMALISATION
      - OUTPUT WHAT IS IN TRAINING
      - X SAVED AS ONE BIG ARRAY:
          T * (C_IM_RES) * (C_RE_RES)

    DATA FORMAT PER ROW:
      [zr, zi, cr, ci]

    SHAPES:
      P = c_re_n * c_im_n
      X  : (T*P, 4)
      X1 : ((T-1)*P, 4)
      X2 : ((T-1)*P, 4)
    """
    _ = int(seed)  # UNUSED  # KEEP FOR CALL COMPAT

    # ------------------------------ SETUP -----------------------------------
    T = int(max_iters)  # ITERS
    Nr = int(c_re_n)  # RE RES
    Ni = int(c_im_n)  # IM RES
    r = float(escape_r)  # R
    if r <= 0.0:  # BAD
        r = 2.0  # DEFAULT

    # ------------------------------ C GRID ----------------------------------
    cr = np.linspace(float(c_re_min), float(c_re_max), Nr, dtype=np.float32)  # RE LIN
    ci = np.linspace(float(c_im_min), float(c_im_max), Ni, dtype=np.float32)  # IM LIN
    C_re, C_im = np.meshgrid(cr, ci, indexing="xy")  # (Ni, Nr)
    P = int(C_re.size)  # POINTS

    cr_flat = C_re.reshape(-1).astype(np.float32, copy=False)  # (P,)
    ci_flat = C_im.reshape(-1).astype(np.float32, copy=False)  # (P,)

    # ------------------------------ ITERATE ---------------------------------
    zr = np.zeros((P,), dtype=np.float32)  # ZR
    zi = np.zeros((P,), dtype=np.float32)  # ZI

    X_tp4 = np.zeros((T, P, 4), dtype=np.float32)  # STORE (T,P,4)

    for t in range(T):  # LOOP T
        # z <- z^2 + c  (VECTORISED)
        zr2 = zr * zr  # zr^2
        zi2 = zi * zi  # zi^2
        zri = 2.0 * zr * zi  # 2*zr*zi

        zr = (zr2 - zi2) + cr_flat  # RE UPDATE
        zi = (zri) + ci_flat  # IM UPDATE

        # FIX NaN/Inf
        bad = (~np.isfinite(zr)) | (~np.isfinite(zi))  # BAD
        if np.any(bad):  # RESET BAD
            zr = np.where(bad, 0.0, zr).astype(np.float32, copy=False)  # FIX
            zi = np.where(bad, 0.0, zi).astype(np.float32, copy=False)  # FIX

        # CLAMP EVERY ITERATION (NO BREAK)
        zr, zi = _cap_complex_vec(zr, zi, r)  # UPPER BOUND

        # WRITE BLOCK
        X_tp4[t, :, 0] = zr  # ZR
        X_tp4[t, :, 1] = zi  # ZI
        X_tp4[t, :, 2] = cr_flat  # CR
        X_tp4[t, :, 3] = ci_flat  # CI
    # ------------------------------ BIG GRID ---------------------------------
    # SAVE WHAT IS IN TRAINING (UNFLATTENED)
    # SHAPE: (T, Ni, Nr, 4)  # [zr, zi, cr, ci]
    X_grid = X_tp4.reshape(T, Ni, Nr, 4).astype(np.float32, copy=False)  # BIG

    # ------------------------------ FLATTEN ---------------------------------
    X = X_tp4.reshape(T * P, 4).astype(np.float32)  # (T*P,4)
    #flat everyting si problem
    X1 = X_tp4[:-1].reshape((T - 1) * P, 4).astype(np.float32)  # ((T-1)*P,4)
    X2 = X_tp4[1:].reshape((T - 1) * P, 4).astype(np.float32)  # ((T-1)*P,4)

    X = _sanitize_finite(X, "X")  # FIX
    X1 = _sanitize_finite(X1, "X1")  # FIX
    X2 = _sanitize_finite(X2, "X2")  # FIX

    # ------------------------------ OUTPUT -----------------------------------
    print("[TRAINING] X shape     :", X.shape, "dtype:", X.dtype)  # LOG
    print("[TRAINING] X1 shape    :", X1.shape)  # LOG
    print("[TRAINING] X2 shape    :", X2.shape)  # LOG
    print("[TRAINING] X_grid shape:", X_grid.shape, "dtype:", X_grid.dtype)  # LOG
    print("[TRAINING] zr range:", float(X[:, 0].min()), "to", float(X[:, 0].max()))  # LOG
    print("[TRAINING] zi range:", float(X[:, 1].min()), "to", float(X[:, 1].max()))  # LOG
    print("[TRAINING] cr range:", float(X[:, 2].min()), "to", float(X[:, 2].max()))  # LOG
    print("[TRAINING] ci range:", float(X[:, 3].min()), "to", float(X[:, 3].max()))  # LOG
    print("[TRAINING] head row 0:", X[0].tolist())  # LOG

    return TrainingData(  # PACK
        X=X,  # STORE
        X1=X1,  # STORE
        X2=X2,  # STORE
        meta={  # META
            "mode": "mandelbrot_grid",  # TAG
            "bbox": (float(c_re_min), float(c_re_max), float(c_im_min), float(c_im_max)),  # BOX
            "c_re_n": int(Nr),  # SAVE
            "c_im_n": int(Ni),  # SAVE
            "P": int(P),  # SAVE
            "max_iters": int(T),  # SAVE
            "escape_r": float(r),  # SAVE
            "X_size_formula": "max_iters * c_im_n * c_re_n",  # YOU ASKED
        },
        X_grid=X_grid,  # SAVE BIG ARRAY
    )

# ================================ SAVE NPZ ===================================
def save_training_npz(out_path: str | Path, td: TrainingData) -> str:  # SAVE
    out_path = Path(out_path)  # PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)  # MKDIR
    if td.X_grid is None:  # NO GRID
        np.savez(
            out_path,
            X=td.X,
            X1=td.X1,
            X2=td.X2,
            meta=np.array([td.meta], dtype=object),
        )
    else:  # WITH GRID
        np.savez(
            out_path,
            X=td.X,
            X1=td.X1,
            X2=td.X2,
            X_grid=td.X_grid,  # BIG ARRAY
            meta=np.array([td.meta], dtype=object),
        )
    return str(out_path)  # RETURN


def build_matrix_c_grid_training_data(  # BUILD GRID DATA WITH A
    data_dir: str | Path,  # DATA FOLDER
    *,  # KWONLY
    source: str = "emotion",  # WHICH FILE
    index: int = 0,  # WHICH MATRIX
    c_re_min: float,  # RE MIN
    c_re_max: float,  # RE MAX
    c_im_min: float,  # IM MIN
    c_im_max: float,  # IM MAX
    c_re_n: int = 256,  # RE RES
    c_im_n: int = 256,  # IM RES
    max_iters: int = 40,  # ITER COUNT
    escape_r: float = 2.0,  # ESCAPE RADIUS
) -> TrainingData:  # RETURN DATA
    data_dir = Path(data_dir)  # PATH
    A = load_one_A_matrix(data_dir, source=source, index=index)  # LOAD A
    A = np.asarray(A, dtype=np.float32)  # FP32

    d = int(A.shape[0])  # STATE DIM
    T = int(max_iters)  # ITERS
    Nr = int(c_re_n)  # RE RES
    Ni = int(c_im_n)  # IM RES
    r = float(escape_r)  # R
    r2 = r * r  # R2

    cr = np.linspace(float(c_re_min), float(c_re_max), Nr, dtype=np.float32)  # RE GRID
    ci = np.linspace(float(c_im_min), float(c_im_max), Ni, dtype=np.float32)  # IM GRID
    C_re, C_im = np.meshgrid(cr, ci, indexing="xy")  # GRID
    P = int(C_re.size)  # POINT COUNT

    cr_flat = C_re.reshape(-1).astype(np.float32, copy=False)  # FLAT RE
    ci_flat = C_im.reshape(-1).astype(np.float32, copy=False)  # FLAT IM
    c_flat = (cr_flat + 1j * ci_flat).astype(np.complex64)  # COMPLEX C

    z = np.zeros((P, d), dtype=np.complex64)  # Z0
    z[:, 0] = c_flat  # SEED FIRST COMP

    feat_dim = 2 * d + 2  # [RE_d, IM_d, CR, CI]
    X_tp = np.zeros((T, P, feat_dim), dtype=np.float32)  # STORE ALL

    for t in range(T):  # TIME LOOP
        Az = (z @ A.T).astype(np.complex64)  # APPLY A
        z = (Az * Az).astype(np.complex64)  # NONLINEAR STEP
        z[:, 0] = (z[:, 0] + c_flat).astype(np.complex64)  # ADD C FIRST ONLY

        mag2 = (z.real * z.real + z.imag * z.imag).astype(np.float32)  # |Z|^2 PER COMP
        bad = (~np.isfinite(mag2)) | (mag2 > r2)  # BAD MASK
        if np.any(bad):  # CLAMP
            mag = np.sqrt(np.maximum(mag2, 1e-30)).astype(np.float32)  # |Z|
            scale = (r / mag).astype(np.float32)  # SCALE
            z_real = np.where(bad, z.real * scale, z.real).astype(np.float32)  # CLAMP RE
            z_imag = np.where(bad, z.imag * scale, z.imag).astype(np.float32)  # CLAMP IM
            z = (z_real + 1j * z_imag).astype(np.complex64)  # WRITE BACK

        X_tp[t, :, 0:d] = z.real.astype(np.float32)  # SAVE RE
        X_tp[t, :, d:2 * d] = z.imag.astype(np.float32)  # SAVE IM
        X_tp[t, :, 2 * d] = cr_flat  # SAVE CR
        X_tp[t, :, 2 * d + 1] = ci_flat  # SAVE CI

    X_grid = X_tp.reshape(T, Ni, Nr, feat_dim).astype(np.float32, copy=False)  # GRID VIEW
    X = X_tp.reshape(T * P, feat_dim).astype(np.float32)  # FLAT X
    X1 = X_tp[:-1].reshape((T - 1) * P, feat_dim).astype(np.float32)  # PAIRS L
    X2 = X_tp[1:].reshape((T - 1) * P, feat_dim).astype(np.float32)  # PAIRS R

    X = _sanitize_finite(X, "X")  # FIX
    X1 = _sanitize_finite(X1, "X1")  # FIX
    X2 = _sanitize_finite(X2, "X2")  # FIX

    return TrainingData(  # PACK
        X=X,  # STORE
        X1=X1,  # STORE
        X2=X2,  # STORE
        meta={  # META
            "mode": "matrix_c_grid",  # TAG
            "source": str(source),  # WHICH FILE
            "index": int(index),  # WHICH MATRIX
            "state_dim": int(d),  # DIM
            "bbox": (float(c_re_min), float(c_re_max), float(c_im_min), float(c_im_max)),  # BOX
            "c_re_n": int(Nr),  # SAVE
            "c_im_n": int(Ni),  # SAVE
            "P": int(P),  # SAVE
            "max_iters": int(T),  # SAVE
            "escape_r": float(r),  # SAVE
        },
        X_grid=X_grid,  # STORE GRID
    )