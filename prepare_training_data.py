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
    load_all_A_matrices,  # LOAD ALL A
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

def determine_escape_radius(A: np.ndarray) -> float:  # MATRIX-DEPENDENT ESCAPE RADIUS (fikl's commonlib.py)
    
    A = np.asarray(A, dtype=np.float64)  # FP64 FOR THE SVD
    if A.ndim != 2 or A.shape[0] != A.shape[1]:  # SQUARE CHECK
        raise ValueError(f"'A' must be a square matrix: {A.shape}")  # ERROR
    n = int(A.shape[0])  # SIZE
    sigma = np.linalg.svd(A, compute_uv=False)  # SINGULAR VALUES
    sigma_min = float(np.min(sigma))  # SMALLEST
    if sigma_min <= 0.0:  # SINGULAR MATRIX GUARD
        return float("inf")  # NO FINITE BOUND MAKES SENSE
    return float(2.0 * np.sqrt(n) / (sigma_min ** 2))  # fikl's FORMULA


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


def _build_matrix_c_grid_training_data_from_A(  # BUILD GRID DATA WITH GIVEN A
    A: np.ndarray,  # ONE MATRIX
    *,  # KWONLY
    c_re_min: float,  # RE MIN
    c_re_max: float,  # RE MAX
    c_im_min: float,  # IM MIN
    c_im_max: float,  # IM MAX
    c_re_n: int = 256,  # RE RES
    c_im_n: int = 256,  # IM RES
    max_iters: int = 40,  # ITER COUNT
    escape_r: float = 2.0,  # NUMERICAL CLAMP DURING ITERATION (E.G. D.DYNAMICS_CLAMP_R)
    matrix_index: int | None = None,  # OPTIONAL INDEX
    matrix_source: str | None = None,  # OPTIONAL SOURCE
    filter_escaped: bool = True,  # DROP (X1,X2) PAIRS FOR TRAJECTORIES THAT ESCAPE BY THE FINAL ITERATION
    classify_r: float | None = None,  # "ESCAPED" THRESHOLD FOR FILTERING (E.G. D.ESCAPE_R); DEFAULTS TO escape_r
    keep_escaped_fraction: float = 0.0,  # EXPERIMENTAL: RANDOMLY KEEP THIS FRACTION OF ESCAPED TRAJECTORIES TOO
    keep_escaped_seed: int = 0,  # REPRODUCIBLE SUBSAMPLE
) -> TrainingData:

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
    #z[:] = c_flat[:, None]  # SEED ALL COMPONENTS

    feat_dim = 2 * d + 2  # [RE_d, IM_d, CR, CI]
    X_tp = np.zeros((T, P, feat_dim), dtype=np.float32)  # STORE ALL

    for t in range(T):  # TIME LOOP
        Az = (z @ A.T).astype(np.complex64)  # APPLY A
        z = (Az * Az).astype(np.complex64)  # NONLINEAR STEP
        z = (z + c_flat[:, None]).astype(np.complex64)  # ADD C TO ALL COMPONENTS

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

    X_grid = X_tp.reshape(T, Ni, Nr, feat_dim).astype(np.float32, copy=False)  # BIG GRID -- ALWAYS FULL GRID
    X = X_tp.reshape(T * P, feat_dim).astype(np.float32)  # FLAT -- ALWAYS FULL GRID

    # ---------------------- ALIVE MASK (FOR FILTERING ONLY) -----------------
    c_r = float(escape_r if classify_r is None else classify_r)  # THRESHOLD FOR "ESCAPED"
    final_zr = X_tp[-1, :, 0:d]  # LAST STORED RE
    final_zi = X_tp[-1, :, d:2 * d]  # LAST STORED IM
    final_mag2 = np.max(final_zr * final_zr + final_zi * final_zi, axis=1)  # MAX COMPONENT MAG2 PER GRID POINT
    alive = np.isfinite(final_mag2) & (final_mag2 < c_r * c_r)  # BOUNDED AT FINAL ITER -> KEEP

    if bool(filter_escaped) and int(np.count_nonzero(alive)) > 0:  # HAVE SOMETHING TO KEEP
        keep = alive.copy()  # START FROM THE ALIVE MASK

        frac = float(keep_escaped_fraction)  # EXPERIMENTAL KNOB, DEFAULT 0.0 = OFF
        if frac > 0.0:  # OPT-IN: ALSO KEEP A RANDOM SLICE OF THE ESCAPED TRAJECTORIES
            escaped_idx = np.flatnonzero(~alive)  # INDICES OF ESCAPED GRID POINTS
            n_keep = int(round(frac * escaped_idx.size))  # HOW MANY TO ADD BACK
            if n_keep > 0:  # SOMETHING TO ADD
                rng = np.random.default_rng(int(keep_escaped_seed))  # REPRODUCIBLE
                chosen = rng.choice(escaped_idx, size=min(n_keep, escaped_idx.size), replace=False)  # SUBSAMPLE
                keep[chosen] = True  # ADD THEM BACK IN

        X1 = X_tp[:-1][:, keep, :].reshape(-1, feat_dim).astype(np.float32)  # LEFT
        X2 = X_tp[1:][:, keep, :].reshape(-1, feat_dim).astype(np.float32)  # RIGHT
    else:  # NO FILTERING (OR NOTHING SURVIVED -- FALL BACK RATHER THAN RETURN EMPTY ARRAYS)
        if bool(filter_escaped):  # WARN, SINCE THIS IS PROBABLY NOT WHAT YOU WANT
            print(f"[WARN] filter_escaped=True BUT 0/{P} POINTS SURVIVED AT classify_r={c_r} "
                  f"-- FALLING BACK TO THE UNFILTERED GRID FOR THIS MATRIX")  # WARN
        X1 = X_tp[:-1].reshape((T - 1) * P, feat_dim).astype(np.float32)  # LEFT, UNFILTERED
        X2 = X_tp[1:].reshape((T - 1) * P, feat_dim).astype(np.float32)  # RIGHT, UNFILTERED

    X = _sanitize_finite(X, "X")  # FIX
    X1 = _sanitize_finite(X1, "X1")  # FIX
    X2 = _sanitize_finite(X2, "X2")  # FIX

    n_alive = int(np.count_nonzero(alive))  # HOW MANY GRID POINTS WERE BOUNDED
    n_kept = int(X1.shape[0] // max(T - 1, 1))  # HOW MANY GRID POINTS ACTUALLY WENT INTO X1/X2
    extra = f", +{n_kept - n_alive} ESCAPED (keep_escaped_fraction={keep_escaped_fraction:g})" if n_kept > n_alive else ""
    print(f"[TRAINING] matrix_c_grid: {n_alive}/{P} points bounded at classify_r={c_r:g} "
          f"(clamp_r={r:g}) -> X1/X2 rows={X1.shape[0]}{extra}"
          + (" (UNFILTERED)" if not filter_escaped else ""))  # LOG

    return TrainingData(
        X=X,  # FULL GRID, ALWAYS
        X1=X1,  # LEFT, POSSIBLY FILTERED
        X2=X2,  # RIGHT, POSSIBLY FILTERED
        meta={
            "mode": "matrix_c_grid",  # TAG
            "matrix_index": None if matrix_index is None else int(matrix_index),  # SAVE
            "matrix_source": matrix_source,  # SAVE
            "state_dim": int(d),  # SAVE
            "max_iters": int(T),  # SAVE
            "c_re_n": int(Nr),  # SAVE
            "c_im_n": int(Ni),  # SAVE
            "escape_r": float(r),  # SAVE (THIS IS THE ITERATION CLAMP, E.G. DYNAMICS_CLAMP_R)
            "classify_r": float(c_r),  # SAVE (THE "ESCAPED" THRESHOLD USED FOR FILTERING)
            "filter_escaped": bool(filter_escaped),  # SAVE
            "keep_escaped_fraction": float(keep_escaped_fraction),  # SAVE
            "n_alive": n_alive,  # SAVE
            "n_kept": n_kept,  # SAVE (>= n_alive IF keep_escaped_fraction > 0)
            "n_total": int(P),  # SAVE
            "alive_mask_grid": alive.reshape(Ni, Nr),  # (H,W) BOOL -- WHICH PIXELS WERE BOUNDED (NOT NECESSARILY = TRAINED-ON WHEN keep_escaped_fraction > 0)
        },
        X_grid=X_grid,  # GRID -- ALWAYS FULL, UNFILTERED (NEEDED FOR IMAGES / ROLLOUT-VS-GROUND-TRUTH)
    )

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
    escape_r: float = 2.0,  # NUMERICAL CLAMP DURING ITERATION (E.G. D.DYNAMICS_CLAMP_R)
    filter_escaped: bool = True,  # DROP (X1,X2) PAIRS FOR TRAJECTORIES THAT ESCAPE BY THE FINAL ITERATION
    classify_r: float | None = None,  # "ESCAPED" THRESHOLD (E.G. D.ESCAPE_R); DEFAULTS TO escape_r
    keep_escaped_fraction: float = 0.0,  # EXPERIMENTAL, OFF BY DEFAULT -- SEE _build_matrix_c_grid_training_data_from_A
) -> TrainingData:  # RETURN DATA
    data_dir = Path(data_dir)  # PATH
    A = load_one_A_matrix(data_dir, source=source, index=index)  # LOAD A
    return _build_matrix_c_grid_training_data_from_A(  # BUILD
        A,
        c_re_min=c_re_min,
        c_re_max=c_re_max,
        c_im_min=c_im_min,
        c_im_max=c_im_max,
        c_re_n=c_re_n,
        c_im_n=c_im_n,
        max_iters=max_iters,
        escape_r=escape_r,
        matrix_index=index,
        matrix_source=source,
        filter_escaped=filter_escaped,
        classify_r=classify_r,
        keep_escaped_fraction=keep_escaped_fraction,
    )

def build_matrix_c_grid_training_data_many_matrices(  # MANY MATRICES
    data_dir: str | Path,  # DATA FOLDER
    *,  # KWONLY
    source: str = "emotion",  # WHICH FILE
    indices: list[int] | np.ndarray,  # WHICH MATRICES
    c_re_min: float,  # RE MIN
    c_re_max: float,  # RE MAX
    c_im_min: float,  # IM MIN
    c_im_max: float,  # IM MAX
    c_re_n: int = 8,  # SMALL RE RES
    c_im_n: int = 8,  # SMALL IM RES
    max_iters: int = 40,  # ITER COUNT
    escape_r: float = 2.0,  # NUMERICAL CLAMP DURING ITERATION (E.G. D.DYNAMICS_CLAMP_R)
    filter_escaped: bool = True,  # DROP (X1,X2) PAIRS FOR TRAJECTORIES THAT ESCAPE BY THE FINAL ITERATION
    classify_r: float | None = None,  # "ESCAPED" THRESHOLD (E.G. D.ESCAPE_R); DEFAULTS TO escape_r
) -> list[TrainingData]:
    data_dir = Path(data_dir)  # PATH
    A_all = load_all_A_matrices(data_dir, source=source)  # LOAD ALL
    out: list[TrainingData] = []  # STORE

    for idx in np.asarray(indices, dtype=np.int64):  # LOOP MATRICES
        td = _build_matrix_c_grid_training_data_from_A(  # BUILD ONE
            A_all[int(idx)],
            c_re_min=c_re_min,
            c_re_max=c_re_max,
            c_im_min=c_im_min,
            c_im_max=c_im_max,
            c_re_n=c_re_n,
            c_im_n=c_im_n,
            max_iters=max_iters,
            escape_r=escape_r,
            matrix_index=int(idx),
            matrix_source=source,
            filter_escaped=filter_escaped,
            classify_r=classify_r,
        )
        out.append(td)  # STORE

    return out  # RETURN LIST
