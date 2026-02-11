# =========================== PREPARE TRAINING DATA ===========================
from __future__ import annotations  # ENABLE PY<3.11 REDDIT SAID IT IS USEFUL

# ================================ IMPORTS ====================================
from dataclasses import dataclass  # DATACLASS SUPPORT
from pathlib import Path  # PATH UTILITIES
import numpy as np  # NUMERICAL ARRAYS

from data_loader import (  # LOCAL DATA UTILITIES
    load_task_npz_pair,  # LOAD TASK/REST NPZ
    normalize_data,  # NORMALISATION HELPER
    generate_state_trajectories,  # TRAJECTORY GENERATOR
)  # END IMPORT LIST

# ============================== DATA STRUCTURES ==============================
@dataclass  # DATACLASS DECORATOR
class TrainingData:  # TRAINING DATA CONTAINER
    X: np.ndarray   # (N, D)  # FULL SNAPSHOTS
    X1: np.ndarray  # (N1, D)  # LEFT PAIRS
    X2: np.ndarray  # (N1, D)  # RIGHT PAIRS
    meta: dict  # METADATA DICT

# =============================== HELPERS =====================================
def _sanitize_finite(x: np.ndarray, name: str) -> np.ndarray:  # SANITISE ARRAY
    x = np.asarray(x, dtype=np.float32)  # FORCE FLOAT32
    if np.isfinite(x).all():  # CHECK FINITE
        return x  # RETURN CLEAN
    print(f"[WARN] {name} had NaN/Inf -> sanitising")  # WARN USER
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)  # REPLACE BAD VALUES

# ========================= BUILD DATA FROM A-SEQUENCES =======================
def build_from_A_sequences(  # BUILD FROM MATLAB NPZ A
    data_dir: str | Path,  # DATASET FOLDER
    *,  # KWONLY ARGS
    n_traj: int = 500,  # NUM TRAJECTORIES
    x0_scale: float = 1.0,  # INITIAL SCALE
    noise_std: float = 0.0,  # NOISE LEVEL
    flatten: bool = False,  # FLATTEN MATRICES
) -> TrainingData:  # RETURNS TRAININGDATA
    data_dir = Path(data_dir)  # COERCE TO PATH

    A_emotion, A_rest = load_task_npz_pair(data_dir, key=None, flatten=flatten)  # LOAD TWO SETS
    A_seq = np.concatenate([A_emotion, A_rest], axis=0)  # MERGE SEQUENCES

    X, X1, X2 = generate_state_trajectories(  # GENERATE TRAJECTORIES
        A_seq=A_seq,  # PASS SEQUENCE
        n_traj=n_traj,  # PASS COUNT
        x0_scale=x0_scale,  # PASS SCALE
        noise_std=noise_std,  # PASS NOISE
    )  # END CALL

    X = _sanitize_finite(X, "X")  # CLEAN X
    X1 = _sanitize_finite(X1, "X1")  # CLEAN X1
    X2 = _sanitize_finite(X2, "X2")  # CLEAN X2

    # IMPORTANT: scale all with SAME min/max (your current approach in older main.py) :contentReference[oaicite:5]{index=5}  # KEEP SAME SCALE
    mn, mx = float(X.min()), float(X.max())  # GET RANGE
    X  = (X  - mn) / (mx - mn + 1e-12)  # SCALE X
    X1 = (X1 - mn) / (mx - mn + 1e-12)  # SCALE X1
    X2 = (X2 - mn) / (mx - mn + 1e-12)  # SCALE X2

    return TrainingData(  # BUILD OUTPUT
        X=X.astype(np.float32),  # STORE X
        X1=X1.astype(np.float32),  # STORE X1
        X2=X2.astype(np.float32),  # STORE X2
        meta={  # STORE META
            "mode": "A_sequences",  # MODE TAG
            "data_dir": str(data_dir),  # SOURCE PATH
            "n_traj": n_traj,  # SAVE COUNT
            "x0_scale": x0_scale,  # SAVE SCALE
            "noise_std": noise_std,  # SAVE NOISE
            "mn": mn,  # SAVE MIN
            "mx": mx,  # SAVE MAX
        },  # END META
    )  # END RETURN

# ====================== BUILD DATA FROM C-GRID SCALAR MAP =====================
def build_from_c_grid_quadratic_scalar(  # BUILD FROM C GRID
    *,  # KWONLY ARGS
    x_min: float,  # RE MIN
    x_max: float,  # RE MAX
    y_min: float,  # IM MIN
    y_max: float,  # IM MAX
    grid_n: int = 256,  # GRID SIZE
    max_iters: int = 80,  # MAX STEPS
    escape_r: float = 2.0,  # ESCAPE RADIUS
) -> TrainingData:  # RETURNS TRAININGDATA
    # matches the scalar recurrence you currently run in DATA_MODE=1 :contentReference[oaicite:6]{index=6}  # MATCH OLD MODE
    xs = np.linspace(x_min, x_max, grid_n, dtype=np.float32)  # MAKE X GRID
    ys = np.linspace(y_min, y_max, grid_n, dtype=np.float32)  # MAKE Y GRID
    C = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)  # BUILD C LIST

    X_list, X1_list, X2_list = [], [], []  # INIT BUFFERS
    r2 = float(escape_r) * float(escape_r)  # PRECOMPUTE R^2

    for c in C:  # LOOP C VALUES
        z = np.zeros(2, dtype=np.float32)  # INIT Z
        orbit = []  # INIT ORBIT

        for _ in range(max_iters):  # ITERATE
            zr, zi = float(z[0]), float(z[1])  # SPLIT Z
            cr, ci = float(c[0]), float(c[1])  # SPLIT C
            z = np.array([zr * zr - zi * zi + cr, 2.0 * zr * zi + ci], dtype=np.float32)  # QUADRATIC STEP
            orbit.append(z.copy())  # APPEND STATE
            if (z[0] * z[0] + z[1] * z[1]) > r2:  # ESCAPE TEST
                break  # STOP EARLY

        if len(orbit) >= 2:  # NEED PAIRS
            o = np.stack(orbit, axis=0)     # (T,2)  # STACK ORBIT
            X_list.append(o)  # ADD X
            X1_list.append(o[:-1])  # ADD X1
            X2_list.append(o[1:])  # ADD X2

    X  = np.concatenate(X_list, axis=0)  # FLATTEN X
    X1 = np.concatenate(X1_list, axis=0)  # FLATTEN X1
    X2 = np.concatenate(X2_list, axis=0)  # FLATTEN X2

    X  = _sanitize_finite(X,  "X")  # CLEAN X
    X1 = _sanitize_finite(X1, "X1")  # CLEAN X1
    X2 = _sanitize_finite(X2, "X2")  # CLEAN X2

    # normalise each with common scaling  # COMMON SCALE NOTE
    mn, mx = float(X.min()), float(X.max())  # GET RANGE
    X  = (X  - mn) / (mx - mn + 1e-12)  # SCALE X
    X1 = (X1 - mn) / (mx - mn + 1e-12)  # SCALE X1
    X2 = (X2 - mn) / (mx - mn + 1e-12)  # SCALE X2

    return TrainingData(  # BUILD OUTPUT
        X=X.astype(np.float32),  # STORE X
        X1=X1.astype(np.float32),  # STORE X1
        X2=X2.astype(np.float32),  # STORE X2
        meta={  # STORE META
            "mode": "c_grid_scalar_quadratic",  # MODE TAG
            "bbox": (x_min, x_max, y_min, y_max),  # SAVE BOX
            "grid_n": grid_n,  # SAVE GRID
            "max_iters": max_iters,  # SAVE ITERS
            "escape_r": escape_r,  # SAVE R
            "mn": mn,  # SAVE MIN
            "mx": mx,  # SAVE MAX
        },  # END META
    )  # END RETURN

# ================================ SAVE NPZ ===================================
def save_training_npz(out_path: str | Path, td: TrainingData) -> str:  # SAVE DATASET
    out_path = Path(out_path)  # COERCE PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ENSURE DIR
    np.savez(  # WRITE NPZ
        out_path,  # OUTPUT PATH
        X=td.X,  # SAVE X
        X1=td.X1,  # SAVE X1
        X2=td.X2,  # SAVE X2
        meta=np.array([td.meta], dtype=object),  # SAVE META
    )  # END SAVE
    return str(out_path)  # RETURN PATH

# ===================== MANDELBROT TRAINING DATA (AUGMENTED) =====================
def _step_quadratic(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    zr, zi = float(z[0]), float(z[1])
    cr, ci = float(c[0]), float(c[1])
    return np.array([zr * zr - zi * zi + cr, 2.0 * zr * zi + ci], dtype=np.float32)

def build_mandelbrot_training_data(
    *,
    c_re_min: float,
    c_re_max: float,
    c_im_min: float,
    c_im_max: float,
    n_c: int = 6000,
    max_iters: int = 40,
    escape_r: float = 2.0,
    seed: int = 0,
) -> TrainingData:
    rng = np.random.default_rng(seed)
    cr = rng.uniform(c_re_min, c_re_max, size=n_c).astype(np.float32)
    ci = rng.uniform(c_im_min, c_im_max, size=n_c).astype(np.float32)
    C = np.stack([cr, ci], axis=1).astype(np.float32)  # (n_c,2)

    r2 = float(escape_r) * float(escape_r)

    X_list, X1_list, X2_list = [], [], []

    for c in C:
        z = np.zeros(2, dtype=np.float32)
        orbit = []

        for _ in range(max_iters):
            z = _step_quadratic(z, c)
            # IMPORTANT: AUGMENT STATE WITH c SO AE/DMD CAN RECONSTRUCT THE SET
            x = np.array([z[0], z[1], c[0], c[1]], dtype=np.float32)  # (4,)
            orbit.append(x)
            if (z[0] * z[0] + z[1] * z[1]) > r2:
                break

        if len(orbit) >= 2:
            o = np.stack(orbit, axis=0).astype(np.float32)  # (T,4)
            X_list.append(o)
            X1_list.append(o[:-1])
            X2_list.append(o[1:])

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    X1 = np.concatenate(X1_list, axis=0).astype(np.float32)
    X2 = np.concatenate(X2_list, axis=0).astype(np.float32)

    X = _sanitize_finite(X, "X")
    X1 = _sanitize_finite(X1, "X1")
    X2 = _sanitize_finite(X2, "X2")

    mn, mx = float(X.min()), float(X.max())
    X = (X - mn) / (mx - mn + 1e-12)
    X1 = (X1 - mn) / (mx - mn + 1e-12)
    X2 = (X2 - mn) / (mx - mn + 1e-12)

    return TrainingData(
        X=X.astype(np.float32),
        X1=X1.astype(np.float32),
        X2=X2.astype(np.float32),
        meta={
            "mode": "mandelbrot_augmented_state",
            "n_c": int(n_c),
            "max_iters": int(max_iters),
            "escape_r": float(escape_r),
            "bbox": (float(c_re_min), float(c_re_max), float(c_im_min), float(c_im_max)),
            "mn": mn,
            "mx": mx,
            "seed": int(seed),
        },
    )
