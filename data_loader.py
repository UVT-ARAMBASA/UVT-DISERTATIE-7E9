# ================================ data_loader.py ================================
from __future__ import annotations  # TYPE HINTS

import numpy as np  # NUMPY
from pathlib import Path  # PATH
from scipy.io import loadmat  # MATLAB

# ================================ LOADERS =====================================
def load_matlab_matrix(path: str, key: str | None = None):  # LOAD MAT
    mat = loadmat(path)  # READ
    if key is None:  # AUTO KEY
        key = next(k for k in mat.keys() if not k.startswith("__"))  # PICK
    return mat[key]  # RETURN

def load_npz_array(path: str | Path, key: str | None = None) -> np.ndarray:  # LOAD NPZ
    path = Path(path)  # PATH
    with np.load(path) as data:  # OPEN
        if key is None:  # AUTO KEY
            key = data.files[0]  # FIRST
        return data[key]  # RETURN

def load_task_npz_pair(  # LOAD TWO NPZ
    data_dir: str | Path,
    emotion_file: str = "task-emotion.npz",
    rest_file: str = "task-rest.npz",
    key: str | None = None,
    flatten: bool = True,
):
    data_dir = Path(data_dir)  # PATH
    X_emotion = load_npz_array(data_dir / emotion_file, key=key)  # LOAD
    X_rest = load_npz_array(data_dir / rest_file, key=key)  # LOAD

    if flatten:  # FLATTEN
        X_emotion = X_emotion.reshape(X_emotion.shape[0], -1)  # FLAT
        X_rest = X_rest.reshape(X_rest.shape[0], -1)  # FLAT

    return X_emotion, X_rest  # RETURN

# =============================== PAIRS ========================================
def make_dmd_pairs(X: np.ndarray):  # DMD PAIRS
    return X[:-1], X[1:]  # SHIFT

# ========================= TRAJECTORY GENERATION ==============================
def generate_state_trajectories(  # GEN TRAJ
    A_seq: np.ndarray,  # (T,n,n) REAL
    n_traj: int = 500,  # COUNT
    x0_scale: float = 1.0,  # INIT SCALE
    noise_std: float = 0.0,  # NOISE
    escape_r: float = 2.0,  # CLAMP R
    c: complex | np.ndarray | None = None,  # COMPLEX C
    c_scale: float = 0.0,  # RANDOM C SCALE
    x0_mode: str = "random",  # "random" OR "zeros"
    use_square: bool = True,  # APPLY SQUARE
):
    T, n, _ = A_seq.shape  # SHAPES
    A_seq = np.asarray(A_seq, dtype=np.float32)  # FP32

    r = float(escape_r)  # R
    if r <= 0.0:  # SAFE
        r = 2.0  # DEFAULT

    rng = np.random.default_rng(0)  # RNG

    # BUILD C LIST  # COMPLEX
    if c is None:  # RANDOM C
        if c_scale > 0.0:  # USE SCALE
            cr = rng.uniform(-c_scale, c_scale, size=n_traj).astype(np.float32)  # RE
            ci = rng.uniform(-c_scale, c_scale, size=n_traj).astype(np.float32)  # IM
            C_list = (cr + 1j * ci).astype(np.complex64)  # COMPLEX
        else:
            C_list = (np.zeros((n_traj,), dtype=np.complex64))  # ZERO
    elif np.isscalar(c):  # ONE C
        C_list = (np.full((n_traj,), np.complex64(c), dtype=np.complex64))  # REPEAT
    else:  # ARRAY C
        C_list = np.asarray(c, dtype=np.complex64).reshape(-1)  # FLAT
        if C_list.size != n_traj:  # FIX SIZE
            C_list = np.resize(C_list, (n_traj,)).astype(np.complex64)  # RESIZE

    X_all, X1_all, X2_all = [], [], []  # BUFFERS

    for j in range(int(n_traj)):  # EACH TRAJ
        # INIT Z0  # COMPLEX VECTOR
        if x0_mode == "zeros":  # ZERO
            z = np.zeros((n,), dtype=np.complex64)  # Z0
        else:  # RANDOM
            zr = rng.normal(0.0, 1.0, size=n).astype(np.float32)  # RE
            zi = rng.normal(0.0, 1.0, size=n).astype(np.float32)  # IM
            z = (x0_scale * (zr + 1j * zi)).astype(np.complex64)  # COMPLEX

        cj = np.complex64(C_list[j])  # C

        traj = []  # STORE

        for t in range(int(T)):  # TIME
            A = A_seq[t]  # MATRIX
            Az = (A @ z).astype(np.complex64)  # APPLY A

            if use_square:  # SQUARE
                z_next = (Az * Az + cj).astype(np.complex64)  # (Az)^2 + c
            else:
                z_next = (Az + cj).astype(np.complex64)  # Az + c

            # ADD NOISE  # OPTIONAL
            if noise_std > 0.0:  # NOISE
                nr = rng.normal(0.0, noise_std, size=n).astype(np.float32)  # RE
                ni = rng.normal(0.0, noise_std, size=n).astype(np.float32)  # IM
                z_next = (z_next + (nr + 1j * ni).astype(np.complex64)).astype(np.complex64)  # ADD

            # CLAMP IF EXPLODE  # ESCAPE RADIUS
            mag = np.abs(z_next).astype(np.float32)  # |Z|
            bad = (~np.isfinite(mag)) | (mag > r)  # MASK
            if np.any(bad):  # CLAMP
                # SCALE EACH BAD COMPONENT TO R  # COMPONENTWISE
                mag_safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)  # SAFE
                z_next = np.where(bad, (r / mag_safe) * z_next, z_next).astype(np.complex64)  # SCALE

            # SAVE SNAPSHOT  # REAL STACK
            xr = z_next.real.astype(np.float32)  # RE
            xi = z_next.imag.astype(np.float32)  # IM
            traj.append(np.concatenate([xr, xi], axis=0).astype(np.float32))  # (2n,)

            z = z_next  # ADVANCE

        X = np.stack(traj, axis=0).astype(np.float32)  # (T,2n)
        X1, X2 = X[:-1], X[1:]  # PAIRS

        X_all.append(X)  # ADD
        X1_all.append(X1)  # ADD
        X2_all.append(X2)  # ADD

    X_all = np.concatenate(X_all, axis=0).astype(np.float32)  # FLAT
    X1_all = np.concatenate(X1_all, axis=0).astype(np.float32)  # FLAT
    X2_all = np.concatenate(X2_all, axis=0).astype(np.float32)  # FLAT

    return X_all, X1_all, X2_all  # RETURN

def load_one_A_matrix(  # LOAD ONE MATRIX
    data_dir: str | Path,  # DATA FOLDER
    source: str = "emotion",  # WHICH FILE
    index: int = 0,  # WHICH MATRIX
):
    data_dir = Path(data_dir)  # PATH
    if str(source).lower() == "rest":  # PICK REST
        A_all = load_npz_array(data_dir / "task-rest.npz")  # LOAD REST
    else:  # PICK EMOTION
        A_all = load_npz_array(data_dir / "task-emotion.npz")  # LOAD EMOTION

    A_all = np.asarray(A_all, dtype=np.float32)  # FP32
    A = A_all[int(index)]  # PICK ONE
    return A.astype(np.float32)  # RETURN MATRIX

def load_all_A_matrices(  # LOAD ALL MATRICES
    data_dir: str | Path,  # DATA FOLDER
    source: str = "emotion",  # WHICH FILE
) -> np.ndarray:
    data_dir = Path(data_dir)  # PATH
    if str(source).lower() == "rest":  # PICK REST
        A_all = load_npz_array(data_dir / "task-rest.npz")  # LOAD REST
    else:  # PICK EMOTION
        A_all = load_npz_array(data_dir / "task-emotion.npz")  # LOAD EMOTION

    A_all = np.asarray(A_all, dtype=np.float32)  # FP32
    return A_all  # RETURN ALL

def split_explicit_matrix_indices(  # EXPLICIT TRAIN/TEST SPLIT
    total_count: int,  # HOW MANY MATRICES
    train_count: int,  # HOW MANY TRAIN
    test_count: int,  # HOW MANY TEST
    seed: int = 0,  # REPRO
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(int(total_count), dtype=np.int64)  # ALL INDICES
    rng = np.random.default_rng(int(seed))  # RNG
    rng.shuffle(idx)  # SHUFFLE

    train_idx = np.sort(idx[:int(train_count)])  # TRAIN
    test_idx = np.sort(idx[int(train_count):int(train_count) + int(test_count)])  # TEST
    return train_idx, test_idx  # RETURN