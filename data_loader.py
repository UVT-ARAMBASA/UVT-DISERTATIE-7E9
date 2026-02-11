# data_loader.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy.io import loadmat

def load_matlab_matrix(path: str, key: str | None = None):
    mat = loadmat(path)
    if key is None:
        key = next(k for k in mat.keys() if not k.startswith("__"))
    return mat[key]

def load_npz_array(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    with np.load(path) as data:
        if key is None:
            # pick first array inside the .npz
            key = data.files[0]
        return data[key]

def load_task_npz_pair(data_dir: str | Path,
                       emotion_file: str = "task-emotion.npz",
                       rest_file: str = "task-rest.npz",
                       key: str | None = None,
                       flatten: bool = True):
    data_dir = Path(data_dir)
    X_emotion = load_npz_array(data_dir / emotion_file, key=key)
    X_rest    = load_npz_array(data_dir / rest_file, key=key)

    # If arrays are time x (anything...), flatten spatial dims into features
    if flatten:
        X_emotion = X_emotion.reshape(X_emotion.shape[0], -1)
        X_rest    = X_rest.reshape(X_rest.shape[0], -1)

    return X_emotion, X_rest

def normalize_data(x):  # NORMALISE FN
    x = np.asarray(x, dtype=np.float32)  # FP32 ARRAY
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # FINITE ONLY
    mn = float(np.min(x))  # MIN
    mx = float(np.max(x))  # MAX
    den = (mx - mn)  # DENOM
    if (not np.isfinite(mn)) or (not np.isfinite(mx)):  # BAD RANGE
        return np.zeros_like(x, dtype=np.float32)  # ZERO ARRAY
    if abs(den) < 1e-12:  # NEAR CONSTANT
        return np.zeros_like(x, dtype=np.float32)  # ZERO ARRAY
    return (x - mn) / (den + 1e-12)  # [0,1] SCALE



def make_dmd_pairs(X: np.ndarray):
    # Classic DMD snapshot pairs: (x_k, x_{k+1})
    return X[:-1], X[1:]

def generate_state_trajectories(A_seq: np.ndarray,  # A SEQ IN
                                n_traj: int = 500,  # TRAJ COUNT
                                x0_scale: float = 1.0,  # INIT SCALE
                                noise_std: float = 0.0,  # NOISE STD
                                x_cap: float = 1e3):  # NORM CAP
    T, n, _ = A_seq.shape  # SHAPES
    A_seq = np.asarray(A_seq, dtype=np.float32)  # FP32 A SEQ

    X_list = []  # X LIST
    X1_list = []  # X1 LIST
    X2_list = []  # X2 LIST

    for _ in range(n_traj):  # TRAJ LOOP
        x = (np.random.randn(n).astype(np.float32) * x0_scale)  # X0
        traj = []  # TRAJ LIST

        for k in range(T):  # TIME LOOP
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)  # FINITE X
            norm = float(np.linalg.norm(x))  # NORM
            if norm > x_cap:  # CAP CHECK
                x = (x / (norm + 1e-12)) * x_cap  # CAP SCALE

            traj.append(x.copy())  # PUSH STATE

            x = A_seq[k] @ x  # STATE UPDATE
            if noise_std > 0.0:  # NOISE FLAG
                x = x + noise_std * np.random.randn(n).astype(np.float32)  # NOISE ADD
#ELIMINATE
        traj = np.stack(traj, axis=0).astype(np.float32)  # (T,N) TRAJ
        traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)  # FINITE TRAJ

        X_list.append(traj)  # X PUSH
        X1_list.append(traj[:-1])  # X1 PUSH
        X2_list.append(traj[1:])  # X2 PUSH

    X = np.concatenate(X_list, axis=0).astype(np.float32)  # X STACK
    X1 = np.concatenate(X1_list, axis=0).astype(np.float32)  # X1 STACK
    X2 = np.concatenate(X2_list, axis=0).astype(np.float32)  # X2 STACK
    return X, X1, X2  # OUTPUT



