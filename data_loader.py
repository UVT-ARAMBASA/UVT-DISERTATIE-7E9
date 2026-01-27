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

def normalize_data(x):
    x = np.asarray(x, dtype=np.float32)
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + 1e-12)

def make_dmd_pairs(X: np.ndarray):
    # Classic DMD snapshot pairs: (x_k, x_{k+1})
    return X[:-1], X[1:]

def generate_state_trajectories(A_seq: np.ndarray,
                                n_traj: int = 500,
                                x0_scale: float = 1.0,
                                noise_std: float = 0.0):
    """
    A_seq: (T, n, n) time-varying matrices A_k
    returns:
      X:  (n_traj*T, n) stacked states
      X1: (n_traj*(T-1), n)
      X2: (n_traj*(T-1), n)
    """
    T, n, _ = A_seq.shape

    X_list = []
    X1_list = []
    X2_list = []

    for _ in range(n_traj):
        x = (np.random.randn(n).astype(np.float32) * x0_scale)

        traj = []
        for k in range(T):
            traj.append(x.copy())
            x = A_seq[k] @ x
            if noise_std > 0:
                x = x + noise_std * np.random.randn(n).astype(np.float32)

        traj = np.stack(traj, axis=0)          # (T, n)
        X_list.append(traj)

        X1_list.append(traj[:-1])
        X2_list.append(traj[1:])

    X  = np.concatenate(X_list,  axis=0)       # (n_traj*T, n)
    X1 = np.concatenate(X1_list, axis=0)       # (n_traj*(T-1), n)
    X2 = np.concatenate(X2_list, axis=0)       # (n_traj*(T-1), n)
    return X, X1, X2


