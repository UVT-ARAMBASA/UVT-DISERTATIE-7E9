from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================ IMPORTS ==================================
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH

import defines as D  # DEFINES
from utils import to_tensor  # TENSOR HELPER
from apply_dmd import fit_dmd_from_latent_covariances  # STREAMED DMD FIT


# ============================== DEVICE PICKER ===============================
def pick_device() -> torch.device:  # SELECT DEVICE
    if D.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():  # CUDA AVAILABLE
        return torch.device("cuda")  # GPU
    return torch.device("cpu")  # CPU


# ============================== OUTPUT LAYOUT ===============================
def make_out_dirs(mode: str) -> dict:  # BUILD out/<mode>/{training-data,results}
    root = Path("out") / mode  # MODE ROOT
    td = root / "training-data"  # TRAINING DATA DIR
    res = root / "results"  # RESULTS DIR
    td.mkdir(parents=True, exist_ok=True)  # MAKE TD
    res.mkdir(parents=True, exist_ok=True)  # MAKE RES
    return {"root": root, "td": td, "res": res}  # RETURN DICT


# =============================== DMD HELPERS ================================
def fit_streamed_dmd_from_td_list(enc, td_list: list, device: torch.device):  # STREAMED LATENT DMD
    latent_dim = int(D.LATENT_DIM)  # LATENT DIM
    G = np.zeros((latent_dim, latent_dim), dtype=np.float64)  # Z1^T Z1
    H = np.zeros((latent_dim, latent_dim), dtype=np.float64)  # Z1^T Z2

    with torch.no_grad():  # NO GRAD
        for td in td_list:  # LOOP DATASETS
            Z1 = enc(to_tensor(td.X1, device)).detach().cpu().numpy().astype(np.float64)  # Z1
            Z2 = enc(to_tensor(td.X2, device)).detach().cpu().numpy().astype(np.float64)  # Z2
            G += Z1.T @ Z1  # ACCUMULATE LEFT COVARIANCE
            H += Z1.T @ Z2  # ACCUMULATE CROSS COVARIANCE

    return fit_dmd_from_latent_covariances(G, H, device=device)  # BUILD DMD


# ================================ PRINTING ==================================
def print_metric_block(name: str, metrics: dict) -> None:  # PRETTY PRINT
    print(f"[{name}]")  # HEADER
    for k, v in metrics.items():  # LOOP METRICS
        print(f"  {k}: {v:.8e}")  # PRINT VALUE


def mean_metric_dict(metric_list: list) -> dict:  # MEAN OVER LIST
    keys = metric_list[0].keys()  # KEYS
    out = {}  # OUTPUT
    for k in keys:  # LOOP KEYS
        out[k] = float(np.mean([m[k] for m in metric_list]))  # MEAN
    return out  # RETURN


def write_metrics_txt(path, metrics: dict) -> None:  # SAVE METRICS TO TXT
    path = Path(path)  # PATH
    path.parent.mkdir(parents=True, exist_ok=True)  # MKDIR
    with open(path, "w", encoding="utf-8") as f:  # OPEN
        for k, v in metrics.items():  # LOOP
            f.write(f"{k}: {v:.10e}\n")  # WRITE


def debug_final_state_stats(name: str, Z_final: np.ndarray, escape_r: float) -> None:  # DEBUG FINAL STATE
    d = Z_final.shape[-1] // 2  # STATE DIM
    zr = Z_final[..., 0:d].astype(np.float32)  # RE
    zi = Z_final[..., d:2 * d].astype(np.float32)  # IM
    mag2 = zr * zr + zi * zi  # MAG2
    max_mag = np.sqrt(np.max(mag2, axis=-1))  # MAX COMPONENT MAG PER PIXEL

    print(f"\n[{name}] FINAL PRED STATS")  # HEADER
    print("min max_mag:", float(np.min(max_mag)))  # MIN
    print("mean max_mag:", float(np.mean(max_mag)))  # MEAN
    print("max max_mag:", float(np.max(max_mag)))  # MAX
    print("escape_r:", float(escape_r))  # RADIUS
    print("white ratio:", float(np.mean(max_mag < float(escape_r))))  # WHITE FRACTION