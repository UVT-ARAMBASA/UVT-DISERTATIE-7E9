# ============================ optional_baselines.py ==========================
from __future__ import annotations  # MODERN TYPE HINTS

# ================================= IMPORTS ==================================
from pathlib import Path  # PATHS
import numpy as np  # NUMPY
import torch  # TORCH
from PIL import Image  # IMAGE SAVE
from torch.utils.data import DataLoader, TensorDataset  # TRAIN LOADER

import defines as D  # PROJECT DEFINES

from encoder import Encoder  # ENCODER MODEL
from decoder import Decoder  # DECODER MODEL
from losses import make_reconstruction_loss  # BASIC AE LOSS
from utils import to_tensor  # TENSOR HELPER

from data_loader import load_all_A_matrices, split_explicit_matrix_indices  # MATRIX SPLIT

from prepare_training_data import (  # TRAINING DATA BUILDERS
    build_matrix_c_grid_training_data,  # SINGLE MATRIX DATA
    build_matrix_c_grid_training_data_many_matrices,  # MULTI MATRIX DATA
    save_training_npz,  # SAVE NPZ
)

from apply_dmd import fit_dmd_from_latent_covariances  # STREAMED DMD FIT

from eval_matrix_dmd_ae import (  # EXISTING VISUAL HELPERS
    save_loss_curve,  # LOSS PLOT
    autoencoder_reconstruction_metrics,  # AE METRICS
    save_ground_truth_final_mask,  # GT MASK
    save_ground_truth_escape_iters,  # GT ESCAPE ITERS
)

from mandelbrot_reconstruct import save_final_snapshot_image  # FINAL SNAPSHOT IMAGE


# =============================== PATH HELPERS ================================
def _mkdir(path: str | Path) -> Path:  # MAKE DIR
    p = Path(path)  # CONVERT
    p.mkdir(parents=True, exist_ok=True)  # CREATE
    return p  # RETURN


def _phase_dirs(root: str | Path, phase: str) -> dict[str, Path]:  # BUILD OUTPUT DIRS
    root = Path(root)  # ROOT
    return {  # RETURN DICT
        "td_single": _mkdir(root / "training-data" / "single-matrix"),  # SINGLE TRAINING DATA
        "td_multi": _mkdir(root / "training-data" / "multiple-matrices"),  # MULTI TRAINING DATA
        "res_single": _mkdir(root / "results" / "single-matrix"),  # SINGLE RESULTS
        "res_multi": _mkdir(root / "results" / "multiple-matrices"),  # MULTI RESULTS
    }


# =============================== METRICS =====================================
def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:  # RELATIVE L2
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))  # SAFE


def _mse(pred: np.ndarray, true: np.ndarray) -> float:  # MSE
    err = pred - true  # ERROR
    return float(np.mean(err * err))  # MEAN SQUARED ERROR


def _write_metrics(path: str | Path, metrics: dict[str, float]) -> None:  # SAVE METRICS TXT
    path = Path(path)  # PATH
    path.parent.mkdir(parents=True, exist_ok=True)  # DIR
    with open(path, "w", encoding="utf-8") as f:  # OPEN
        for k, v in metrics.items():  # LOOP
            f.write(f"{k}: {v:.10e}\n")  # WRITE


def _print_metrics(title: str, metrics: dict[str, float]) -> None:  # PRINT METRICS
    print(f"\n[{title}]")  # HEADER
    for k, v in metrics.items():  # LOOP
        print(f"  {k}: {v:.10e}")  # VALUE


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:  # MEAN METRICS
    if not items:  # EMPTY
        return {}  # RETURN EMPTY
    keys = items[0].keys()  # KEYS
    return {k: float(np.mean([m[k] for m in items])) for k in keys}  # MEAN


# ============================= IMAGE HELPERS =================================
def _save_final_mask_95(  # SAVE MASK LIKE YOUR CURRENT FINAL MASK
    Z_final: np.ndarray,  # FINAL SNAPSHOT, SHAPE (H,W,2D)
    escape_r: float,  # ESCAPE RADIUS
    out_png: str | Path,  # OUTPUT
    scale: int = 64,  # UPSCALE
) -> str:
    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # DIR

    d = int(Z_final.shape[-1] // 2)  # STATE DIM
    zr = Z_final[..., 0:d].astype(np.float32, copy=False)  # REAL
    zi = Z_final[..., d:2 * d].astype(np.float32, copy=False)  # IMAG
    r2 = float(escape_r) * float(escape_r)  # RADIUS SQUARED

    comp_mag2 = zr * zr + zi * zi  # COMPONENT MAGNITUDES
    stable_comp = comp_mag2 < r2  # COMPONENTWISE STABLE
    stable_frac = np.mean(stable_comp, axis=-1)  # FRACTION STABLE
    mask = stable_frac >= 0.95  # SAME IDEA AS CURRENT CODE

    img = (mask.astype(np.uint8) * 255)  # BLACK WHITE
    pil = Image.fromarray(img, mode="L")  # IMAGE
    pil = pil.resize(  # UPSCALE
        (img.shape[1] * int(scale), img.shape[0] * int(scale)),  # SIZE
        resample=Image.NEAREST,  # SHARP PIXELS
    )
    pil.save(out_png)  # SAVE
    return str(out_png)  # RETURN


# ============================= AE ONLY TRAINER ===============================
def train_autoencoder_reconstruction_only(  # TRAIN PURE AE
    X: np.ndarray | list[np.ndarray],  # DATA OR LIST OF DATA BLOCKS
    *,
    latent_dim: int,  # LATENT SIZE
    epochs: int,  # EPOCHS
    batch_size: int,  # BATCH SIZE
    lr: float,  # LEARNING RATE
    device: torch.device,  # DEVICE
) -> tuple[Encoder, Decoder, list[float]]:  # RETURN MODELS + LOSSES
    if isinstance(X, np.ndarray):  # SINGLE ARRAY
        X_list = [X]  # WRAP
    else:  # LIST
        X_list = list(X)  # COPY

    if len(X_list) == 0:  # EMPTY CHECK
        raise ValueError("AE ONLY RECEIVED EMPTY X LIST")  # ERROR

    in_dim = int(X_list[0].shape[1])  # INPUT DIM
    enc = Encoder(in_dim, latent_dim).to(device)  # ENCODER
    dec = Decoder(latent_dim, in_dim).to(device)  # DECODER

    loss_fn = make_reconstruction_loss(0, beta=0.01, w_pow=1.0)  # MSE
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)  # OPTIMIZER

    losses: list[float] = []  # STORE LOSSES

    for e in range(int(epochs)):  # EPOCH LOOP
        rng = np.random.default_rng(e)  # RNG
        order = rng.permutation(len(X_list))  # SHUFFLE BLOCKS

        s = 0.0  # LOSS SUM
        n_batches = 0  # BATCH COUNT

        for block_id in order:  # BLOCK LOOP
            X_block = np.asarray(X_list[int(block_id)], dtype=np.float32)  # BLOCK
            X_cpu = torch.tensor(X_block, dtype=torch.float32)  # CPU TENSOR

            loader = DataLoader(  # LOADER
                TensorDataset(X_cpu),  # ONLY X
                batch_size=int(batch_size),  # BATCH SIZE
                shuffle=True,  # SHUFFLE
                drop_last=False,  # KEEP ALL
            )

            for (b_cpu,) in loader:  # BATCH LOOP
                b = b_cpu.to(device)  # MOVE TO DEVICE

                z = enc(b)  # ENCODE
                x_rec = dec(z)  # DECODE
                loss = loss_fn(x_rec, b)  # PURE RECONSTRUCTION LOSS

                opt.zero_grad()  # ZERO GRAD
                loss.backward()  # BACKPROP
                opt.step()  # UPDATE

                s += float(loss.detach().cpu().item())  # ACCUMULATE
                n_batches += 1  # COUNT

        s /= max(1, n_batches)  # MEAN LOSS
        losses.append(s)  # STORE
        print(f"[AE ONLY] EPOCH {e + 1}/{epochs} LOSS={s:.6e}")  # LOG

    return enc, dec, losses  # RETURN


# ============================= AE ONLY OUTPUT ================================
@torch.no_grad()  # NO GRAD
def reconstruct_final_snapshot_ae_only(  # ENCODE-DECODE FINAL GT SNAPSHOT
    td,  # TRAINING DATA
    encoder,  # ENCODER
    decoder,  # DECODER
    device: torch.device,  # DEVICE
    batch_size: int = 50000,  # BATCH
) -> np.ndarray:  # RETURNS (H,W,2D)
    if td.X_grid is None:  # NEED GRID
        raise ValueError("AE ONLY FINAL SNAPSHOT NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEATURE DIM
    d = int((feat_dim - 2) // 2)  # STATE DIM
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH

    X_final = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)  # TRUE FINAL X_T
    C = X_final[:, 2 * d:2 * d + 2].copy().astype(np.float32)  # C VALUES

    X_out = np.zeros((X_final.shape[0], 2 * d), dtype=np.float32)  # OUTPUT

    for i0 in range(0, X_final.shape[0], int(batch_size)):  # BATCH LOOP
        i1 = min(X_final.shape[0], i0 + int(batch_size))  # END
        x = to_tensor(X_final[i0:i1], device)  # TO DEVICE

        x_rec = decoder(encoder(x))  # AE ONLY: ENCODE -> DECODE
        x_rec[:, 2 * d] = to_tensor(C[i0:i1, 0], device)  # KEEP CR EXACT
        x_rec[:, 2 * d + 1] = to_tensor(C[i0:i1, 1], device)  # KEEP CI EXACT

        x_np = x_rec.detach().cpu().numpy().astype(np.float32)  # CPU
        X_out[i0:i1, 0:d] = x_np[:, 0:d]  # REAL
        X_out[i0:i1, d:2 * d] = x_np[:, d:2 * d]  # IMAG

    return X_out.reshape(H, W, 2 * d)  # IMAGE SHAPE


# ============================= DMD ONLY FIT ==================================
def fit_full_state_dmd_streamed(  # FIT RAW DMD WITHOUT AE
    X1: np.ndarray | list[np.ndarray],  # LEFT SNAPSHOTS
    X2: np.ndarray | list[np.ndarray],  # RIGHT SNAPSHOTS
    *,
    device: torch.device,  # DEVICE
):  # RETURNS DMDDynamics
    if isinstance(X1, np.ndarray):  # SINGLE
        X1_list = [X1]  # WRAP
    else:  # LIST
        X1_list = list(X1)  # COPY

    if isinstance(X2, np.ndarray):  # SINGLE
        X2_list = [X2]  # WRAP
    else:  # LIST
        X2_list = list(X2)  # COPY

    if len(X1_list) != len(X2_list):  # MISMATCH
        raise ValueError("DMD ONLY X1 AND X2 LISTS MUST HAVE SAME LENGTH")  # ERROR

    feat_dim = int(X1_list[0].shape[1])  # FEATURE DIM
    G = np.zeros((feat_dim, feat_dim), dtype=np.float64)  # X1^T X1
    H = np.zeros((feat_dim, feat_dim), dtype=np.float64)  # X1^T X2

    for a, b in zip(X1_list, X2_list):  # BLOCK LOOP
        A = np.asarray(a, dtype=np.float64)  # LEFT
        B = np.asarray(b, dtype=np.float64)  # RIGHT
        G += A.T @ A  # ACCUMULATE
        H += A.T @ B  # ACCUMULATE

    return fit_dmd_from_latent_covariances(G, H, device=device)  # SAME FORMULA, FULL STATE


@torch.no_grad()  # NO GRAD
def dmd_only_one_step_metrics(  # RAW DMD METRICS
    dmd,  # DMD MODEL
    X1: np.ndarray,  # LEFT
    X2: np.ndarray,  # RIGHT
    device: torch.device,  # DEVICE
) -> dict[str, float]:
    x1 = to_tensor(X1.astype(np.float32), device)  # LEFT
    x2_hat = dmd.predict(x1, steps=1)[-1]  # ONE STEP
    pred = x2_hat.detach().cpu().numpy().astype(np.float32)  # CPU

    return {  # METRICS
        "dmd_only_mse": _mse(pred, X2.astype(np.float32)),  # MSE
        "dmd_only_rel_l2": _rel_l2(pred, X2.astype(np.float32)),  # REL L2
        "dmd_only_fit": float(1.0 - _rel_l2(pred, X2.astype(np.float32))),  # FIT
    }


@torch.no_grad()  # NO GRAD
def reconstruct_final_snapshot_dmd_only(  # RAW DMD ROLLOUT
    td,  # TRAINING DATA
    dmd,  # DMD MODEL
    device: torch.device,  # DEVICE
    escape_r: float,  # ESCAPE RADIUS
    batch_size: int = 50000,  # BATCH
) -> np.ndarray:  # RETURNS (H,W,2D)
    if td.X_grid is None:  # NEED GRID
        raise ValueError("DMD ONLY FINAL SNAPSHOT NEEDS td.X_grid")  # ERROR

    feat_dim = int(td.X_grid.shape[-1])  # FEATURE DIM
    d = int((feat_dim - 2) // 2)  # STATE DIM
    T = int(td.X_grid.shape[0])  # STORED STEPS
    H = int(td.X_grid.shape[1])  # HEIGHT
    W = int(td.X_grid.shape[2])  # WIDTH

    X0 = td.X_grid[0].reshape(-1, feat_dim).astype(np.float32)  # START FROM x1
    C = X0[:, 2 * d:2 * d + 2].copy().astype(np.float32)  # C VALUES

    X_out = np.zeros((X0.shape[0], 2 * d), dtype=np.float32)  # OUTPUT
    n_roll = max(T - 1, 0)  # x1 -> xT

    for i0 in range(0, X0.shape[0], int(batch_size)):  # BATCH LOOP
        i1 = min(X0.shape[0], i0 + int(batch_size))  # END
        X_t = to_tensor(X0[i0:i1], device)  # START
        C_t = to_tensor(C[i0:i1], device)  # C

        for _ in range(n_roll):  # ROLLOUT
            X_t = dmd.predict(X_t, steps=1)[-1]  # RAW DMD STEP

            X_t[:, 2 * d] = C_t[:, 0]  # KEEP CR EXACT
            X_t[:, 2 * d + 1] = C_t[:, 1]  # KEEP CI EXACT

            x_np = X_t.detach().cpu().numpy().astype(np.float32)  # CPU
            zr = x_np[:, 0:d]  # REAL
            zi = x_np[:, d:2 * d]  # IMAG

            mag2 = zr * zr + zi * zi  # MAG SQUARED
            mag = np.sqrt(np.maximum(mag2, 1e-30)).astype(np.float32)  # MAG
            bad = (~np.isfinite(mag)) | (mag > float(escape_r))  # EXPLODED

            if np.any(bad):  # CLAMP
                safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)  # SAFE
                scale = (float(escape_r) / safe).astype(np.float32)  # SCALE
                zr = np.where(bad, zr * scale, zr).astype(np.float32)  # CLAMP REAL
                zi = np.where(bad, zi * scale, zi).astype(np.float32)  # CLAMP IMAG
                x_np[:, 0:d] = zr  # WRITE REAL
                x_np[:, d:2 * d] = zi  # WRITE IMAG

            X_t = to_tensor(x_np, device)  # BACK TO DEVICE

        x_final = X_t.detach().cpu().numpy().astype(np.float32)  # FINAL
        X_out[i0:i1, 0:d] = x_final[:, 0:d]  # REAL
        X_out[i0:i1, d:2 * d] = x_final[:, d:2 * d]  # IMAG

    return X_out.reshape(H, W, 2 * d)  # IMAGE SHAPE


# =========================== DATA BUILDERS USED HERE =========================
def _build_single_td():  # BUILD SINGLE MATRIX DATA
    return build_matrix_c_grid_training_data(  # SAME AS MAIN
        data_dir=D.A_DATA_DIR,  # DATA DIR
        source=D.SINGLE_MATRIX_SOURCE,  # SOURCE
        index=D.SINGLE_MATRIX_INDEX,  # MATRIX INDEX
        c_re_min=D.C_RE_MIN,  # RE MIN
        c_re_max=D.C_RE_MAX,  # RE MAX
        c_im_min=D.C_IM_MIN,  # IM MIN
        c_im_max=D.C_IM_MAX,  # IM MAX
        c_re_n=D.SINGLE_MATRIX_C_RE_N,  # RE RES
        c_im_n=D.SINGLE_MATRIX_C_IM_N,  # IM RES
        max_iters=D.TRAIN_MAX_ITERS,  # ITERS
        escape_r=D.ESCAPE_R,  # ESCAPE
    )


def _build_multi_train_test():  # BUILD MULTI MATRIX DATA
    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)  # LOAD ALL
    total = int(A_all.shape[0])  # COUNT

    train_idx, test_idx = split_explicit_matrix_indices(  # SPLIT
        total_count=total,  # TOTAL
        train_count=D.MULTI_MATRIX_TRAIN_COUNT,  # TRAIN COUNT
        test_count=D.MULTI_MATRIX_TEST_COUNT,  # TEST COUNT
        seed=D.MULTI_MATRIX_SPLIT_SEED,  # SEED
    )

    td_train_list = build_matrix_c_grid_training_data_many_matrices(  # TRAIN DATA
        data_dir=D.A_DATA_DIR,  # DATA DIR
        source=D.MULTI_MATRIX_SOURCE,  # SOURCE
        indices=train_idx,  # TRAIN IDX
        c_re_min=D.C_RE_MIN,  # RE MIN
        c_re_max=D.C_RE_MAX,  # RE MAX
        c_im_min=D.C_IM_MIN,  # IM MIN
        c_im_max=D.C_IM_MAX,  # IM MAX
        c_re_n=D.MULTI_MATRIX_C_RE_N,  # RE RES
        c_im_n=D.MULTI_MATRIX_C_IM_N,  # IM RES
        max_iters=D.TRAIN_MAX_ITERS,  # ITERS
        escape_r=D.ESCAPE_R,  # ESCAPE
    )

    td_test_list = build_matrix_c_grid_training_data_many_matrices(  # TEST DATA
        data_dir=D.A_DATA_DIR,  # DATA DIR
        source=D.MULTI_MATRIX_SOURCE,  # SOURCE
        indices=test_idx,  # TEST IDX
        c_re_min=D.C_RE_MIN,  # RE MIN
        c_re_max=D.C_RE_MAX,  # RE MAX
        c_im_min=D.C_IM_MIN,  # IM MIN
        c_im_max=D.C_IM_MAX,  # IM MAX
        c_re_n=D.MULTI_MATRIX_C_RE_N,  # RE RES
        c_im_n=D.MULTI_MATRIX_C_IM_N,  # IM RES
        max_iters=D.TRAIN_MAX_ITERS,  # ITERS
        escape_r=D.ESCAPE_R,  # ESCAPE
    )

    return train_idx, test_idx, td_train_list, td_test_list  # RETURN


# =============================== AE ONLY RUN =================================
def run_ae_only_baseline(device: torch.device) -> None:  # RUN AE ONLY
    print("\n================ AE ONLY BASELINE ================\n")  # HEADER

    dirs = _phase_dirs(D.AE_ONLY_OUT_DIR, "AE-only")  # DIRS

    # --------------------------- SINGLE MATRIX ------------------------------
    print("\n---------- AE ONLY / SINGLE MATRIX ----------\n")  # HEADER
    td_single = _build_single_td()  # BUILD DATA

    save_ground_truth_escape_iters(td_single, D.ESCAPE_R, dirs["td_single"] / "gt_escape_iters.png")  # GT ESC
    save_training_npz(dirs["td_single"] / "training_single_matrix.npz", td_single)  # SAVE NPZ
    save_ground_truth_final_mask(td_single, D.ESCAPE_R, dirs["td_single"] / "gt_final_mask.png", scale=64)  # GT MASK

    enc, dec, losses = train_autoencoder_reconstruction_only(  # TRAIN PURE AE
        td_single.X,  # ALL SNAPSHOTS
        latent_dim=D.LATENT_DIM,  # LATENT
        epochs=int(getattr(D, "AE_ONLY_EPOCHS", D.AE_EPOCHS)),  # EPOCHS
        batch_size=int(getattr(D, "AE_ONLY_BATCH_SIZE", D.AE_BATCH_SIZE)),  # BATCH
        lr=float(getattr(D, "AE_ONLY_LR", D.AE_LR)),  # LR
        device=device,  # DEVICE
    )

    save_loss_curve(losses, dirs["res_single"] / "loss_curve.png", "AE Only Single Matrix Loss")  # LOSS
    Z_rec = reconstruct_final_snapshot_ae_only(td_single, enc, dec, device)  # FINAL RECON
    _save_final_mask_95(Z_rec, D.ESCAPE_R, dirs["res_single"] / "ae_recon_final_mask.png", scale=64)  # MASK
    save_final_snapshot_image(Z_rec, escape_r=D.ESCAPE_R, out_png=dirs["res_single"] / "ae_recon_final_snapshot_mag.png", mode="mag")  # MAG

    m = autoencoder_reconstruction_metrics(enc, dec, td_single.X, device)  # METRICS
    _print_metrics("AE ONLY SINGLE", m)  # PRINT
    _write_metrics(dirs["res_single"] / "metrics.txt", m)  # SAVE

    # -------------------------- MULTIPLE MATRICES ---------------------------
    print("\n---------- AE ONLY / MULTIPLE MATRICES ----------\n")  # HEADER
    train_idx, test_idx, td_train_list, td_test_list = _build_multi_train_test()  # BUILD

    save_ground_truth_final_mask(td_train_list[0], D.ESCAPE_R, dirs["td_multi"] / "train_example_gt_final_mask.png", scale=64)  # TRAIN GT
    save_training_npz(dirs["td_multi"] / "training_train_example.npz", td_train_list[0])  # SAVE EXAMPLE

    enc_m, dec_m, losses_m = train_autoencoder_reconstruction_only(  # TRAIN MULTI AE
        [td.X for td in td_train_list],  # ALL TRAIN SNAPSHOTS
        latent_dim=D.LATENT_DIM,  # LATENT
        epochs=int(getattr(D, "AE_ONLY_EPOCHS", D.AE_EPOCHS)),  # EPOCHS
        batch_size=int(getattr(D, "AE_ONLY_BATCH_SIZE", D.AE_BATCH_SIZE)),  # BATCH
        lr=float(getattr(D, "AE_ONLY_LR", D.AE_LR)),  # LR
        device=device,  # DEVICE
    )

    save_loss_curve(losses_m, dirs["res_multi"] / "loss_curve.png", "AE Only Multiple Matrices Loss")  # LOSS

    metric_list = []  # STORE TEST METRICS
    for k, td_test in enumerate(td_test_list):  # TEST LOOP
        mm = autoencoder_reconstruction_metrics(enc_m, dec_m, td_test.X, device)  # METRICS
        metric_list.append(mm)  # STORE
        _print_metrics(f"AE ONLY TEST MATRIX {int(test_idx[k])}", mm)  # PRINT

        if k == 0:  # SAVE FIRST TEST VISUAL
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, dirs["res_multi"] / "test_gt_final_mask.png", scale=64)  # GT
            Z_test = reconstruct_final_snapshot_ae_only(td_test, enc_m, dec_m, device)  # RECON
            _save_final_mask_95(Z_test, D.ESCAPE_R, dirs["res_multi"] / "test_ae_recon_final_mask.png", scale=64)  # MASK
            save_final_snapshot_image(Z_test, escape_r=D.ESCAPE_R, out_png=dirs["res_multi"] / "test_ae_recon_final_snapshot_mag.png", mode="mag")  # MAG

    mean_m = _mean_metrics(metric_list)  # MEAN
    _print_metrics("AE ONLY MEAN TEST", mean_m)  # PRINT
    _write_metrics(dirs["res_multi"] / "mean_test_metrics.txt", mean_m)  # SAVE


# ============================== DMD ONLY RUN =================================
def run_dmd_only_baseline(device: torch.device) -> None:  # RUN RAW DMD ONLY
    print("\n================ DMD ONLY BASELINE ================\n")  # HEADER

    dirs = _phase_dirs(D.DMD_ONLY_OUT_DIR, "DMD-only")  # DIRS

    # --------------------------- SINGLE MATRIX ------------------------------
    print("\n---------- DMD ONLY / SINGLE MATRIX ----------\n")  # HEADER
    td_single = _build_single_td()  # BUILD DATA

    save_ground_truth_escape_iters(td_single, D.ESCAPE_R, dirs["td_single"] / "gt_escape_iters.png")  # GT ESC
    save_training_npz(dirs["td_single"] / "training_single_matrix.npz", td_single)  # SAVE NPZ
    save_ground_truth_final_mask(td_single, D.ESCAPE_R, dirs["td_single"] / "gt_final_mask.png", scale=64)  # GT MASK

    dmd = fit_full_state_dmd_streamed(td_single.X1, td_single.X2, device=device)  # FIT RAW DMD

    A = dmd.A.detach().cpu().numpy()  # MATRIX
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))  # SPECTRAL RADIUS
    print("DMD ONLY SINGLE SPECTRAL RADIUS:", rho)  # PRINT

    Z_pred = reconstruct_final_snapshot_dmd_only(td_single, dmd, device, D.ESCAPE_R)  # PRED FINAL
    _save_final_mask_95(Z_pred, D.ESCAPE_R, dirs["res_single"] / "dmd_only_pred_final_mask.png", scale=64)  # MASK
    save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res_single"] / "dmd_only_pred_final_snapshot_mag.png", mode="mag")  # MAG

    m = dmd_only_one_step_metrics(dmd, td_single.X1, td_single.X2, device)  # METRICS
    m["dmd_only_spectral_radius"] = rho  # ADD RHO
    _print_metrics("DMD ONLY SINGLE", m)  # PRINT
    _write_metrics(dirs["res_single"] / "metrics.txt", m)  # SAVE

    # -------------------------- MULTIPLE MATRICES ---------------------------
    print("\n---------- DMD ONLY / MULTIPLE MATRICES ----------\n")  # HEADER
    train_idx, test_idx, td_train_list, td_test_list = _build_multi_train_test()  # BUILD

    save_ground_truth_final_mask(td_train_list[0], D.ESCAPE_R, dirs["td_multi"] / "train_example_gt_final_mask.png", scale=64)  # TRAIN GT
    save_training_npz(dirs["td_multi"] / "training_train_example.npz", td_train_list[0])  # SAVE EXAMPLE

    dmd_m = fit_full_state_dmd_streamed(  # FIT SHARED RAW DMD
        [td.X1 for td in td_train_list],  # LEFT BLOCKS
        [td.X2 for td in td_train_list],  # RIGHT BLOCKS
        device=device,  # DEVICE
    )

    A_m = dmd_m.A.detach().cpu().numpy()  # MATRIX
    rho_m = float(np.max(np.abs(np.linalg.eigvals(A_m))))  # SPECTRAL RADIUS
    print("DMD ONLY MULTI SPECTRAL RADIUS:", rho_m)  # PRINT

    metric_list = []  # STORE TEST METRICS
    for k, td_test in enumerate(td_test_list):  # TEST LOOP
        mm = dmd_only_one_step_metrics(dmd_m, td_test.X1, td_test.X2, device)  # METRICS
        mm["dmd_only_spectral_radius"] = rho_m  # ADD RHO
        metric_list.append(mm)  # STORE
        _print_metrics(f"DMD ONLY TEST MATRIX {int(test_idx[k])}", mm)  # PRINT

        if k == 0:  # SAVE FIRST TEST VISUAL
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, dirs["res_multi"] / "test_gt_final_mask.png", scale=64)  # GT
            Z_test = reconstruct_final_snapshot_dmd_only(td_test, dmd_m, device, D.ESCAPE_R)  # PRED
            _save_final_mask_95(Z_test, D.ESCAPE_R, dirs["res_multi"] / "test_dmd_only_pred_final_mask.png", scale=64)  # MASK
            save_final_snapshot_image(Z_test, escape_r=D.ESCAPE_R, out_png=dirs["res_multi"] / "test_dmd_only_pred_final_snapshot_mag.png", mode="mag")  # MAG

    mean_m = _mean_metrics(metric_list)  # MEAN
    _print_metrics("DMD ONLY MEAN TEST", mean_m)  # PRINT
    _write_metrics(dirs["res_multi"] / "mean_test_metrics.txt", mean_m)  # SAVE


# ============================ PUBLIC ENTRYPOINT ==============================
def run_optional_baselines(device: torch.device) -> None:  # CALLED FROM MAIN
    if bool(getattr(D, "RUN_AE_ONLY_BASELINE", False)):  # AE ONLY FLAG
        run_ae_only_baseline(device)  # RUN AE ONLY

    if bool(getattr(D, "RUN_DMD_ONLY_BASELINE", False)):  # DMD ONLY FLAG
        run_dmd_only_baseline(device)  # RUN DMD ONLY