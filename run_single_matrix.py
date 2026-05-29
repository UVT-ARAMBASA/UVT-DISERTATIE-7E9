from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================ IMPORTS ==================================
import os  # OS
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH

import defines as D  # DEFINES
from utils import save_model, to_tensor  # HELPERS
from prepare_training_data import build_matrix_c_grid_training_data, save_training_npz  # DATA
from train_autoencoder import train_autoencoder  # AE TRAINING
from apply_dmd import fit_dmd_on_arrays  # DMD FIT
from eval_matrix_dmd_ae import (  # EVAL HELPERS
    save_loss_curve,
    autoencoder_reconstruction_metrics,
    dmd_one_step_metrics,
    save_ground_truth_final_mask,
    save_predicted_final_mask,
    save_ground_truth_escape_iters,
)
from mandelbrot_reconstruct import reconstruct_final_snapshot, save_final_snapshot_image  # RECON
from experiment_common import pick_device, print_metric_block, debug_final_state_stats  # COMMON


# ============================= SINGLE MATRIX RUN ============================
def run_single_matrix(device: torch.device | None = None) -> None:  # RUN SINGLE EXPERIMENT
    if device is None:  # AUTO DEVICE
        device = pick_device()  # PICK DEVICE

    print("DEVICE:", device)  # PRINT
    print("CWD:", os.getcwd())  # PRINT

    Path(D.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)  # MAKE DIR
    Path("out/training-data/single-matrix").mkdir(parents=True, exist_ok=True)  # MAKE DIR
    Path("out/result/single-matrix").mkdir(parents=True, exist_ok=True)  # MAKE DIR

    print("\n================ SINGLE MATRIX CHECK ================\n")  # HEADER

    td_single = build_matrix_c_grid_training_data(  # BUILD ONE MATRIX DATA
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

    save_ground_truth_escape_iters(td_single, D.ESCAPE_R, "out/training-data/single-matrix/gt_escape_iters.png")  # GT ESCAPE
    save_training_npz("out/training-data/single-matrix/training_single_matrix.npz", td_single)  # SAVE DATA
    save_ground_truth_final_mask(td_single, D.ESCAPE_R, "out/training-data/single-matrix/gt_final_mask.png", scale=64)  # GT MASK

    enc_single, dec_single, losses_single = train_autoencoder(  # TRAIN AE
        td_single.X1,  # LEFT
        td_single.X2,  # RIGHT
        latent_dim=D.LATENT_DIM,  # LATENT
        epochs=D.AE_EPOCHS,  # EPOCHS
        batch_size=D.AE_BATCH_SIZE,  # BATCH
        lr=D.AE_LR,  # LR
        device=device,  # DEVICE
    )

    save_loss_curve(losses_single, "out/result/single-matrix/loss_curve.png", "Single Matrix AE Loss")  # LOSS
    save_model(enc_single, os.path.join(D.CHECKPOINT_DIR, "encoder_single_matrix.pth"))  # SAVE ENC
    save_model(dec_single, os.path.join(D.CHECKPOINT_DIR, "decoder_single_matrix.pth"))  # SAVE DEC

    with torch.no_grad():  # NO GRAD
        Z1_single = enc_single(to_tensor(td_single.X1, device)).detach().cpu().numpy()  # ENCODE X1
        Z2_single = enc_single(to_tensor(td_single.X2, device)).detach().cpu().numpy()  # ENCODE X2

    dmd_single = fit_dmd_on_arrays(Z1_single, Z2_single, device=device)  # FIT DMD

    A_single = dmd_single.A.detach().cpu().numpy()  # DMD MATRIX
    rho_single = float(np.max(np.abs(np.linalg.eigvals(A_single))))  # SPECTRAL RADIUS
    print("SINGLE DMD SPECTRAL RADIUS:", rho_single)  # PRINT

    save_predicted_final_mask(  # SAVE PRED MASK
        td_single,
        enc_single,
        dec_single,
        dmd_single,
        device,
        D.ESCAPE_R,
        "out/result/single-matrix/pred_final_mask.png",
        scale=64,
    )

    feat_dim = int(td_single.X_grid.shape[-1])  # FEATURE DIM
    d = (feat_dim - 2) // 2  # STATE DIM
    T = int(td_single.X_grid.shape[0])  # STEPS
    C_single = td_single.X_grid[0, :, :, 2 * d:2 * d + 2].reshape(-1, 2).astype(np.float32)  # C GRID

    Z_single_final = reconstruct_final_snapshot(  # FINAL SNAPSHOT
        encoder=enc_single,
        decoder=dec_single,
        dmd=dmd_single,
        C=C_single,
        grid_n=D.SINGLE_MATRIX_C_RE_N,
        steps=T,
        escape_r=D.ESCAPE_R,
        device=device,
        batch_size=50000,
        state_dim=d,
        feat_dim=feat_dim,
    )

    debug_final_state_stats("SINGLE MATRIX", Z_single_final, D.ESCAPE_R)  # DEBUG
    save_final_snapshot_image(  # SAVE FINAL IMAGE
        Z_single_final,
        escape_r=D.ESCAPE_R,
        out_png="out/result/single-matrix/pred_final_snapshot_mag.png",
        mode="mask",
    )

    ae_single_metrics = autoencoder_reconstruction_metrics(enc_single, dec_single, td_single.X, device)  # AE METRICS
    dmd_single_metrics = dmd_one_step_metrics(enc_single, dec_single, dmd_single, td_single.X1, td_single.X2, device)  # DMD METRICS
    print_metric_block("SINGLE MATRIX AE", ae_single_metrics)  # PRINT
    print_metric_block("SINGLE MATRIX DMD", dmd_single_metrics)  # PRINT


if __name__ == "__main__":  # DIRECT RUN
    run_single_matrix()  # RUN