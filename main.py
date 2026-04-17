from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================= IMPORTS ==================================
import os  # OS
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH

import defines as D  # DEFINES
from utils import save_model, to_tensor  # YOUR UTILS

from data_loader import load_all_A_matrices, split_explicit_matrix_indices  # MATRIX SPLIT
from prepare_training_data import (  # DATA
    build_mandelbrot_training_data,
    build_matrix_c_grid_training_data,
    build_matrix_c_grid_training_data_many_matrices,
    save_training_npz,
)
from train_autoencoder import train_autoencoder  # TRAIN AE
from apply_dmd import fit_dmd_on_arrays, fit_dmd_from_latent_covariances  # DMD FIT
from eval_matrix_dmd_ae import (  # EVAL
    save_loss_curve,
    autoencoder_reconstruction_metrics,
    dmd_one_step_metrics,
    save_ground_truth_final_mask,
    save_predicted_final_mask,
    save_ground_truth_escape_iters,
)
# ============================== DEVICE PICKER ================================
def pick_device() -> torch.device:  # SELECT DEVICE
    if D.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():  # CUDA AVAILABLE
        return torch.device("cuda")  # GPU
    return torch.device("cpu")  # CPU


#=====HELPERS======

def fit_streamed_dmd_from_td_list(  # STREAMED LATENT DMD
    enc,  # ENCODER
    td_list: list,  # LIST OF TRAINING DATA
    device: torch.device,  # DEVICE
) :
    latent_dim = int(D.LATENT_DIM)  # LATENT DIM
    G = np.zeros((latent_dim, latent_dim), dtype=np.float64)  # Z1^T Z1
    H = np.zeros((latent_dim, latent_dim), dtype=np.float64)  # Z1^T Z2

    with torch.no_grad():  # NO GRAD
        for td in td_list:  # LOOP TD
            Z1 = enc(to_tensor(td.X1, device)).detach().cpu().numpy().astype(np.float64)  # Z1
            Z2 = enc(to_tensor(td.X2, device)).detach().cpu().numpy().astype(np.float64)  # Z2
            G += Z1.T @ Z1  # ACCUM
            H += Z1.T @ Z2  # ACCUM

    return fit_dmd_from_latent_covariances(G, H, device=device)  # BUILD DMD

def print_metric_block(name: str, metrics: dict) -> None:  # PRETTY PRINT
    print(f"[{name}]")  # HEADER
    for k, v in metrics.items():  # LOOP
        print(f"  {k}: {v:.8e}")  # PRINT

def mean_metric_dict(metric_list: list[dict]) -> dict:  # MEAN OVER LIST
    keys = metric_list[0].keys()  # KEYS
    out = {}  # STORE
    for k in keys:  # LOOP KEYS
        out[k] = float(np.mean([m[k] for m in metric_list]))  # MEAN
    return out  # RETURN


# =================================== MAIN ===================================
def main() -> None:  # ENTRYPOINT
    device = pick_device()  # DEVICE
    print("DEVICE:", device)  # PRINT
    print("CWD:", os.getcwd())  # SHOW WORKING DIR

    Path(D.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)  # MAKE DIR
    Path(D.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)  # MAKE DIR

    Path("out/training-data/single-matrix").mkdir(parents=True, exist_ok=True)  # MAKE DIR
    Path("out/training-data/multi-matrix").mkdir(parents=True, exist_ok=True)  # MAKE DIR
    Path("out/result/single-matrix").mkdir(parents=True, exist_ok=True)  # MAKE DIR
    Path("out/result/multi-matrix").mkdir(parents=True, exist_ok=True)  # MAKE DIR

    # ===================== PART 1: SINGLE MATRIX SANITY CHECK =====================
    print("\n================ SINGLE MATRIX CHECK ================\n")  # HEADER

    td_single = build_matrix_c_grid_training_data(  # BUILD ONE MATRIX DATA
        data_dir=D.A_DATA_DIR,  # DATA DIR
        source=D.SINGLE_MATRIX_SOURCE,  # SOURCE
        index=D.SINGLE_MATRIX_INDEX,  # MATRIX INDEX
        c_re_min=D.C_RE_MIN,  # RE MIN
        c_re_max=D.C_RE_MAX,  # RE MAX
        c_im_min=D.C_IM_MIN,  # IM MIN
        c_im_max=D.C_IM_MAX,  # IM MAX
        c_re_n=D.SINGLE_MATRIX_C_RE_N,  # SMALL GRID
        c_im_n=D.SINGLE_MATRIX_C_IM_N,  # SMALL GRID
        max_iters=D.TRAIN_MAX_ITERS,  # ITERS
        escape_r=D.ESCAPE_R,  # ESCAPE
    )

    save_ground_truth_escape_iters(  # SAVE TRUE FRACTAL
        td_single,  # TRAIN DATA
        D.ESCAPE_R,  # ESCAPE RADIUS
        "out/training-data/single-matrix/gt_escape_iters.png",  # OUTPUT PATH
    )  # END

    save_training_npz(  # SAVE NPZ
        "out/training-data/single-matrix/training_single_matrix.npz",  # PATH
        td_single,  # DATA
    )
    save_ground_truth_final_mask(  # SAVE GT
        td_single,  # DATA
        D.ESCAPE_R,  # ESCAPE
        "out/training-data/single-matrix/gt_final_mask.png",  # PATH
        scale=64,  # UPSCALE
    )

    enc_single, dec_single, losses_single = train_autoencoder(  # TRAIN AE
        td_single.X,  # DATA
        latent_dim=D.LATENT_DIM,  # LATENT
        epochs=D.AE_EPOCHS,  # EPOCHS
        batch_size=D.AE_BATCH_SIZE,  # BATCH
        lr=D.AE_LR,  # LR
        device=device,  # DEVICE
    )

    save_loss_curve(  # LOSS PLOT
        losses_single,  # LOSSES
        "out/result/single-matrix/loss_curve.png",  # PATH
        "Single Matrix AE Loss",  # TITLE
    )
    save_model(enc_single, os.path.join(D.CHECKPOINT_DIR, "encoder_single_matrix.pth"))  # SAVE ENC
    save_model(dec_single, os.path.join(D.CHECKPOINT_DIR, "decoder_single_matrix.pth"))  # SAVE DEC

    with torch.no_grad():  # NO GRAD
        Z1_single = enc_single(to_tensor(td_single.X1, device)).detach().cpu().numpy()  # ENCODE X1
        Z2_single = enc_single(to_tensor(td_single.X2, device)).detach().cpu().numpy()  # ENCODE X2

    dmd_single = fit_dmd_on_arrays(Z1_single, Z2_single, device=device)  # FIT DMD
    save_predicted_final_mask(  # PRED PLOT
        td_single,  # DATA
        enc_single,  # ENCODER
        dec_single,  # DECODER
        dmd_single,  # DMD
        device,  # DEVICE
        D.ESCAPE_R,  # ESCAPE
        "out/result/single-matrix/pred_final_mask.png",  # PATH
        scale=64,  # UPSCALE
    )

    ae_single_metrics = autoencoder_reconstruction_metrics(enc_single, dec_single, td_single.X, device)  # AE METRICS
    dmd_single_metrics = dmd_one_step_metrics(enc_single, dec_single, dmd_single, td_single.X1, td_single.X2, device)  # DMD METRICS
    print_metric_block("SINGLE MATRIX AE", ae_single_metrics)  # PRINT
    print_metric_block("SINGLE MATRIX DMD", dmd_single_metrics)  # PRINT

    # ===================== PART 2: MULTI MATRIX 40 / 8 ===========================
    print("\n================ MULTI MATRIX 40 / 8 ================\n")  # HEADER

    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)  # LOAD ALL
    total_matrices = int(A_all.shape[0])  # COUNT
    print("TOTAL MATRICES:", total_matrices)  # LOG

    train_idx, test_idx = split_explicit_matrix_indices(  # EXPLICIT SPLIT
        total_count=total_matrices,  # TOTAL
        train_count=D.MULTI_MATRIX_TRAIN_COUNT,  # 40
        test_count=D.MULTI_MATRIX_TEST_COUNT,  # 8
        seed=D.MULTI_MATRIX_SPLIT_SEED,  # REPRO
    )

    print("TRAIN IDX:", train_idx.tolist())  # LOG
    print("TEST IDX :", test_idx.tolist())  # LOG

    td_train_list = build_matrix_c_grid_training_data_many_matrices(  # BUILD TRAIN LIST
        data_dir=D.A_DATA_DIR,  # DATA DIR
        source=D.MULTI_MATRIX_SOURCE,  # SOURCE
        indices=train_idx,  # TRAIN MATRICES
        c_re_min=D.C_RE_MIN,  # RE MIN
        c_re_max=D.C_RE_MAX,  # RE MAX
        c_im_min=D.C_IM_MIN,  # IM MIN
        c_im_max=D.C_IM_MAX,  # IM MAX
        c_re_n=D.MULTI_MATRIX_C_RE_N,  # SMALL GRID
        c_im_n=D.MULTI_MATRIX_C_IM_N,  # SMALL GRID
        max_iters=D.TRAIN_MAX_ITERS,  # ITERS
        escape_r=D.ESCAPE_R,  # ESCAPE
    )

    save_ground_truth_final_mask(  # SAVE TRAIN EXAMPLE
        td_train_list[0],  # FIRST TRAIN EXAMPLE
        D.ESCAPE_R,  # ESCAPE
        "out/training-data/multi-matrix/train_example_gt_final_mask.png",  # PATH
        scale=64,  # UPSCALE
    )

    for td in td_train_list:  # FREE BIG GRIDS
        td.X_grid = None  # NOT NEEDED FOR TRAINING

    enc_multi, dec_multi, losses_multi = train_autoencoder(  # TRAIN AE ON 40 MATRICES
        [td.X for td in td_train_list],  # STREAMED DATA
        latent_dim=D.LATENT_DIM,  # LATENT
        epochs=D.AE_EPOCHS,  # EPOCHS
        batch_size=D.AE_BATCH_SIZE,  # BATCH
        lr=D.AE_LR,  # LR
        device=device,  # DEVICE
    )

    save_loss_curve(  # LOSS PLOT
        losses_multi,  # LOSSES
        "out/result/multi-matrix/loss_curve.png",  # PATH
        "40-Matrix AE Loss",  # TITLE
    )
    save_model(enc_multi, os.path.join(D.CHECKPOINT_DIR, "encoder_multi_matrix.pth"))  # SAVE ENC
    save_model(dec_multi, os.path.join(D.CHECKPOINT_DIR, "decoder_multi_matrix.pth"))  # SAVE DEC

    dmd_multi = fit_streamed_dmd_from_td_list(enc_multi, td_train_list, device)  # STREAMED DMD
    print("MULTI-MATRIX DMD FIT DONE")  # LOG

    td_test_list = build_matrix_c_grid_training_data_many_matrices(  # BUILD TEST LIST
        data_dir=D.A_DATA_DIR,  # DATA DIR
        source=D.MULTI_MATRIX_SOURCE,  # SOURCE
        indices=test_idx,  # TEST MATRICES
        c_re_min=D.C_RE_MIN,  # RE MIN
        c_re_max=D.C_RE_MAX,  # RE MAX
        c_im_min=D.C_IM_MIN,  # IM MIN
        c_im_max=D.C_IM_MAX,  # IM MAX
        c_re_n=D.MULTI_MATRIX_C_RE_N,  # SMALL GRID
        c_im_n=D.MULTI_MATRIX_C_IM_N,  # SMALL GRID
        max_iters=D.TRAIN_MAX_ITERS,  # ITERS
        escape_r=D.ESCAPE_R,  # ESCAPE
    )

    ae_test_metrics_all = []  # STORE
    dmd_test_metrics_all = []  # STORE

    for k, td_test in enumerate(td_test_list):  # LOOP TEST MATRICES
        ae_m = autoencoder_reconstruction_metrics(enc_multi, dec_multi, td_test.X, device)  # AE METRICS
        dmd_m = dmd_one_step_metrics(enc_multi, dec_multi, dmd_multi, td_test.X1, td_test.X2, device)  # DMD METRICS

        ae_test_metrics_all.append(ae_m)  # STORE
        dmd_test_metrics_all.append(dmd_m)  # STORE

        print_metric_block(f"TEST MATRIX {int(test_idx[k])} AE", ae_m)  # PRINT
        print_metric_block(f"TEST MATRIX {int(test_idx[k])} DMD", dmd_m)  # PRINT

        if k == 0:  # SAVE ONE VISUAL EXAMPLE
            save_ground_truth_final_mask(  # SAVE TEST GT
                td_test,  # DATA
                D.ESCAPE_R,  # ESCAPE
                "out/result/multi-matrix/test_gt_final_mask.png",  # PATH
                scale=64,  # UPSCALE
            )
            save_predicted_final_mask(  # SAVE TEST PRED
                td_test,  # DATA
                enc_multi,  # ENCODER
                dec_multi,  # DECODER
                dmd_multi,  # DMD
                device,  # DEVICE
                D.ESCAPE_R,  # ESCAPE
                "out/result/multi-matrix/test_pred_final_mask.png",  # PATH
                scale=64,  # UPSCALE
            )

    print_metric_block("MEAN TEST AE", mean_metric_dict(ae_test_metrics_all))  # PRINT MEAN
    print_metric_block("MEAN TEST DMD", mean_metric_dict(dmd_test_metrics_all))  # PRINT MEAN

if __name__ == "__main__":  # MAIN GUARD
    main()  # RUN