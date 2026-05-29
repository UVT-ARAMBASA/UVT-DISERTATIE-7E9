from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================ IMPORTS ==================================
import os  # OS
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH

import defines as D  # DEFINES
from utils import save_model  # HELPERS
from data_loader import load_all_A_matrices, split_explicit_matrix_indices  # DATA SPLIT
from prepare_training_data import build_matrix_c_grid_training_data_many_matrices  # DATA
from train_autoencoder import train_autoencoder  # AE TRAINING
from eval_matrix_dmd_ae import (  # EVAL HELPERS
    save_loss_curve,
    autoencoder_reconstruction_metrics,
    dmd_one_step_metrics,
    save_ground_truth_final_mask,
    save_predicted_final_mask,
)
from experiment_common import (  # COMMON HELPERS
    pick_device,
    fit_streamed_dmd_from_td_list,
    print_metric_block,
    mean_metric_dict,
)


# ============================= MULTI MATRIX RUN =============================
def run_multi_matrix(device: torch.device | None = None) -> None:  # RUN MULTI EXPERIMENT
    if device is None:  # AUTO DEVICE
        device = pick_device()  # PICK DEVICE

    print("DEVICE:", device)  # PRINT
    print("CWD:", os.getcwd())  # PRINT

    Path(D.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)  # MAKE DIR
    Path("out/training-data/multi-matrix").mkdir(parents=True, exist_ok=True)  # MAKE DIR
    Path("out/result/multi-matrix").mkdir(parents=True, exist_ok=True)  # MAKE DIR

    print("\n================ MULTI MATRIX 40 / 8 ================\n")  # HEADER

    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)  # LOAD ALL
    total_matrices = int(A_all.shape[0])  # COUNT
    print("TOTAL MATRICES:", total_matrices)  # LOG

    train_idx, test_idx = split_explicit_matrix_indices(  # EXPLICIT SPLIT
        total_count=total_matrices,
        train_count=D.MULTI_MATRIX_TRAIN_COUNT,
        test_count=D.MULTI_MATRIX_TEST_COUNT,
        seed=D.MULTI_MATRIX_SPLIT_SEED,
    )

    print("TRAIN IDX:", train_idx.tolist())  # LOG
    print("TEST IDX :", test_idx.tolist())  # LOG

    td_train_list = build_matrix_c_grid_training_data_many_matrices(  # BUILD TRAIN DATA
        data_dir=D.A_DATA_DIR,
        source=D.MULTI_MATRIX_SOURCE,
        indices=train_idx,
        c_re_min=D.C_RE_MIN,
        c_re_max=D.C_RE_MAX,
        c_im_min=D.C_IM_MIN,
        c_im_max=D.C_IM_MAX,
        c_re_n=D.MULTI_MATRIX_C_RE_N,
        c_im_n=D.MULTI_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS,
        escape_r=D.ESCAPE_R,
    )

    save_ground_truth_final_mask(  # SAVE TRAIN EXAMPLE
        td_train_list[0],
        D.ESCAPE_R,
        "out/training-data/multi-matrix/train_example_gt_final_mask.png",
        scale=64,
    )

    for td in td_train_list:  # FREE BIG GRIDS
        td.X_grid = None  # NOT NEEDED FOR TRAINING

    enc_multi, dec_multi, losses_multi = train_autoencoder(  # TRAIN AE ON TRAIN MATRICES
        [td.X1 for td in td_train_list],
        [td.X2 for td in td_train_list],
        latent_dim=D.LATENT_DIM,
        epochs=D.AE_EPOCHS,
        batch_size=D.AE_BATCH_SIZE,
        lr=D.AE_LR,
        device=device,
    )

    save_loss_curve(losses_multi, "out/result/multi-matrix/loss_curve.png", "40-Matrix AE Loss")  # LOSS
    save_model(enc_multi, os.path.join(D.CHECKPOINT_DIR, "encoder_multi_matrix.pth"))  # SAVE ENC
    save_model(dec_multi, os.path.join(D.CHECKPOINT_DIR, "decoder_multi_matrix.pth"))  # SAVE DEC

    dmd_multi = fit_streamed_dmd_from_td_list(enc_multi, td_train_list, device)  # FIT SHARED MULTI DMD

    A_multi = dmd_multi.A.detach().cpu().numpy()  # DMD MATRIX
    rho_multi = float(np.max(np.abs(np.linalg.eigvals(A_multi))))  # SPECTRAL RADIUS
    print("MULTI DMD SPECTRAL RADIUS:", rho_multi)  # PRINT
    print("MULTI-MATRIX DMD FIT DONE")  # PRINT

    td_test_list = build_matrix_c_grid_training_data_many_matrices(  # BUILD TEST DATA
        data_dir=D.A_DATA_DIR,
        source=D.MULTI_MATRIX_SOURCE,
        indices=test_idx,
        c_re_min=D.C_RE_MIN,
        c_re_max=D.C_RE_MAX,
        c_im_min=D.C_IM_MIN,
        c_im_max=D.C_IM_MAX,
        c_re_n=D.MULTI_MATRIX_C_RE_N,
        c_im_n=D.MULTI_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS,
        escape_r=D.ESCAPE_R,
    )

    ae_test_metrics_all = []  # STORE AE METRICS
    dmd_test_metrics_all = []  # STORE DMD METRICS

    for k, td_test in enumerate(td_test_list):  # LOOP TEST MATRICES
        ae_m = autoencoder_reconstruction_metrics(enc_multi, dec_multi, td_test.X, device)  # AE METRICS
        dmd_m = dmd_one_step_metrics(enc_multi, dec_multi, dmd_multi, td_test.X1, td_test.X2, device)  # DMD METRICS

        ae_test_metrics_all.append(ae_m)  # STORE
        dmd_test_metrics_all.append(dmd_m)  # STORE

        print_metric_block(f"TEST MATRIX {int(test_idx[k])} AE", ae_m)  # PRINT
        print_metric_block(f"TEST MATRIX {int(test_idx[k])} DMD", dmd_m)  # PRINT

        if k == 0:  # SAVE ONE VISUAL EXAMPLE
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, "out/result/multi-matrix/test_gt_final_mask.png", scale=64)  # GT
            save_predicted_final_mask(  # PRED
                td_test,
                enc_multi,
                dec_multi,
                dmd_multi,
                device,
                D.ESCAPE_R,
                "out/result/multi-matrix/test_pred_final_mask.png",
                scale=64,
            )

    print_metric_block("MEAN TEST AE", mean_metric_dict(ae_test_metrics_all))  # MEAN AE
    print_metric_block("MEAN TEST DMD", mean_metric_dict(dmd_test_metrics_all))  # MEAN DMD


if __name__ == "__main__":  # DIRECT RUN
    run_multi_matrix()  # RUN