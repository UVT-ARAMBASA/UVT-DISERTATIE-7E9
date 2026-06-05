from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================ IMPORTS ==================================
import os  # OS
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
    save_ground_truth_escape_iters,
    reconstruct_true_final_snapshot,
    iterate_true_next_snapshot,
    predict_next_snapshot,
    next_step_prediction_metrics,
    teacher_forced_escape_iters,
)
from mandelbrot_reconstruct import save_final_snapshot_image, save_escape_image  # IMAGE SAVERS
from experiment_common import (  # COMMON HELPERS
    pick_device,
    make_out_dirs,
    fit_streamed_dmd_from_td_list,
    print_metric_block,
    mean_metric_dict,
    write_metrics_txt,
)


# ============================= MULTI MATRIX RUN =============================
def run_multi_matrix(device: torch.device | None = None) -> None:  # RUN MULTI EXPERIMENT
    if device is None:  # AUTO DEVICE
        device = pick_device()  # PICK DEVICE

    print("DEVICE:", device)  # PRINT
    print("CWD:", os.getcwd())  # PRINT

    dirs = make_out_dirs("multi-matrix")  # out/multi-matrix/{training-data,results}

    print("\n================ MULTI MATRIX TRAIN / TEST ================\n")  # HEADER

    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)  # LOAD ALL
    total_matrices = int(A_all.shape[0])  # COUNT
    print("TOTAL MATRICES:", total_matrices)  # LOG

    train_idx, test_idx = split_explicit_matrix_indices(  # 40 TRAIN / 8 TEST
        total_count=total_matrices,
        train_count=D.MULTI_MATRIX_TRAIN_COUNT,
        test_count=D.MULTI_MATRIX_TEST_COUNT,
        seed=D.MULTI_MATRIX_SPLIT_SEED,
    )
    print("TRAIN COUNT:", int(train_idx.size), "TEST COUNT:", int(test_idx.size))  # LOG
    print("TRAIN IDX:", train_idx.tolist())  # LOG
    print("TEST IDX :", test_idx.tolist())  # LOG

    # ------------------------------ TRAIN DATA ------------------------------
    td_train_list = build_matrix_c_grid_training_data_many_matrices(  # BUILD TRAIN DATA (40)
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

    save_ground_truth_final_mask(td_train_list[0], D.ESCAPE_R, dirs["td"] / "train_example_gt_final_mask.png", scale=D.IMAGE_SCALE)  # TRAIN GT MASK
    save_ground_truth_escape_iters(td_train_list[0], D.ESCAPE_R, dirs["td"] / "train_example_gt_escape_iters.png")  # TRAIN GT FRACTAL

    for td in td_train_list:  # FREE BIG GRIDS
        td.X_grid = None  # NOT NEEDED FOR TRAINING

    # ------------------------------ TRAIN AE + DMD --------------------------
    enc, dec, losses = train_autoencoder(  # TRAIN AE ON THE 40 TRAIN MATRICES
        [td.X1 for td in td_train_list],
        [td.X2 for td in td_train_list],
        latent_dim=D.LATENT_DIM,
        epochs=D.AE_EPOCHS,
        batch_size=D.AE_BATCH_SIZE,
        lr=D.AE_LR,
        device=device,
    )

    save_loss_curve(losses, dirs["res"] / "loss_curve.png", "Multi-Matrix AE Loss")  # LOSS
    save_model(enc, os.path.join(D.CHECKPOINT_DIR, "encoder_multi_matrix.pth"))  # SAVE ENC
    save_model(dec, os.path.join(D.CHECKPOINT_DIR, "decoder_multi_matrix.pth"))  # SAVE DEC

    dmd = fit_streamed_dmd_from_td_list(enc, td_train_list, device)  # FIT ONE SHARED LATENT DMD
    rho = float(np.max(np.abs(np.linalg.eigvals(dmd.A.detach().cpu().numpy()))))  # SPECTRAL RADIUS
    print("MULTI DMD SPECTRAL RADIUS:", rho)  # PRINT

    # ------------------------------- TEST DATA ------------------------------
    td_test_list = build_matrix_c_grid_training_data_many_matrices(  # BUILD TEST DATA (8 HELD-OUT)
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

    k = int(D.PREDICT_EXTRA_STEPS)  # STEPS AHEAD
    ae_metrics_all = []  # STORE AE METRICS
    dmd_metrics_all = []  # STORE DMD ONE-STEP METRICS
    pred_metrics_all = []  # STORE NEXT-STEP PREDICTION METRICS

    for j, td_test in enumerate(td_test_list):  # LOOP HELD-OUT TEST MATRICES
        A_test = A_all[int(test_idx[j])]  # TRUE MATRIX FOR THIS TEST CASE

        ae_m = autoencoder_reconstruction_metrics(enc, dec, td_test.X, device)  # RECON METRICS
        dmd_m = dmd_one_step_metrics(enc, dec, dmd, td_test.X1, td_test.X2, device)  # ONE-STEP METRICS

        Z_pred = predict_next_snapshot(td_test, enc, dec, dmd, device, steps=k, escape_r=D.ESCAPE_R)  # MODEL x_{T+k}
        Z_true_next = iterate_true_next_snapshot(td_test, A_test, steps=k, escape_r=D.ESCAPE_R)  # TRUE x_{T+k}
        pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)  # PRED vs TRUE NEXT

        ae_metrics_all.append(ae_m)  # STORE
        dmd_metrics_all.append(dmd_m)  # STORE
        pred_metrics_all.append(pred_m)  # STORE

        print_metric_block(f"TEST MATRIX {int(test_idx[j])} AE (RECON)", ae_m)  # PRINT
        print_metric_block(f"TEST MATRIX {int(test_idx[j])} DMD (ONE-STEP)", dmd_m)  # PRINT
        print_metric_block(f"TEST MATRIX {int(test_idx[j])} PREDICT (+{k})", pred_m)  # PRINT

        if j == 0:  # SAVE ONE VISUAL EXAMPLE (GT + RECON + PREDICTION + TRUE NEXT)
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_final_mask.png", scale=D.IMAGE_SCALE)  # GT MASK
            save_ground_truth_escape_iters(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_escape_iters.png")  # GT FRACTAL

            Z_recon = reconstruct_true_final_snapshot(td_test, enc, dec, device)  # RECON xT
            save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_recon_final_mask.png", mode="mask")  # RECON MASK
            save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_recon_final_snapshot_mag.png", mode="mag")  # RECON MAG

            save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_pred_final_mask.png", mode="mask")  # PRED MASK
            save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_pred_final_snapshot_mag.png", mode="mag")  # PRED MAG
            save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_true_next_final_mask.png", mode="mask")  # TRUE NEXT MASK
            save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_true_next_final_snapshot_mag.png", mode="mag")  # TRUE NEXT MAG

            iters_pred = teacher_forced_escape_iters(td_test, enc, dec, dmd, device, escape_r=D.ESCAPE_R)  # PRED FRACTAL
            save_escape_image(iters_pred, max_iters=int(td_test.X_grid.shape[0]), out_png=dirs["res"] / "test_pred_escape_iters.png")  # SAVE FRACTAL

    mean_ae = mean_metric_dict(ae_metrics_all)  # MEAN AE
    mean_dmd = mean_metric_dict(dmd_metrics_all)  # MEAN DMD ONE-STEP
    mean_pred = mean_metric_dict(pred_metrics_all)  # MEAN PREDICTION
    print_metric_block("MEAN TEST AE (RECON)", mean_ae)  # PRINT
    print_metric_block("MEAN TEST DMD (ONE-STEP)", mean_dmd)  # PRINT
    print_metric_block(f"MEAN TEST PREDICT (+{k})", mean_pred)  # PRINT
    write_metrics_txt(  # SAVE
        dirs["res"] / "mean_test_metrics.txt",
        {**mean_ae, **mean_dmd, **mean_pred, "predict_extra_steps": float(k), "dmd_spectral_radius": rho},
    )


if __name__ == "__main__":  # DIRECT RUN
    run_multi_matrix()  # RUN