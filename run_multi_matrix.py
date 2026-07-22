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
    autoencoder_reconstruction_metrics_alive,  # NEW: THE APPLES-TO-APPLES AE NUMBER
    dmd_one_step_metrics,
    save_ground_truth_final_mask,
    save_ground_truth_escape_iters,
    reconstruct_true_final_snapshot,
    iterate_true_next_snapshot,
    predict_next_snapshot,
    predict_rollout_from_start_ae_dmd,  # NEW: THE "SERIOUS" TEST FROM ALEX'S EMAIL (SEE BELOW)
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
        escape_r=D.DYNAMICS_CLAMP_R,  # NUMERICAL CLAMP DURING ITERATION (NOT THE CLASSIFY THRESHOLD)
        classify_r=D.ESCAPE_R,  # "ESCAPED" THRESHOLD USED ONLY TO DECIDE WHAT'S ALIVE FOR TRAINING
        filter_escaped=D.FILTER_ESCAPED_FOR_TRAINING,  # DROP ESCAPED TRAJECTORIES FROM X1/X2
    )

    for i, td in enumerate(td_train_list):  # ONE GT IMAGE PER TRAIN MATRIX (WAS: ONLY INDEX 0)
        matrix_id = int(train_idx[i])  # ACTUAL MATRIX INDEX, NOT LIST POSITION
        save_ground_truth_final_mask(
            td, D.ESCAPE_R,
            dirs["td"] / f"train_gt_final_mask_{matrix_id:02d}.png",
            scale=D.IMAGE_SCALE,
        )  # TRAIN GT MASK
        save_ground_truth_escape_iters(
            td, D.ESCAPE_R,
            dirs["td"] / f"train_gt_escape_iters_{matrix_id:02d}.png",
        )  # TRAIN GT FRACTAL

    for td in td_train_list:  # FREE BIG GRIDS
        td.X_grid = None  # NOT NEEDED FOR TRAINING

    # ------------------------------ TRAIN AE + DMD --------------------------
    enc, dec, losses, loss_components, val_losses = train_autoencoder(  # TRAIN AE ON THE 40 TRAIN MATRICES
        [td.X1 for td in td_train_list],
        [td.X2 for td in td_train_list],
        latent_dim=D.LATENT_DIM,
        epochs=D.AE_EPOCHS,
        batch_size=D.AE_BATCH_SIZE,
        lr=D.AE_LR,
        device=device,
    )

    has_val = bool(np.any(np.isfinite(val_losses))) if len(val_losses) else False  # ANYTHING TO OVERLAY?
    save_loss_curve(
        losses, dirs["res"] / "loss_curve.png", "Multi-Matrix AE Loss",
        extra_series=({"Validation": val_losses} if has_val else None),
        primary_label="Train" if has_val else None,
    )  # LOSS (+ VALIDATION IF WE HAVE ONE)
    save_loss_curve(loss_components["rec"], dirs["res"] / "loss_curve_rec.png",
                    "Multi-Matrix AE Loss -- Reconstruction Component")
    save_loss_curve(loss_components["lin"], dirs["res"] / "loss_curve_lin.png",
                    "Multi-Matrix AE Loss -- Latent Linearity Component")
    save_loss_curve(loss_components["pred"], dirs["res"] / "loss_curve_pred.png",
                    "Multi-Matrix AE Loss -- Decoded Prediction Component")
    save_model(enc, os.path.join(D.CHECKPOINT_DIR, "encoder_multi_matrix.pth"))  # SAVE ENC
    save_model(dec, os.path.join(D.CHECKPOINT_DIR, "decoder_multi_matrix.pth"))  # SAVE DEC

    dmd = fit_streamed_dmd_from_td_list(enc, td_train_list, device)  # FIT ONE SHARED LATENT DMD
    rho = float(np.max(np.abs(np.linalg.eigvals(dmd.A.detach().cpu().numpy()))))  # SPECTRAL RADIUS
    print("MULTI DMD SPECTRAL RADIUS:", rho)  # PRINT
    # ---------------------- TRAIN-SET SANITY CHECK --------------------
    ae_train_metrics_all, dmd_train_metrics_all = [], []
    for i, td in enumerate(td_train_list):
        ae_tr = autoencoder_reconstruction_metrics_alive(enc, dec, td, device)
        dmd_tr = dmd_one_step_metrics(enc, dec, dmd, td.X1, td.X2, device)
        ae_train_metrics_all.append(ae_tr)
        dmd_train_metrics_all.append(dmd_tr)
        print_metric_block(f"TRAIN MATRIX {int(train_idx[i])} AE (RECON, ALIVE-ONLY)", ae_tr)
        print_metric_block(f"TRAIN MATRIX {int(train_idx[i])} DMD (ONE-STEP)", dmd_tr)

    mean_ae_train = mean_metric_dict(ae_train_metrics_all)
    mean_dmd_train = mean_metric_dict(dmd_train_metrics_all)
    print_metric_block("MEAN TRAIN AE (RECON, ALIVE-ONLY)", mean_ae_train)
    print_metric_block("MEAN TRAIN DMD (ONE-STEP)", mean_dmd_train)
    write_metrics_txt(
        dirs["res"] / "mean_train_metrics.txt",
        {
            **{f"train_{k2}": v for k2, v in mean_ae_train.items()},
            **{f"train_{k2}": v for k2, v in mean_dmd_train.items()},
            "n_train_matrices": float(len(td_train_list)),
        },
    )
    # ------------------------------------------------------------------------
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
        escape_r=D.DYNAMICS_CLAMP_R,  # NUMERICAL CLAMP DURING ITERATION (NOT THE CLASSIFY THRESHOLD)
        classify_r=D.ESCAPE_R,  # "ESCAPED" THRESHOLD USED ONLY TO DECIDE WHAT'S ALIVE FOR TRAINING
        filter_escaped=D.FILTER_ESCAPED_FOR_TRAINING,  # DROP ESCAPED TRAJECTORIES FROM X1/X2
    )

    k = int(D.PREDICT_EXTRA_STEPS)  # STEPS AHEAD
    ae_metrics_all = []  # STORE AE METRICS (FULL GRID)
    ae_alive_metrics_all = []  # NEW: STORE AE METRICS (ALIVE-ONLY)
    dmd_metrics_all = []  # STORE DMD ONE-STEP METRICS
    pred_metrics_all = []  # STORE NEXT-STEP PREDICTION METRICS (FULL GRID)
    pred_alive_metrics_all = []  # STORE NEXT-STEP PREDICTION METRICS (ALIVE-ONLY)

    maxit = int(D.TRAIN_MAX_ITERS)  # T
    n_check = min(int(getattr(D, "PREDICT_ROLLOUT_CHECK_STEPS", 10)), max(maxit - 1, 0))  # HOW MANY STEPS TO CHECK
    rollout_final_metrics_all = []  # TEST 1 (MACRO), FULL GRID, PER TEST MATRIX
    rollout_final_alive_metrics_all = []  # TEST 1 (MACRO), ALIVE-ONLY, PER TEST MATRIX (WHERE AVAILABLE)
    rollout_step_curves_all = []  # TEST 2 (QUANTITATIVE), FULL GRID, ONE (n_check,) CURVE PER TEST MATRIX
    rollout_step_curves_alive_all = []  # TEST 2, ALIVE-ONLY, SAME (WHERE AVAILABLE)

    for j, td_test in enumerate(td_test_list):  # LOOP HELD-OUT TEST MATRICES
        A_test = A_all[int(test_idx[j])]  # TRUE MATRIX FOR THIS TEST CASE
        alive_grid = td_test.meta.get("alive_mask_grid", None)  # (H,W) BOOL -- WHERE THE MODEL IS IN-DOMAIN

        ae_m = autoencoder_reconstruction_metrics(enc, dec, td_test.X, device)  # RECON METRICS, FULL GRID (DIAGNOSTIC)
        ae_m_alive = autoencoder_reconstruction_metrics_alive(enc, dec, td_test, device)  # NEW
        dmd_m = dmd_one_step_metrics(enc, dec, dmd, td_test.X1, td_test.X2, device)  # ONE-STEP METRICS

        # NUMERICAL ITERATION CLAMP -- MATCHES WHAT BUILT td_test.X_grid.
        Z_pred = predict_next_snapshot(td_test, enc, dec, dmd, device, steps=k, escape_r=D.DYNAMICS_CLAMP_R)  # MODEL x_{T+k}
        Z_true_next = iterate_true_next_snapshot(td_test, A_test, steps=k, escape_r=D.DYNAMICS_CLAMP_R)  # TRUE x_{T+k}
        pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)  # PRED vs TRUE NEXT (FULL GRID)

        pred_m_alive = None  # DEFAULT
        if alive_grid is not None and bool(np.any(alive_grid)):  # NEW: ALIVE-ONLY VARIANT
            pred_m_alive = next_step_prediction_metrics(Z_pred[alive_grid], Z_true_next[alive_grid])

        # ---- NEW: ROLLOUT-FROM-START (SEE BLOCK COMMENT ABOVE THE LOOP) ----
        rollout = predict_rollout_from_start_ae_dmd(  # ONE ROLLOUT SERVES BOTH TESTS BELOW
            td_test, enc, dec, dmd, device, steps=maxit, escape_r=D.DYNAMICS_CLAMP_R,
        )  # rollout[s] = AE+DMD PREDICTION OF x_{s+2}, s = 0 .. maxit-2
        d_state = int((int(td_test.X_grid.shape[-1]) - 2) // 2)  # td_test.X_grid HAS +2 C CHANNELS, rollout DOES NOT

        Z_pred_final = rollout[-1]  # PREDICTED x_{maxit}
        Z_true_final = td_test.X_grid[-1][..., :2 * d_state]  # DROP TRAILING C SO SHAPES MATCH
        final_m = next_step_prediction_metrics(Z_pred_final, Z_true_final)  # TEST 1: MACRO/EYEBALL, QUANTIFIED
        rollout_final_metrics_all.append(final_m)  # STORE

        if alive_grid is not None and bool(np.any(alive_grid)):  # ALIVE-ONLY VARIANT
            final_m_alive = next_step_prediction_metrics(Z_pred_final[alive_grid], Z_true_final[alive_grid])
            rollout_final_alive_metrics_all.append(final_m_alive)  # STORE
            print_metric_block(
                f"TEST MATRIX {int(test_idx[j])} AE+DMD ROLLOUT x1 -> x{maxit} (MACRO, ALIVE-ONLY, "
                f"{int(np.count_nonzero(alive_grid))}/{alive_grid.size} px)", final_m_alive,
            )  # PRINT

        step_curve = []  # THIS TEST MATRIX'S rel_l2-vs-STEP CURVE, FULL GRID
        step_curve_alive = []  # SAME, ALIVE-ONLY
        for s in range(n_check):  # TEST 2: QUANTITATIVE, s STEPS BEYOND x1
            Z_pred_s = rollout[s]  # PREDICTED x_{s+2}
            Z_true_s = td_test.X_grid[s + 1][..., :2 * d_state]  # TRUE x_{s+2}
            m_s = next_step_prediction_metrics(Z_pred_s, Z_true_s)  # rel_l2, mse, fit
            step_curve.append(float(m_s["pred_rel_l2"]))  # COLLECT

            if alive_grid is not None and bool(np.any(alive_grid)):  # ALIVE-ONLY VARIANT
                m_s_alive = next_step_prediction_metrics(Z_pred_s[alive_grid], Z_true_s[alive_grid])
                step_curve_alive.append(float(m_s_alive["pred_rel_l2"]))  # COLLECT

        rollout_step_curves_all.append(step_curve)  # STORE THIS MATRIX'S CURVE
        if step_curve_alive:  # ONLY IF WE HAD A USABLE MASK
            rollout_step_curves_alive_all.append(step_curve_alive)  # STORE
        # ---------------------------------------------------------------------

        ae_metrics_all.append(ae_m)  # STORE
        ae_alive_metrics_all.append(ae_m_alive)  # STORE
        dmd_metrics_all.append(dmd_m)  # STORE
        pred_metrics_all.append(pred_m)  # STORE
        if pred_m_alive is not None:  # STORE IF WE HAVE ONE
            pred_alive_metrics_all.append(pred_m_alive)  # STORE

        print_metric_block(f"TEST MATRIX {int(test_idx[j])} AE (RECON, FULL GRID)", ae_m)  # PRINT
        print_metric_block(f"TEST MATRIX {int(test_idx[j])} AE (RECON, ALIVE-ONLY)", ae_m_alive)  # PRINT -- NEW
        print_metric_block(f"TEST MATRIX {int(test_idx[j])} DMD (ONE-STEP)", dmd_m)  # PRINT
        print_metric_block(f"TEST MATRIX {int(test_idx[j])} PREDICT (+{k}, FULL GRID)", pred_m)  # PRINT
        if pred_m_alive is not None:  # PRINT -- NEW
            print_metric_block(f"TEST MATRIX {int(test_idx[j])} PREDICT (+{k}, ALIVE-ONLY)", pred_m_alive)  # PRINT
        print_metric_block(f"TEST MATRIX {int(test_idx[j])} AE+DMD ROLLOUT x1 -> x{maxit} (MACRO, FULL GRID)", final_m)  # PRINT -- NEW

        if j == 0:  # SAVE ONE VISUAL EXAMPLE (GT + RECON + PREDICTION + TRUE NEXT)
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_final_mask.png", scale=D.IMAGE_SCALE)  # GT MASK
            save_ground_truth_escape_iters(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_escape_iters.png")  # GT FRACTAL

            Z_recon = reconstruct_true_final_snapshot(td_test, enc, dec, device)  # RECON xT
            save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_recon_final_mask.png",
                                       mode="mask", alive_mask=alive_grid)  # RECON MASK
            save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_recon_final_snapshot_mag.png",
                                       mode="mag", alive_mask=alive_grid)  # RECON MAG

            save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_pred_final_mask.png",
                                       mode="mask", alive_mask=alive_grid)  # PRED MASK
            save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_pred_final_snapshot_mag.png",
                                       mode="mag", alive_mask=alive_grid)  # PRED MAG
            save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_true_next_final_mask.png", mode="mask")  # TRUE NEXT MASK
            save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_true_next_final_snapshot_mag.png", mode="mag")  # TRUE NEXT MAG

            iters_pred = teacher_forced_escape_iters(td_test, enc, dec, dmd, device, escape_r=D.ESCAPE_R)  # PRED FRACTAL
            save_escape_image(iters_pred, max_iters=int(td_test.X_grid.shape[0]), out_png=dirs["res"] / "test_pred_escape_iters.png",
                               alive_mask=alive_grid)  # SAVE FRACTAL

            # NEW: ROLLOUT-FROM-START VISUAL
            save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                                       out_png=dirs["res"] / "test_rollout_from_start_final_mask.png",
                                       mode="mask", alive_mask=alive_grid)  # ROLLOUT MASK
            save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                                       out_png=dirs["res"] / "test_rollout_from_start_final_mag.png",
                                       mode="mag", alive_mask=alive_grid)  # ROLLOUT MAG

    mean_ae = mean_metric_dict(ae_metrics_all)  # MEAN AE (FULL GRID)
    mean_ae_alive = mean_metric_dict(ae_alive_metrics_all)  # NEW: MEAN AE (ALIVE-ONLY)
    mean_dmd = mean_metric_dict(dmd_metrics_all)  # MEAN DMD ONE-STEP
    mean_pred = mean_metric_dict(pred_metrics_all)  # MEAN PREDICTION (FULL GRID)
    print_metric_block("MEAN TEST AE (RECON, FULL GRID)", mean_ae)  # PRINT
    print_metric_block("MEAN TEST AE (RECON, ALIVE-ONLY)", mean_ae_alive)  # PRINT -- NEW
    print_metric_block("MEAN TEST DMD (ONE-STEP)", mean_dmd)  # PRINT
    print_metric_block(f"MEAN TEST PREDICT (+{k}, FULL GRID)", mean_pred)  # PRINT

    mean_pred_alive = {}  # DEFAULT: NOTHING TO REPORT
    if pred_alive_metrics_all:  # NEW: MEAN PREDICTION (ALIVE-ONLY), IF ANY TEST MATRIX HAD ALIVE PIXELS
        mean_pred_alive_raw = mean_metric_dict(pred_alive_metrics_all)  # MEAN
        print_metric_block(f"MEAN TEST PREDICT (+{k}, ALIVE-ONLY)", mean_pred_alive_raw)  # PRINT
        mean_pred_alive = {f"pred_alive_{key}": val for key, val in mean_pred_alive_raw.items()}  # PREFIX FOR METRICS FILE

    # ------------------------- NEW: MEAN ROLLOUT-FROM-START ------------------
    mean_rollout_final = mean_metric_dict(rollout_final_metrics_all)  # TEST 1, FULL GRID, AVERAGED OVER TEST MATRICES
    print_metric_block(f"MEAN TEST AE+DMD ROLLOUT x1 -> x{maxit} (MACRO, FULL GRID)", mean_rollout_final)  # PRINT
    mean_rollout_final_alive = {}  # DEFAULT
    if rollout_final_alive_metrics_all:  # AT LEAST ONE TEST MATRIX HAD ALIVE PIXELS
        mean_rollout_final_alive_raw = mean_metric_dict(rollout_final_alive_metrics_all)  # MEAN
        print_metric_block(f"MEAN TEST AE+DMD ROLLOUT x1 -> x{maxit} (MACRO, ALIVE-ONLY)", mean_rollout_final_alive_raw)  # PRINT
        mean_rollout_final_alive = {f"rollout_final_alive_{key}": val for key, val in mean_rollout_final_alive_raw.items()}  # PREFIX

    rollout_step_metrics: dict[str, float] = {}  # FOR mean_test_metrics.txt
    mean_step_curve: list[float] = []  # FOR THE PLOT
    if rollout_step_curves_all:  # SHOULD ALWAYS BE TRUE IF n_check > 0
        mean_step_curve = np.mean(np.asarray(rollout_step_curves_all, dtype=np.float64), axis=0).tolist()  # (n_check,)
        for s, val in enumerate(mean_step_curve):  # SAVE EACH STEP
            rollout_step_metrics[f"rollout_rel_l2_step_{s + 1:03d}"] = float(val)  # SAVE

    mean_step_curve_alive: list[float] = []  # SAME, ALIVE-ONLY
    if rollout_step_curves_alive_all:  # AT LEAST ONE TEST MATRIX CONTRIBUTED
        mean_step_curve_alive = np.mean(np.asarray(rollout_step_curves_alive_all, dtype=np.float64), axis=0).tolist()
        for s, val in enumerate(mean_step_curve_alive):  # SAVE EACH STEP
            rollout_step_metrics[f"rollout_rel_l2_alive_step_{s + 1:03d}"] = float(val)  # SAVE

    if mean_step_curve:  # SOMETHING TO PLOT
        save_loss_curve(
            mean_step_curve, dirs["res"] / "rollout_rel_l2_vs_step.png",
            f"AE+DMD Rollout Relative L2 Error vs Steps Beyond x1 (Mean Over {len(rollout_step_curves_all)} Test Matrices, Full Grid)",
            xlabel="Steps beyond x1", ylabel="Relative L2 error", log_scale=False,
        )
    if mean_step_curve_alive:  # SOMETHING TO PLOT
        save_loss_curve(
            mean_step_curve_alive, dirs["res"] / "rollout_rel_l2_vs_step_alive.png",
            f"AE+DMD Rollout Relative L2 Error vs Steps Beyond x1 (Mean Over {len(rollout_step_curves_alive_all)} Test Matrices, Alive-Only)",
            xlabel="Steps beyond x1", ylabel="Relative L2 error", log_scale=False,
        )
    # ---------------------------------------------------------------------

    write_metrics_txt(  # SAVE
        dirs["res"] / "mean_test_metrics.txt",
        {
            **mean_ae, **mean_ae_alive, **mean_dmd, **mean_pred, **mean_pred_alive,
            "predict_extra_steps": float(k), "dmd_spectral_radius": rho,
            **{f"rollout_final_{key}": val for key, val in mean_rollout_final.items()},
            **mean_rollout_final_alive,
            **rollout_step_metrics,
        },
    )


if __name__ == "__main__":  # DIRECT RUN
    run_multi_matrix()  # RUN