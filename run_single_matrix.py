from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================ IMPORTS ==================================
import os  # OS
import numpy as np  # NUMPY
import torch  # TORCH

import defines as D  # DEFINES
from utils import save_model, to_tensor  # HELPERS
from data_loader import load_one_A_matrix  # TRUE MATRIX FOR GROUND-TRUTH NEXT STEP
from prepare_training_data import build_matrix_c_grid_training_data, save_training_npz  # DATA
from train_autoencoder import train_autoencoder  # AE TRAINING
from apply_dmd import fit_dmd_on_arrays  # DMD FIT
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
    predict_rollout_from_start_ae_dmd,
    teacher_forced_escape_iters,
)
from mandelbrot_reconstruct import save_final_snapshot_image, save_escape_image  # IMAGE SAVERS
from experiment_common import (  # COMMON
    pick_device,
    make_out_dirs,
    print_metric_block,
    write_metrics_txt,
    debug_final_state_stats,
)


# ============================= SINGLE MATRIX RUN ============================
def run_single_matrix(device: torch.device | None = None) -> None:  # RUN SINGLE EXPERIMENT
    if device is None:  # AUTO DEVICE
        device = pick_device()  # PICK DEVICE

    print("DEVICE:", device)  # PRINT
    print("CWD:", os.getcwd())  # PRINT

    dirs = make_out_dirs("single-matrix")  # out/single-matrix/{training-data,results}

    print("\n================ SINGLE MATRIX CHECK ================\n")  # HEADER

    # ------------------------------- BUILD DATA -----------------------------
    A = load_one_A_matrix(D.A_DATA_DIR, source=D.SINGLE_MATRIX_SOURCE, index=D.SINGLE_MATRIX_INDEX)  # TRUE MATRIX

    td = build_matrix_c_grid_training_data(  # BUILD ONE MATRIX DATA
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

    # ------------------------------ SAVE TRAINING DATA ----------------------
    save_training_npz(dirs["td"] / "training_single_matrix.npz", td)  # SAVE NPZ
    save_ground_truth_escape_iters(td, D.ESCAPE_R, dirs["td"] / "gt_escape_iters.png")  # GT FRACTAL
    save_ground_truth_final_mask(td, D.ESCAPE_R, dirs["td"] / "gt_final_mask.png", scale=D.IMAGE_SCALE)  # GT MASK

    feat_dim = int(td.X_grid.shape[-1])  # FEAT DIM
    d = (feat_dim - 2) // 2  # STATE DIM

    # ------------------------------ TRAIN AE + DMD --------------------------
    enc, dec, losses = train_autoencoder(  # TRAIN AE (KOOPMAN LOSS)
        td.X1,  # LEFT
        td.X2,  # RIGHT
        latent_dim=D.LATENT_DIM,  # LATENT
        epochs=D.AE_EPOCHS,  # EPOCHS
        batch_size=D.AE_BATCH_SIZE,  # BATCH
        lr=D.AE_LR,  # LR
        device=device,  # DEVICE
    )

    save_loss_curve(losses, dirs["res"] / "loss_curve.png", "Single Matrix AE Loss")  # LOSS
    save_model(enc, os.path.join(D.CHECKPOINT_DIR, "encoder_single_matrix.pth"))  # SAVE ENC
    save_model(dec, os.path.join(D.CHECKPOINT_DIR, "decoder_single_matrix.pth"))  # SAVE DEC

    with torch.no_grad():  # NO GRAD
        Z1 = enc(to_tensor(td.X1, device)).detach().cpu().numpy()  # ENCODE X1
        Z2 = enc(to_tensor(td.X2, device)).detach().cpu().numpy()  # ENCODE X2

    #dmd = fit_dmd_on_arrays(Z1, Z2, device=device)  # FIT DMD IN LATENT SPACE
    #v33
    dmd = fit_dmd_on_arrays(Z1, Z2, device=device, ridge=D.DMD_RIDGE)  # FIT DMD IN LATENT SPACE
    rho = float(np.max(np.abs(np.linalg.eigvals(dmd.A.detach().cpu().numpy()))))  # SPECTRAL RADIUS
    print("SINGLE DMD SPECTRAL RADIUS:", rho)  # PRINT

    # ----------------------------- RECONSTRUCTION ---------------------------
    # THE SYSTEM AS IS: ENCODE-DECODE THE TRUE FINAL STATE xT (NO TIME STEP)
    Z_recon = reconstruct_true_final_snapshot(td, enc, dec, device)  # RECON xT
    save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "recon_final_mask.png", mode="mask")  # RECON MASK
    save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "recon_final_snapshot_mag.png", mode="mag")  # RECON MAG

    # ------------------------------- PREDICTION -----------------------------
    # THEN PREDICT THE NEXT PREDICT_EXTRA_STEPS STEP(S) STARTING FROM THE TRUE xT
    k = int(D.PREDICT_EXTRA_STEPS)  # HOW MANY STEPS AHEAD

    Z_pred = predict_next_snapshot(td, enc, dec, dmd, device, steps=k, escape_r=D.ESCAPE_R)  # MODEL x_{T+k}
    Z_true_next = iterate_true_next_snapshot(td, A, steps=k, escape_r=D.ESCAPE_R)  # TRUE x_{T+k}
    debug_final_state_stats(f"SINGLE MATRIX (+{k})", Z_pred, D.ESCAPE_R)  # DEBUG

    save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "pred_final_mask.png", mode="mask")  # PRED MASK
    save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "pred_final_snapshot_mag.png", mode="mag")  # PRED MAG
    save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "true_next_final_mask.png", mode="mask")  # TRUE NEXT MASK
    save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "true_next_final_snapshot_mag.png", mode="mag")  # TRUE NEXT MAG

    # PREDICTED FRACTAL (TEACHER FORCED: PREDICT x_{t+1} FROM EACH TRUE x_t, NO COMPOUNDING)
    iters_pred = teacher_forced_escape_iters(td, enc, dec, dmd, device, escape_r=D.ESCAPE_R)  # PRED FRACTAL
    save_escape_image(iters_pred, max_iters=int(td.X_grid.shape[0]), out_png=dirs["res"] / "pred_escape_iters.png")  # SAVE FRACTAL

    #------V33
    maxit = int(D.TRAIN_MAX_ITERS)
    rollout = predict_rollout_from_start_ae_dmd(
        td, enc, dec, dmd, device, steps=maxit, escape_r=D.ESCAPE_R,
    )  # rollout[s] = AE+DMD PREDICTION OF x_{s+2}, s = 0 .. maxit-2

    # ---- TEST 1: MACRO / EYEBALL -----------------------------------
    d_state = int((int(td.X_grid.shape[-1]) - 2) // 2)  # td.X_grid HAS +2 C CHANNELS, rollout DOES NOT

    Z_pred_final = rollout[-1]  # (H,W,2d)
    Z_true_final = td.X_grid[-1][..., :2 * d_state]  # DROP TRAILING C SO SHAPES MATCH

    save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                              out_png=dirs["res"] / "rollout_from_start_final_mask.png",
                              mode="mask")
    save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                              out_png=dirs["res"] / "rollout_from_start_final_mag.png", mode="mag")

    rollout_final_m = next_step_prediction_metrics(Z_pred_final, Z_true_final)
    print_metric_block(f"SINGLE MATRIX AE+DMD ROLLOUT x1 -> x{maxit} (MACRO)", rollout_final_m)

    # ---- TEST 2: QUANTITATIVE ---------------------------------------
    n_check = min(int(getattr(D, "PREDICT_ROLLOUT_CHECK_STEPS", 10)), rollout.shape[0])
    rollout_rel_l2: list[float] = []
    rollout_step_metrics: dict = {}

    for s in range(n_check):
        true_iter = s + 2
        Z_pred_s = rollout[s]  # (H,W,2d)
        Z_true_s = td.X_grid[s + 1][..., :2 * d_state]  # DROP TRAILING C SO SHAPES MATCH

        m_s = next_step_prediction_metrics(Z_pred_s, Z_true_s)
        print_metric_block(f"SINGLE MATRIX AE+DMD ROLLOUT, {s + 1} STEP(S) IN (x{true_iter})", m_s)

        rollout_step_metrics[f"rollout_rel_l2_step_{s + 1:03d}"] = float(m_s["pred_rel_l2"])
        rollout_rel_l2.append(float(m_s["pred_rel_l2"]))

    save_loss_curve(
        rollout_rel_l2, dirs["res"] / "rollout_rel_l2_vs_step.png",
        "AE+DMD Rollout Relative L2 Error vs Steps Beyond x1",
    )

    # -------------------------------- METRICS -------------------------------
    ae_m = autoencoder_reconstruction_metrics(enc, dec, td.X, device)  # RECON METRICS
    dmd_m = dmd_one_step_metrics(enc, dec, dmd, td.X1, td.X2, device)  # ONE-STEP TEACHER-FORCED METRICS
    pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)  # PREDICTED vs TRUE NEXT STEP
    print_metric_block("SINGLE MATRIX AE (RECON)", ae_m)  # PRINT
    print_metric_block("SINGLE MATRIX DMD (ONE-STEP)", dmd_m)  # PRINT
    print_metric_block(f"SINGLE MATRIX PREDICT (+{k} FROM TRUE xT)", pred_m)  # PRINT

    #metrics = {**ae_m, **dmd_m, **pred_m, "predict_extra_steps": float(k), "dmd_spectral_radius": rho}  # MERGE
    metrics = {
        **ae_m, **dmd_m, **pred_m,
        "predict_extra_steps": float(k), "dmd_spectral_radius": rho,
        **{f"rollout_final_{key}": val for key, val in rollout_final_m.items()},
        **rollout_step_metrics,
    }  # MERGE
    write_metrics_txt(dirs["res"] / "metrics.txt", metrics)  # SAVE METRICS


if __name__ == "__main__":  # DIRECT RUN
    run_single_matrix()  # RUN