# ================================= main.py ==================================
from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================= IMPORTS ==================================
import os  # OS
from pathlib import Path  # PATH
import numpy as np  # NUMPY
import torch  # TORCH

import defines as D  # DEFINES

from utils import get_device, save_model, to_tensor  # YOUR UTILS
from prepare_training_data import build_mandelbrot_training_data, save_training_npz  # DATA
from train_autoencoder import train_autoencoder  # TRAIN AE
from apply_dmd import fit_dmd_on_arrays  # DMD FIT
from mandelbrot_reconstruct import build_c_grid, reconstruct_mandelbrot, save_escape_image, save_and_show_plot  # RECON

# ============================== DEVICE PICKER ================================
def pick_device() -> torch.device:  # SELECT DEVICE
    if D.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():  # CUDA AVAILABLE
        return torch.device("cuda")  # GPU
    return torch.device("cpu")  # CPU

# =================================== MAIN ===================================
def main() -> None:  # ENTRYPOINT
    device = pick_device()  # DEVICE
    print("DEVICE:", device)  # PRINT

    print("CWD:", os.getcwd())  # SHOW WORKING DIR
    print("OUTPUT_IMAGE_PNG:", D.OUTPUT_IMAGE_PNG)  # SHOW FULL PATH

    Path(D.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)  # MAKE DIR
    Path(D.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)  # MAKE DIR

    # ----------------------- 1) BUILD TRAINING DATA -------------------------
    td = build_mandelbrot_training_data(  # BUILD DATA
        c_re_min=D.C_RE_MIN,  # RE MIN
        c_re_max=D.C_RE_MAX,  # RE MAX
        c_im_min=D.C_IM_MIN,  # IM MIN
        c_im_max=D.C_IM_MAX,  # IM MAX
        n_c=D.TRAIN_C_SAMPLES,  # TRAIN C COUNT
        max_iters=D.TRAIN_MAX_ITERS,  # TRAIN ITERS
        escape_r=D.ESCAPE_R,  # ESCAPE
        seed=D.TRAIN_SEED,  # SEED
    )

    print("X:", td.X.shape, "X1:", td.X1.shape, "X2:", td.X2.shape)  # SHAPES
    save_training_npz(D.DATASET_OUT_NPZ, td)  # SAVE NPZ

    mn = float(td.meta["mn"])  # MIN
    mx = float(td.meta["mx"])  # MAX

    # ----------------------- 2) TRAIN AUTOENCODER ---------------------------
    enc, dec, losses = train_autoencoder(  # TRAIN
        td.X,  # INPUT
        latent_dim=D.LATENT_DIM,  # LATENT
        epochs=D.AE_EPOCHS,  # EPOCHS
        batch_size=D.AE_BATCH_SIZE,  # BATCH
        lr=D.AE_LR,  # LR
        device=device,  # DEVICE
    )

    save_model(enc, os.path.join(D.CHECKPOINT_DIR, "encoder.pth"))  # SAVE ENC
    save_model(dec, os.path.join(D.CHECKPOINT_DIR, "decoder.pth"))  # SAVE DEC

    # ------------------------ 3) FIT DMD IN LATENT --------------------------
    with torch.no_grad():  # NO GRAD
        Z1 = enc(to_tensor(td.X1, device)).detach().cpu().numpy()  # ENCODE X1
        Z2 = enc(to_tensor(td.X2, device)).detach().cpu().numpy()  # ENCODE X2

    dmd = fit_dmd_on_arrays(Z1, Z2, device=device)  # FIT DMD
    print("DMD FIT DONE")  # PRINT

    # ---------------------- 4) RECONSTRUCT MANDELBROT -----------------------
    C = build_c_grid(  # GRID
        c_re_min=D.C_RE_MIN,  # RE MIN
        c_re_max=D.C_RE_MAX,  # RE MAX
        c_im_min=D.C_IM_MIN,  # IM MIN
        c_im_max=D.C_IM_MAX,  # IM MAX
        grid_n=D.GRID_N,  # RES
    )

    esc_iters = reconstruct_mandelbrot(  # RECON
        encoder=enc,  # ENC
        decoder=dec,  # DEC
        dmd=dmd,  # DMD
        C=C,  # GRID
        grid_n=D.GRID_N,  # RES
        max_iters=D.MAX_ITERS,  # ITERS
        escape_r=D.ESCAPE_R,  # ESCAPE
        mn=mn,  # MIN
        mx=mx,  # MAX
        device=device,  # DEVICE
    )

    # ---------------------- 5) SAVE IMAGE (NOT PLOT) ------------------------
    if D.SAVE_IMAGE:  # SAVE PNG
        p = save_escape_image(esc_iters, max_iters=D.MAX_ITERS, out_png=D.OUTPUT_IMAGE_PNG)  # SAVE
        print("SAVED IMAGE:", p)  # PRINT

    # -------------------------- 6) OPTIONAL PLOT ----------------------------
    if D.SAVE_PLOT or D.SHOW_PLOT:  # PLOT FLAG
        p2 = save_and_show_plot(esc_iters, out_png=D.OUTPUT_IMAGE_PLOT_PNG, show=D.SHOW_PLOT)  # PLOT
        if D.SAVE_PLOT:  # PRINT IF SAVED
            print("SAVED PLOT:", p2)  # PRINT

if __name__ == "__main__":  # MAIN GUARD
    main()  # RUN
