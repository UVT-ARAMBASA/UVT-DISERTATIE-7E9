# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# train_autoencoder.py #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# IMPORTS #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
import numpy as np  # NUMPY
import torch  # TORCH
import torch.nn as nn  # NEURAL MODULES
from torch.utils.data import DataLoader, TensorDataset  # DATALOADER

from losses import make_reconstruction_loss, koopman_ae_loss, compute_target_scale  # LOSS FUNCTIONS

import defines as D  # FORGOT DEFINES

from encoder import Encoder  # ENCODER
from decoder import Decoder  # DECODER


# from losses import make_reconstruction_loss  # LOSS

# ============================ VALIDATION SPLIT HELPER =========================
def _split_block_train_val(  # PER-BLOCK TRAIN/VAL ROW SPLIT
    X1_block: np.ndarray,  # ONE BLOCK, LEFT
    X2_block: np.ndarray,  # ONE BLOCK, RIGHT
    *,
    val_fraction: float,  # FRACTION TO HOLD OUT
    seed: int,  # REPRODUCIBLE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # (X1_TRAIN, X2_TRAIN, X1_VAL, X2_VAL)
    n = int(X1_block.shape[0])  # ROWS IN BLOCK
    if val_fraction <= 0.0 or n < 2:  # NOTHING TO SPLIT
        empty = X1_block[:0]  # EMPTY, RIGHT DTYPE/SHAPE
        return X1_block, X2_block, empty, X2_block[:0]  # ALL TRAIN, NO VAL

    rng = np.random.default_rng(seed)  # PER-BLOCK RNG (SAME SEED, DIFFERENT BLOCK SIZE -> DIFFERENT DRAW, STILL REPRODUCIBLE)
    idx = rng.permutation(n)  # SHUFFLE ROW INDICES
    n_val = max(1, int(round(val_fraction * n)))  # AT LEAST 1 VAL ROW
    n_val = min(n_val, n - 1)  # KEEP AT LEAST 1 TRAIN ROW
    val_idx = idx[:n_val]  # VAL ROWS
    train_idx = idx[n_val:]  # TRAIN ROWS

    return (
        X1_block[train_idx], X2_block[train_idx],
        X1_block[val_idx], X2_block[val_idx],
    )  # RETURN SPLIT


@torch.no_grad()  # NO GRAD FOR VALIDATION
def _validation_loss(  # MEAN KOOPMAN LOSS OVER A HELD-OUT SET
    enc, dec, X1_val: np.ndarray, X2_val: np.ndarray, *,
    loss_fn, ridge: float, loss_scale: float, device: torch.device, batch_size: int,
) -> float:
    if X1_val.shape[0] == 0:  # NOTHING TO VALIDATE
        return float("nan")  # NO VAL SET

    X1_cpu = torch.tensor(np.asarray(X1_val, dtype=np.float32))  # CPU
    X2_cpu = torch.tensor(np.asarray(X2_val, dtype=np.float32))  # CPU
    loader = DataLoader(TensorDataset(X1_cpu, X2_cpu), batch_size=batch_size, shuffle=False, drop_last=False)  # LOADER

    s = 0.0  # LOSS SUM
    n_batches = 0  # COUNT
    for b1_cpu, b2_cpu in loader:  # BATCH LOOP
        b1 = b1_cpu.to(device)  # MOVE
        b2 = b2_cpu.to(device)  # MOVE

        z1 = enc(b1)  # ENC
        z2 = enc(b2)  # ENC
        x1_rec = dec(z1)  # DEC
        x2_rec = dec(z2)  # DEC

        loss, _ = koopman_ae_loss(  # SAME LOSS AS TRAINING, NO GRAD
            x1=b1, x2=b2, z1=z1, z2=z2, x1_rec=x1_rec, x2_rec=x2_rec,
            decoder=dec, base_loss=loss_fn,
            alpha_rec=D.AE_REC_WEIGHT, alpha_lin=D.AE_LATENT_WEIGHT, alpha_pred=D.AE_PRED_WEIGHT,
            ridge=ridge, scale=loss_scale,
        )
        s += float(loss.detach().cpu())  # ACCUM
        n_batches += 1  # COUNT

    return s / max(1, n_batches)  # MEAN


def _clone_state_dict(model: nn.Module) -> dict:  # DEEP-ISH COPY FOR "BEST MODEL" TRACKING
    return {k: v.clone() if hasattr(v, "clone") else np.array(v, copy=True)
            for k, v in model.state_dict().items()}  # COPY EVERY TENSOR


# #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# TRAIN LOOP #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
def train_autoencoder(  # TRAIN AE + DMD-LIKE PIPELINE
        X1: np.ndarray | list[np.ndarray],  # LEFT STATE OR LIST
        X2: np.ndarray | list[np.ndarray],  # RIGHT STATE OR LIST
        *,
        latent_dim: int,  # LATENT SIZE
        epochs: int,  # EPOCHS
        batch_size: int,  # BATCH
        lr: float,  # LR
        device: torch.device,  # DEVICE
) -> tuple[Encoder, Decoder, list[float], dict[str, list[float]], list[float]]:  # RETURNS MODELS + LOSSES + COMPONENTS + VAL LOSSES
    if isinstance(X1, np.ndarray):  # SINGLE ARRAY
        X1_list = [X1]  # WRAP
    else:  # MANY ARRAYS
        X1_list = list(X1)  # COPY

    if isinstance(X2, np.ndarray):  # SINGLE ARRAY
        X2_list = [X2]  # WRAP
    else:  # MANY ARRAYS
        X2_list = list(X2)  # COPY

    if len(X1_list) == 0 or len(X2_list) == 0:  # EMPTY
        raise ValueError("X1_list OR X2_list IS EMPTY")  # ERROR
    if len(X1_list) != len(X2_list):  # SIZE MISMATCH
        raise ValueError("X1_list AND X2_list MUST HAVE SAME LENGTH")  # ERROR

    in_dim = int(X1_list[0].shape[1])  # INPUT DIM
    torch.manual_seed(0)  # FIX INIT SO RUNS ARE COMPARABLE
    enc = Encoder(in_dim, latent_dim).to(device)  # BUILD ENCODER
    dec = Decoder(latent_dim, in_dim).to(device)  # BUILD DECODER

    loss_fn = make_reconstruction_loss(0, beta=0.01, w_pow=1.0)  # MSE MODE
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)  # OPTIM

    # ------------------------- NEW: TRAIN/VAL SPLIT --------------------------
    val_fraction = float(getattr(D, "AE_VAL_FRACTION", 0.0))  # OPT-OUT SWITCH
    val_seed = int(getattr(D, "AE_VAL_SEED", 0))  # REPRODUCIBLE

    X1_train_list, X2_train_list = [], []  # TRAIN BLOCKS
    X1_val_parts, X2_val_parts = [], []  # VAL ROWS, POOLED ACROSS BLOCKS

    for bi, (X1_block, X2_block) in enumerate(zip(X1_list, X2_list)):  # SPLIT EACH BLOCK
        x1tr, x2tr, x1va, x2va = _split_block_train_val(  # PER-BLOCK SPLIT
            np.asarray(X1_block, dtype=np.float32), np.asarray(X2_block, dtype=np.float32),
            val_fraction=val_fraction, seed=val_seed + bi,
        )
        X1_train_list.append(x1tr)  # KEEP TRAIN
        X2_train_list.append(x2tr)  # KEEP TRAIN
        if x1va.shape[0] > 0:  # HAVE VAL ROWS
            X1_val_parts.append(x1va)  # POOL
            X2_val_parts.append(x2va)  # POOL

    have_val = len(X1_val_parts) > 0  # ANY VALIDATION DATA AT ALL?
    if have_val:  # POOL INTO ONE VAL SET
        X1_val = np.concatenate(X1_val_parts, axis=0)  # STACK
        X2_val = np.concatenate(X2_val_parts, axis=0)  # STACK
    else:
        X1_val = X1_list[0][:0]  # EMPTY, RIGHT DTYPE
        X2_val = X2_list[0][:0]  # EMPTY, RIGHT DTYPE

    n_train_total = int(sum(b.shape[0] for b in X1_train_list))  # TOTAL TRAIN ROWS
    n_val_total = int(X1_val.shape[0])  # TOTAL VAL ROWS
    print(f"[TRAIN AE] train/val split: {n_train_total} train rows, {n_val_total} val rows "
          f"(AE_VAL_FRACTION={val_fraction:g})")  # LOG SO IT'S NEVER A MYSTERY
    # ---------------------------------------------------------------------

    if bool(getattr(D, "AE_LOSS_AUTO_SCALE", True)):  # OPT-OUT SWITCH
        loss_scale = compute_target_scale(*X1_train_list, *X2_train_list)  # MEAN(X^2) ACROSS TRAIN BLOCKS
    else:
        loss_scale = 1.0  # OLD RAW-MSE BEHAVIOUR
    print(f"[TRAIN AE] loss_scale={loss_scale:.6e} "
          f"(AE_LOSS_AUTO_SCALE={bool(getattr(D, 'AE_LOSS_AUTO_SCALE', True))})")  # LOG SO IT'S NEVER A MYSTERY

    scheduler = None  # DEFAULT: NO SCHEDULER (OLD FIXED-LR BEHAVIOUR)
    if bool(getattr(D, "AE_USE_LR_SCHEDULER", False)):  # OPT-IN, MATCHES fikl's torch_train
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # ADAPTIVE LR
            opt,
            mode="min",
            patience=int(getattr(D, "AE_LR_PATIENCE", 5)),
            factor=float(getattr(D, "AE_LR_FACTOR", 0.5)),
            min_lr=float(getattr(D, "AE_LR_MIN", 1e-6)),
        )

    early_stop_patience = int(getattr(D, "AE_EARLY_STOP_PATIENCE", 0))  # 0 = OFF (fikl's DEFAULT)
    best_val = float("inf")  # BEST VALIDATION LOSS SO FAR
    best_state = None  # (enc_state, dec_state) AT best_val
    best_epoch = -1  # WHICH EPOCH THAT WAS
    bad_epochs = 0  # EPOCHS SINCE LAST IMPROVEMENT

    losses: list[float] = []  # TRAIN LOSS LIST
    val_losses: list[float] = []  # NEW: VALIDATION LOSS LIST (NaN PER EPOCH IF have_val IS False)
    loss_components: dict[str, list[float]] = {"rec": [], "lin": [], "pred": []}  # PER-EPOCH BREAKDOWN

    for e in range(epochs):  # EPOCH LOOP
        rng = np.random.default_rng(e)  # RNG
        order = rng.permutation(len(X1_train_list))  # SHUFFLE DATA BLOCKS (TRAIN PORTION ONLY)

        s = 0.0  # LOSS SUM
        s_rec = 0.0  # RECONSTRUCTION COMPONENT SUM
        s_lin = 0.0  # LATENT LINEARITY COMPONENT SUM
        s_pred = 0.0  # PREDICTION COMPONENT SUM
        n_batches = 0  # BATCH COUNT

        for block_id in order:  # LOOP BLOCKS
            X1_block = np.asarray(X1_train_list[int(block_id)], dtype=np.float32)  # ONE BLOCK (TRAIN ONLY)
            X2_block = np.asarray(X2_train_list[int(block_id)], dtype=np.float32)  # ONE BLOCK (TRAIN ONLY)
            if X1_block.shape[0] == 0:  # EMPTY BLOCK (CAN HAPPEN WITH VERY SMALL BLOCKS + HIGH val_fraction)
                continue  # SKIP

            X1_cpu = torch.tensor(X1_block, dtype=torch.float32)  # KEEP ON CPU
            X2_cpu = torch.tensor(X2_block, dtype=torch.float32)  # KEEP ON CPU

            loader = DataLoader(  # LOADER
                TensorDataset(X1_cpu, X2_cpu),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )

            for b1_cpu, b2_cpu in loader:  # BATCH LOOP
                b1 = b1_cpu.to(device)  # MOVE BATCH ONLY
                b2 = b2_cpu.to(device)  # MOVE BATCH ONLY

                # base_loss = make_reconstruction_loss(loss_mode=0)  # MSE

                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# AE ENCODE - BEGIN #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
                z1 = enc(b1)  # ENCODE X1
                z2 = enc(b2)  # ENCODE X2
                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# AE ENCODE - END #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#

                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# AE RECONSTRUCT - BEGIN #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
                x1_rec = dec(z1)  # RECONSTRUCT X1
                x2_rec = dec(z2)  # RECONSTRUCT X2
                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# AE RECONSTRUCT - END #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#

                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# COMPUTE KOOPMAN AE LOSS - BEGIN #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
                loss, loss_info = koopman_ae_loss(  # COMPUTE FULL AE + DMD LOSS
                    x1=b1,  # TRUE X1
                    x2=b2,  # TRUE X2
                    z1=z1,  # LATENT X1
                    z2=z2,  # LATENT X2
                    x1_rec=x1_rec,  # RECONSTRUCTED X1
                    x2_rec=x2_rec,  # RECONSTRUCTED X2
                    decoder=dec,  # DECODER
                    base_loss=loss_fn,  # BASIC LOSS FUNCTION
                    alpha_rec=D.AE_REC_WEIGHT,  # RECONSTRUCTION WEIGHT
                    alpha_lin=D.AE_LATENT_WEIGHT,  # LATENT DMD WEIGHT
                    alpha_pred=D.AE_PRED_WEIGHT,  # DECODED PREDICTION WEIGHT
                    ridge=D.DMD_RIDGE,  # SAFE DMD FIT
                    scale=loss_scale,  # SCALE-NORMALISE (SEE ABOVE)
                    # VALUES WERE HARDCODED IN V30. 'TWAS A MISTAKE ON MY PART
                )
                # #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=# COMPUTE KOOPMAN AE LOSS - END #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#

                # OPTIMIZER STEP
                # BEGIN #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#
                opt.zero_grad()  # ZERO
                loss.backward()  # BACKPROP
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()),
                                               max_norm=1.0)  # FIXES LOSS SPIKE AT SYMPTOM LEVEL (MAYBE REMOVE) - UNSURE IF THE RIGHT APPROACH
                opt.step()  # STEP

                # OPTIMIZER STEP
                # END #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=#

                s += float(loss.detach().cpu().item())  # ACCUM
                s_rec += float(loss_info["rec"])  # ACCUM RAW (NOT SCALE-NORMALISED) COMPONENTS
                s_lin += float(loss_info["lin"])  # ACCUM
                s_pred += float(loss_info["pred"])  # ACCUM
                n_batches += 1  # COUNT

        s /= max(1, n_batches)  # MEAN
        losses.append(s)  # STORE
        loss_components["rec"].append(s_rec / max(1, n_batches))  # STORE MEAN COMPONENT
        loss_components["lin"].append(s_lin / max(1, n_batches))  # STORE MEAN COMPONENT
        loss_components["pred"].append(s_pred / max(1, n_batches))  # STORE MEAN COMPONENT

        # ------------------------- NEW: VALIDATION PASS -----------------------
        enc.eval(); dec.eval()  # EVAL MODE (NO-OP FOR THIS ARCHITECTURE, BUT CORRECT PRACTICE)
        v = _validation_loss(  # COMPUTE VAL LOSS
            enc, dec, X1_val, X2_val,
            loss_fn=loss_fn, ridge=D.DMD_RIDGE, loss_scale=loss_scale,
            device=device, batch_size=batch_size,
        )
        enc.train(); dec.train()  # BACK TO TRAIN MODE
        val_losses.append(v)  # STORE (NaN IF NO VAL SET)

        # WHICH SIGNAL DRIVES "BEST MODEL" / SCHEDULER: VALIDATION IF WE HAVE
        # ONE (MATCHES fikl's torch_train), ELSE FALL BACK TO TRAIN LOSS (OLD
        # BEHAVIOUR, AE_VAL_FRACTION=0.0).
        monitor = v if have_val else s  # WHAT WE TRACK "BEST" AGAINST

        if scheduler is not None:  # STEP LR SCHEDULER ON THE MONITORED LOSS
            lr_before = opt.param_groups[0]["lr"]  # BEFORE
            scheduler.step(monitor)  # ADAPT
            lr_after = opt.param_groups[0]["lr"]  # AFTER
            if lr_after < lr_before:  # ONLY PRINT ON ACTUAL DROPS
                print(f"  [LR SCHEDULER] dropped {lr_before:.3e} -> {lr_after:.3e}")  # LOG

        # ------------------------- NEW: BEST-CHECKPOINT TRACKING --------------
        if monitor < best_val - 1e-12:  # IMPROVED
            best_val = monitor  # UPDATE
            best_epoch = e + 1  # HUMAN EPOCH NUMBER
            best_state = (_clone_state_dict(enc), _clone_state_dict(dec))  # SNAPSHOT
            bad_epochs = 0  # RESET
        else:
            bad_epochs += 1  # NO IMPROVEMENT

        val_str = f" val={v:.6e}" if have_val else ""  # ONLY SHOW IF WE HAVE A VAL SET
        print(f"EPOCH {e + 1}/{epochs} LOSS={s:.6e}{val_str}  "
              f"[rec={loss_components['rec'][-1]:.3e} lin={loss_components['lin'][-1]:.3e} "
              f"pred={loss_components['pred'][-1]:.3e}]")  # LOG

        if have_val and early_stop_patience > 0 and bad_epochs >= early_stop_patience:  # OPT-IN EARLY STOP
            print(f"  [EARLY STOP] no val improvement for {bad_epochs} epochs "
                  f"(best {best_val:.6e} @ epoch {best_epoch})")  # LOG
            break  # STOP TRAINING

    # ------------------------- NEW: RESTORE BEST CHECKPOINT -------------------
    if best_state is not None:  # HAVE A BEST SNAPSHOT
        enc.load_state_dict(best_state[0])  # RESTORE ENC
        dec.load_state_dict(best_state[1])  # RESTORE DEC
        which = "validation" if have_val else "train"  # WHAT WE MONITORED
        print(f"[TRAIN AE] restored best checkpoint from epoch {best_epoch} "
              f"({which} loss {best_val:.6e})")  # LOG

    return enc, dec, losses, loss_components, val_losses  # RETURN (val_losses IS NEW)