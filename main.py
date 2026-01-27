# --------------------------- IMPORTS ---------------------------
import torch  # TORCH IMPORT
from torch.utils.data import DataLoader, TensorDataset  # DATALOADER IMPORT
import numpy as np  # NUMPY IMPORT
from pathlib import Path  # PATH IMPORT

from losses import make_reconstruction_loss  # LOSS SWITCH
from encoder import Encoder  # ENCODER
from decoder import Decoder  # DECODER
from latent_dynamics import DMDDynamics  # DMD
from classification import StabilityClassifier, classify_orbit  # CLASSIFIER

from data_loader import load_task_npz_pair, normalize_data, make_dmd_pairs, generate_state_trajectories
  # DATA UTILS (NPZ + NORMALISE + DMD PAIRS)
from utils import to_tensor, get_device, save_model  # SYSTEM UTILS
from validation import plot_latent_orbit, plot_reconstruction, plot_loss_curve, mse, save_mandelbrot_learned  # VALIDATION
  # VALIDATION

import os  # OS IMPORT


os.makedirs("plots", exist_ok=True)  # CREATE PLOTS FOLDER
os.makedirs("checkpoints", exist_ok=True)  # CREATE CHECKPOINT FOLDER

print("CWD:", os.getcwd())  # PRINT CURRENT WORKING DIRECTORY --- LOST FILES --- PORCA MISERIA

# --------------------------- DEVICE ---------------------------
device = get_device()  # GET DEVICE
print("DEVICE:", device)  # PRINT DEVICE

# --------------------------- LOAD DATA ---------------------------
DATA_DIR = Path(__file__).resolve().parent / "data-set"  # DATA FOLDER PATH (RELATIVE TO main.py)

# LOAD A MATRICES (NOT DATA)
A_emotion, A_rest = load_task_npz_pair(DATA_DIR, key=None, flatten=False)

# GENERATE STATE DATA FROM DYNAMICS
X_e, X1_e, X2_e = generate_state_trajectories(A_emotion, n_traj=500, x0_scale=1.0, noise_std=0.0)
X_r, X1_r, X2_r = generate_state_trajectories(A_rest,    n_traj=500, x0_scale=1.0, noise_std=0.0)

# MERGE GENERATED STATE SNAPSHOTS
X  = np.concatenate([X_e,  X_r],  axis=0)
X1 = np.concatenate([X1_e, X1_r], axis=0)
X2 = np.concatenate([X2_e, X2_r], axis=0)

# NORMALISE WITH SAME SCALING (IMPORTANT)
mn, mx = X.min(), X.max()
X  = (X  - mn) / (mx - mn + 1e-12)
X1 = (X1 - mn) / (mx - mn + 1e-12)
X2 = (X2 - mn) / (mx - mn + 1e-12)

X_t = to_tensor(X, device)


# --------------------------- DATA LOADER ---------------------------
dataset = TensorDataset(X_t)  # WRAP DATASET
loader = DataLoader(dataset, batch_size=64, shuffle=True)  # CREATE LOADER

# --------------------------- INIT MODELS ---------------------------
input_dim = X.shape[1]  # INPUT SIZE
latent_dim = 8  # LATENT SIZE

enc = Encoder(input_dim, latent_dim).to(device)  # ENCODER TO DEVICE
dec = Decoder(latent_dim, input_dim).to(device)  # DECODER TO DEVICE
clf = StabilityClassifier(latent_dim).to(device)  # CLASSIFIER TO DEVICE

dmd = DMDDynamics(device=device)  # DMD WITH DEVICE
LOSS_MODE = 0  # 0=MSE, 1=MAE, 2=HUBER, 3=WEIGHTED_MSE
loss_fn = make_reconstruction_loss(LOSS_MODE, beta=0.01, w_pow=1.0)  # BUILD LOSS

# --------------------------- TRAIN AUTOENCODER ---------------------------
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)  # OPTIMIZER
epochs = 50  # NUM EPOCHS
losses = []  # LOSS LIST

for e in range(epochs):  # EPOCH LOOP
    epoch_loss = 0.0  # RESET LOSS

    for (batch,) in loader:  # MINI-BATCH LOOP
        Z = enc(batch)  # ENCODE
        X_rec = dec(Z)  # DECODE
        #loss = mse(X_rec, batch)  # MSE LOSS
        loss = loss_fn(X_rec, batch)  # SWITCHED LOSS
        optimizer.zero_grad()  # ZERO GRAD
        loss.backward()  # BACKPROP
        optimizer.step()  # STEP OPT

        epoch_loss += loss.item()  # ACC LOSS

    epoch_loss /= len(loader)  # AVG LOSS
    losses.append(epoch_loss)  # SAVE LOSS

    print(f"EPOCH {e+1}/{epochs} LOSS={epoch_loss:.6f}")  # PRINT STATUS

# --------------------------- LATENT CLOUD (FOR AUTO RANGE) ---------------------------
Z_all = enc(to_tensor(X, device)).detach().cpu().numpy()  # ALL LATENTS (FOR AUTO RANGE)

# --------------------------- SAVE MODELS ---------------------------
save_model(enc, "checkpoints/encoder.pth")  # SAVE ENCODER
save_model(dec, "checkpoints/decoder.pth")  # SAVE DECODER

# --------------------------- FIT DMD ---------------------------
Z1 = enc(to_tensor(X1, device)).detach().cpu().numpy()  # LATENT Z1
Z2 = enc(to_tensor(X2, device)).detach().cpu().numpy()  # LATENT Z2
A = dmd.fit(Z1, Z2)  # FIT DMD

# --------------------------- SIM ORBIT ---------------------------
z0 = enc(to_tensor(X1[0], device))  # INITIAL POINT
orbit = dmd.predict(z0.detach(), steps=50)  # PREDICT ORBIT

# --------------------------- CLASSIFY ---------------------------
label = classify_orbit(orbit)  # CLASSIFY ORBIT
print("STABILITY LABEL:", label)  # PRINT CLASS

# --------------------------- RECONSTRUCT ---------------------------
recon = dec(orbit.to(device))  # DECODE ORBIT

# --------------------------- MAKE PLOTS ---------------------------
plot_latent_orbit(orbit, save_path="plots/orbit.png")  # SAVE ORBIT PLOT

true_sample = to_tensor(X1[0], device)  # TRUE SAMPLE
recon_sample = recon[0]  # FIRST RECON SAMPLE
plot_reconstruction(true_sample, recon_sample, save_path="plots/reconstruction.png")  # SAVE RECON PLOT

plot_loss_curve(losses, save_path="plots/loss.png")  # SAVE LOSS PLOT

# --------------------------- LEARNED MANDELBROT ---------------------------
save_mandelbrot_learned(dmd=dmd,  # PASS DMD
                        classify_orbit_fn=classify_orbit,  # PASS CLASSIFIER
                        Z_ref=Z_all,  # USE LATENT CLOUD TO SET GRID RANGE
                        latent_dim=latent_dim,  # LATENT DIM
                        save_path="plots/mandelbrot_learned.png")  # OUTPUT FILE

print("DONE. ALL FILES SAVED.")  # FINISH MESSAGE