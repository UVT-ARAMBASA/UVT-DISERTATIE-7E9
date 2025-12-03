import torch  # TORCH IMPORT
from torch.utils.data import DataLoader, TensorDataset  # DATALOADER IMPORT

from encoder import Encoder  # ENCODER
from decoder import Decoder  # DECODER
from latent_dynamics import DMDDynamics  # DMD
from classification import StabilityClassifier, classify_orbit  # CLASSIFIER
from data_loader import load_matlab_matrix, normalize_data, split_sequences  # DATA UTILS
from utils import to_tensor, get_device, save_model  # SYSTEM UTILS
from validation import plot_latent_orbit, plot_reconstruction, plot_loss_curve, mse  # VALIDATION

import os  # OS IMPORT
os.makedirs("plots", exist_ok=True)  # CREATE PLOTS FOLDER
os.makedirs("checkpoints", exist_ok=True)  # CREATE CHECKPOINT FOLDER

# --------------------------- DEVICE ---------------------------
device = get_device()  # GET DEVICE
print("DEVICE:", device)  # PRINT DEVICE

# --------------------------- LOAD DATA ---------------------------
X = load_matlab_matrix("matrices.mat", "X")  # LOAD MATLAB MATRIX
X = normalize_data(X)  # NORMALIZE DATA
X1, X2 = split_sequences(X)  # SPLIT INTO PAIRS

X_t = to_tensor(X, device)  # FULL DATA TO DEVICE

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

# --------------------------- TRAIN AUTOENCODER ---------------------------
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)  # OPTIMIZER
epochs = 50  # NUM EPOCHS
losses = []  # LOSS LIST

for e in range(epochs):  # EPOCH LOOP
    epoch_loss = 0.0  # RESET LOSS

    for (batch,) in loader:  # MINI-BATCH LOOP
        Z = enc(batch)  # ENCODE
        X_rec = dec(Z)  # DECODE
        loss = mse(X_rec, batch)  # MSE LOSS

        optimizer.zero_grad()  # ZERO GRAD
        loss.backward()  # BACKPROP
        optimizer.step()  # STEP OPT

        epoch_loss += loss.item()  # ACC LOSS

    epoch_loss /= len(loader)  # AVG LOSS
    losses.append(epoch_loss)  # SAVE LOSS

    print(f"EPOCH {e+1}/{epochs} LOSS={epoch_loss:.6f}")  # PRINT STATUS

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

print("DONE. ALL FILES SAVED.")  # FINISH MESSAGE
