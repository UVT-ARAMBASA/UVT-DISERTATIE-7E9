import matplotlib.pyplot as plt  # PLOTTING IMPORT
import torch  # TORCH IMPORT
import os  # OS IMPORT

def ensure_dir(path):  # ENSURE FOLDER EXISTS
    os.makedirs(os.path.dirname(path), exist_ok=True)  # CREATE FOLDER IF NEEDED

def plot_latent_orbit(orbit, save_path=None):  # PLOT LATENT ORBIT NORM
    ensure_dir(save_path)  # MAKE FOLDER
    plt.figure(figsize=(6,4))  # NEW FIGURE
    plt.plot(torch.norm(orbit, dim=1).cpu().numpy())  # PLOT ORBIT NORM
    plt.title("LATENT ORBIT (NORM)")  # TITLE
    if save_path:  # IF SAVE ENABLED
        plt.savefig(save_path)  # SAVE FIG
    plt.close()  # CLOSE

def plot_reconstruction(true_sample, recon_sample, save_path=None):  # PLOT RECON
    ensure_dir(save_path)  # MAKE FOLDER
    plt.figure(figsize=(6,4))  # FIG
    plt.plot(true_sample.cpu().numpy(), label="TRUE")  # TRUE DATA
    plt.plot(recon_sample.cpu().detach().numpy(), label="RECON")  # RECON DATA
    plt.legend()  # LEGEND
    plt.title("RECONSTRUCTION")  # TITLE
    if save_path:  # SAVE IF REQUESTED
        plt.savefig(save_path)  # SAVE FIG
    plt.close()  # CLOSE

def plot_loss_curve(losses, save_path=None):  # TRAIN LOSS CURVE
    ensure_dir(save_path)  # MAKE FOLDER
    plt.figure(figsize=(6,4))  # FIG
    plt.plot(losses)  # PLOT LOSSES
    plt.title("TRAINING LOSS")  # TITLE
    if save_path:  # IF SAVE
        plt.savefig(save_path)  # SAVE FIG
    plt.close()  # CLOSE

def mse(a, b):  # MSE LOSS
    return torch.mean((a - b)**2)  # RETURN MSE
