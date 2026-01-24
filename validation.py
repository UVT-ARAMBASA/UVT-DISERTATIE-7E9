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
# --------------------------- MANDELBROT (LEARNED LATENT STABILITY) ---------------------------
def save_mandelbrot_learned(dmd, classify_orbit_fn,  # DMD + CLASSIFIER
                            Z_ref=None,  # OPTIONAL: REFERENCE LATENTS FOR AUTO-RANGE
                            latent_dim=8,  # LATENT SIZE
                            save_path="plots/mandelbrot_learned.png",  # OUTPUT PATH
                            width=500, height=500,  # IMAGE RESOLUTION
                            steps=80,  # ORBIT LENGTH
                            range_scale=2.5,  # HOW WIDE THE GRID IS (IN STD UNITS)
                            stable_token="stable"):  # WHAT STRING MEANS "STABLE"
    import numpy as np  # NUMPY IMPORT
    import torch  # TORCH IMPORT
    import matplotlib.pyplot as plt  # MATPLOTLIB IMPORT
    import os  # OS IMPORT

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)  # ENSURE FOLDER EXISTS

    # --------------------------- AUTO RANGE FROM DATA ---------------------------
    if Z_ref is not None:  # IF WE HAVE REFERENCE LATENT CLOUD
        Z_ref = np.asarray(Z_ref)  # MAKE SURE NUMPY
        mu = Z_ref.mean(axis=0)  # MEAN LATENT
        sd = Z_ref.std(axis=0) + 1e-12  # STD LATENT
        x_min = mu[0] - range_scale * sd[0]  # X MIN
        x_max = mu[0] + range_scale * sd[0]  # X MAX
        y_min = mu[1] - range_scale * sd[1]  # Y MIN
        y_max = mu[1] + range_scale * sd[1]  # Y MAX
        base = mu  # BASE POINT FOR OTHER DIMS
    else:  # FALLBACK RANGE
        x_min, x_max = -2.0, 2.0  # X RANGE
        y_min, y_max = -2.0, 2.0  # Y RANGE
        base = np.zeros(latent_dim, dtype=np.float32)  # BASE Z0

    xs = np.linspace(x_min, x_max, width)  # X GRID
    ys = np.linspace(y_min, y_max, height)  # Y GRID
    img = np.zeros((height, width), dtype=np.uint8)  # OUTPUT IMAGE (0/1)

    # --------------------------- GRID SCAN ---------------------------
    for iy, y in enumerate(ys):  # Y LOOP
        for ix, x in enumerate(xs):  # X LOOP
            z0 = torch.tensor(base, dtype=torch.float32)  # START FROM BASE
            z0[0] = float(x)  # SET z0[0]
            z0[1] = float(y)  # SET z0[1]

            orbit = dmd.predict(z0, steps=steps)  # PREDICT ORBIT
            label = classify_orbit_fn(orbit)  # CLASSIFY ORBIT

            is_stable = (str(label).lower() == stable_token)  # CHECK STABLE LABEL
            img[iy, ix] = 1 if is_stable else 0  # WRITE PIXEL

    # --------------------------- PLOT ---------------------------
    plt.figure(figsize=(6, 6))  # NEW FIGURE
    plt.imshow(img, extent=[x_min, x_max, y_min, y_max], origin="lower")  # DRAW MAP
    plt.title("Learned stability set (latent plane)")  # TITLE
    plt.xlabel("z0[0]")  # X LABEL
    plt.ylabel("z0[1]")  # Y LABEL
    plt.tight_layout()  # TIGHT LAYOUT
    plt.savefig(save_path, dpi=200)  # SAVE FIG
    plt.close()  # CLOSE FIG
