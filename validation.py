from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from utils import axis, set_recommended_matplotlib


def save_dmd_matrices(dmd, save_dir: str | Path = "checkpoints") -> None:
    """Export the fitted DMD matrix A in both .npy and .csv form."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    A = getattr(dmd, "A", None)
    if A is None:
        return

    A_cpu = A.detach().cpu().numpy()
    np.save(save_dir / "dmd_A.npy", A_cpu)
    np.savetxt(save_dir / "dmd_A.csv", A_cpu, delimiter=",")


def plot_latent_orbit(orbit: torch.Tensor, save_path: str | Path) -> None:
    """Visualise orbit as a heatmap (imshow), not a 1D line plot."""
    set_recommended_matplotlib()
    save_path = Path(save_path)

    X = orbit.detach().cpu().numpy()
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)

    with axis(save_path) as ax:
        ax.imshow(X.T, aspect="auto", origin="lower")
        ax.set_title("LATENT ORBIT (imshow)")
        ax.set_xlabel("step")
        ax.set_ylabel("latent dimension")


def plot_reconstruction(true_sample: torch.Tensor, recon_sample: torch.Tensor, save_path: str | Path) -> None:
    """Show TRUE / RECON / |DIFF| using imshow (fixes 'flat line' issue)."""
    set_recommended_matplotlib()
    save_path = Path(save_path)

    t = true_sample.detach().cpu().numpy().reshape(1, -1)
    r = recon_sample.detach().cpu().numpy().reshape(1, -1)
    d = np.abs(t - r)
    img = np.concatenate([t, r, d], axis=0)

    with axis(save_path) as ax:
        ax.imshow(img, aspect="auto", origin="lower")
        ax.set_title("RECONSTRUCTION (TRUE / RECON / |DIFF|)")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["TRUE", "RECON", "|DIFF|"])
        ax.set_xlabel("feature index")


def plot_loss_curve(losses: list[float], save_path: str | Path) -> None:
    set_recommended_matplotlib()
    save_path = Path(save_path)

    with axis(save_path) as ax:
        ax.plot(losses)
        ax.set_title("TRAINING LOSS")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")


def save_mandelbrot_learned(
    dmd,
    classify_orbit_fn: Callable[[torch.Tensor], int],
    *,
    Z_ref: np.ndarray | None = None,
    latent_dim: int = 8,
    save_path: str | Path = "plots/mandelbrot_learned.png",
    width: int = 500,
    height: int = 500,
    steps: int = 80,
    range_scale: float = 2.5,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Scan (z0[0], z0[1]) and classify orbits using the learned model."""
    set_recommended_matplotlib()
    save_path = Path(save_path)

    base = np.zeros(latent_dim, dtype=np.float32)

    if (x_min is not None) and (x_max is not None) and (y_min is not None) and (y_max is not None):
        pass
    elif Z_ref is not None:
        Z_ref = np.asarray(Z_ref)
        mu = Z_ref.mean(axis=0)
        sd = Z_ref.std(axis=0) + 1e-12
        x_min = float(mu[0] - range_scale * sd[0])
        x_max = float(mu[0] + range_scale * sd[0])
        y_min = float(mu[1] - range_scale * sd[1])
        y_max = float(mu[1] + range_scale * sd[1])
        base = mu.astype(np.float32, copy=False)
    else:
        x_min, x_max = -2.0, 2.0
        y_min, y_max = -2.0, 2.0

    xs = np.linspace(float(x_min), float(x_max), width)
    ys = np.linspace(float(y_min), float(y_max), height)
    img = np.zeros((height, width), dtype=np.uint8)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            z0 = torch.tensor(base, dtype=torch.float32)
            z0[0] = float(x)
            z0[1] = float(y)

            orbit = dmd.predict(z0, steps=steps)
            label = classify_orbit_fn(orbit)
            img[iy, ix] = 1 if int(label) == 1 else 0

    with axis(save_path) as ax:
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="binary")
        ax.set_title("Learned stability set (latent plane)")
        ax.set_xlabel("z0[0]")
        ax.set_ylabel("z0[1]")
