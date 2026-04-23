# ============================== check-dmd.py ===============================
from __future__ import annotations

# ================================= IMPORTS =================================
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from latent_dynamics import DMDDynamics


# ============================== NUMERIC HELPERS =============================
def rel_l2_err(pred: np.ndarray, true: np.ndarray) -> float:
    num = float(np.linalg.norm(pred - true))
    den = float(np.linalg.norm(true))
    return num / max(den, 1e-12)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_A_numpy(dmd: DMDDynamics) -> np.ndarray:
    if not hasattr(dmd, "A"):
        raise AttributeError("DMDDynamics HAS NO ATTRIBUTE 'A'")
    return dmd.A.detach().cpu().numpy().astype(np.float64)


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(arr))
    if m < 1e-12:
        return arr.copy()
    return arr / m


# ============================== DMD HELPERS =================================
def fit_dmd_on_states(X: np.ndarray, device: torch.device) -> DMDDynamics:
    if X.ndim != 2:
        raise ValueError(f"EXPECTED X SHAPE (T, DIM), GOT {X.shape}")

    X1 = X[:-1].astype(np.float32, copy=False)
    X2 = X[1:].astype(np.float32, copy=False)

    dmd = DMDDynamics(device=device)
    dmd.fit(X1, X2)
    return dmd


@torch.no_grad()
def predict_from_initial(dmd: DMDDynamics, x0: np.ndarray, steps: int) -> np.ndarray:
    z0 = torch.tensor(x0, dtype=torch.float32, device=dmd.device)
    pred = dmd.predict(z0, steps=steps)
    return pred.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_one_step_batch(dmd: DMDDynamics, X1: np.ndarray) -> np.ndarray:
    z = torch.tensor(X1, dtype=torch.float32, device=dmd.device)
    pred = dmd.predict(z, steps=1)[-1]
    return pred.detach().cpu().numpy().astype(np.float32)


def compute_dmd_eigendecomposition(dmd: DMDDynamics) -> tuple[np.ndarray, np.ndarray]:
    A = get_A_numpy(dmd)
    eigvals, eigvecs = np.linalg.eig(A)
    return eigvals, eigvecs


def rank_modes_by_initial_amplitude(eigvecs: np.ndarray, x0: np.ndarray) -> np.ndarray:
    # Phi b ≈ x0
    b, *_ = np.linalg.lstsq(eigvecs, x0.astype(np.complex128), rcond=None)
    scores = np.abs(b)
    idx = np.argsort(scores)[::-1]
    return idx


# =============================== MODE PLOTS =================================
def plot_eigs_complex_plane(eigvals: np.ndarray, out_png: str | Path, title: str) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(eigvals.real, eigvals.imag, s=20)

    # unit circle
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "--")

    ax.axhline(0.0)
    ax.axvline(0.0)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return str(out_png)


def plot_linear_modes(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    out_png: str | Path,
    top_k: int = 2,
) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    k = min(top_k, eigvecs.shape[1])
    fig, axs = plt.subplots(1, k, figsize=(5 * k, 5))
    if k == 1:
        axs = [axs]

    for i in range(k):
        ax = axs[i]
        v = eigvecs[:, i]
        vr = normalize_for_display(v.real)
        vi = normalize_for_display(v.imag)

        ax.quiver(0, 0, vr[0], vr[1], angles="xy", scale_units="xy", scale=1, label="real part")
        ax.quiver(0, 0, vi[0], vi[1], angles="xy", scale_units="xy", scale=1, label="imag part")

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal", adjustable="box")
        lam = eigvals[i]
        ax.set_title(f"Mode {i + 1}\nλ={lam.real:.4f}+{lam.imag:.4f}j")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return str(out_png)


def plot_field_modes(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    x0: np.ndarray,
    shape_hw: tuple[int, int],
    out_png: str | Path,
    top_k: int = 6,
) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    h, w = shape_hw
    idx_ranked = rank_modes_by_initial_amplitude(eigvecs, x0)
    idx_used = idx_ranked[:min(top_k, eigvecs.shape[1])]

    k = len(idx_used)
    fig, axs = plt.subplots(2, k, figsize=(4 * k, 8))
    if k == 1:
        axs = np.array(axs).reshape(2, 1)

    for col, idx in enumerate(idx_used):
        phi = eigvecs[:, idx]
        lam = eigvals[idx]
        field_r = normalize_for_display(phi.real.reshape(h, w))
        field_i = normalize_for_display(phi.imag.reshape(h, w))

        im0 = axs[0, col].imshow(field_r, origin="lower")
        axs[0, col].set_title(f"Mode {idx}\nReal\nλ={lam.real:.4f}+{lam.imag:.4f}j")
        axs[0, col].set_xticks([])
        axs[0, col].set_yticks([])
        plt.colorbar(im0, ax=axs[0, col], fraction=0.046)

        im1 = axs[1, col].imshow(field_i, origin="lower")
        axs[1, col].set_title(f"Mode {idx}\nImag")
        axs[1, col].set_xticks([])
        axs[1, col].set_yticks([])
        plt.colorbar(im1, ax=axs[1, col], fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return str(out_png)


# ============================ SYSTEM 1: LINEAR ==============================
def make_linear_rotation_decay_data(
    steps: int = 120,
    theta: float = 0.18,
    rho: float = 0.985,
) -> tuple[np.ndarray, np.ndarray]:
    c = np.cos(theta)
    s = np.sin(theta)

    A_true = rho * np.array([
        [c, -s],
        [s,  c],
    ], dtype=np.float32)

    x = np.array([2.0, -0.5], dtype=np.float32)
    X = [x.copy()]

    for _ in range(steps - 1):
        x = A_true @ x
        X.append(x.copy())

    return np.stack(X, axis=0).astype(np.float32), A_true


def plot_linear_demo(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    out_png: str | Path,
) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    t = np.arange(X_true.shape[0], dtype=np.int32)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(t, X_true[:, 0], label="true x1")
    axs[0].plot(t, X_pred[:, 0], "--", label="dmd x1")
    axs[0].set_title("Component x1")
    axs[0].set_xlabel("step")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, X_true[:, 1], label="true x2")
    axs[1].plot(t, X_pred[:, 1], "--", label="dmd x2")
    axs[1].set_title("Component x2")
    axs[1].set_xlabel("step")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(X_true[:, 0], X_true[:, 1], label="true")
    axs[2].plot(X_pred[:, 0], X_pred[:, 1], "--", label="dmd")
    axs[2].set_title("Phase portrait")
    axs[2].set_xlabel("x1")
    axs[2].set_ylabel("x2")
    axs[2].axis("equal")
    axs[2].grid(True)
    axs[2].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return str(out_png)


def run_linear_demo(out_dir: Path, device: torch.device) -> None:
    X, A_true = make_linear_rotation_decay_data()
    dmd = fit_dmd_on_states(X, device)

    X1 = X[:-1]
    X2 = X[1:]
    X2_hat = predict_one_step_batch(dmd, X1)

    rollout = predict_from_initial(dmd, X[0], steps=X.shape[0] - 1)
    X_roll = np.vstack([X[0:1], rollout])

    one_step_err = rel_l2_err(X2_hat, X2)
    rollout_err = rel_l2_err(X_roll, X)

    eig_true = np.linalg.eigvals(A_true)
    eig_dmd, modes_dmd = compute_dmd_eigendecomposition(dmd)

    verdict = "PASS" if rollout_err < 1e-4 else "FAIL"

    print("\n================ LINEAR DMD CHECK ================\n")
    print("SYSTEM: stable 2D linear rotation-decay")
    print(f"STATE SHAPE: {X.shape}")
    print(f"ONE-STEP REL L2 ERROR: {one_step_err:.8e}")
    print(f"ROLLOUT  REL L2 ERROR: {rollout_err:.8e}")
    print(f"TRUE  EIGENVALUES: {eig_true}")
    print(f"LEARNED EIGENVALUES: {eig_dmd}")
    print(f"VERDICT: {verdict}")

    out_png = plot_linear_demo(X, X_roll, out_dir / "linear2d_dmd_check.png")
    print(f"PLOT SAVED TO: {out_png}")

    eig_png = plot_eigs_complex_plane(
        eig_dmd,
        out_dir / "linear2d_dmd_eigenvalues.png",
        title="Linear2D DMD Eigenvalues",
    )
    print(f"EIGENVALUE PLOT SAVED TO: {eig_png}")

    modes_png = plot_linear_modes(
        eig_dmd,
        modes_dmd,
        out_dir / "linear2d_dmd_modes.png",
        top_k=2,
    )
    print(f"MODE PLOT SAVED TO: {modes_png}")


# ======================== SYSTEM 2: REACTION-DIFFUSION ======================
def laplacian_periodic(U: np.ndarray) -> np.ndarray:
    return (
        np.roll(U, 1, axis=0) +
        np.roll(U, -1, axis=0) +
        np.roll(U, 1, axis=1) +
        np.roll(U, -1, axis=1) -
        4.0 * U
    )


def make_reaction_diffusion_2d_data(
    nx: int = 32,
    ny: int = 32,
    steps: int = 100,
    dt: float = 0.08,
    diff: float = 0.18,
    alpha: float = 1.0,
    beta: float = 1.0,
    sample_every: int = 1,
) -> tuple[np.ndarray, tuple[int, int]]:
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x, y, indexing="ij")

    U = (
        0.8 * np.exp(-25.0 * ((Xg + 0.35) ** 2 + (Yg + 0.15) ** 2)) -
        0.6 * np.exp(-18.0 * ((Xg - 0.28) ** 2 + (Yg - 0.22) ** 2)) +
        0.1 * np.sin(2.0 * np.pi * Xg) * np.cos(2.0 * np.pi * Yg)
    ).astype(np.float32)

    snaps = [U.reshape(-1).copy()]

    for k in range(steps - 1):
        Lu = laplacian_periodic(U)
        reaction = alpha * U - beta * (U ** 3)
        U = U + dt * (diff * Lu + reaction)
        U = U.astype(np.float32, copy=False)

        if ((k + 1) % sample_every) == 0:
            snaps.append(U.reshape(-1).copy())

    X = np.stack(snaps, axis=0).astype(np.float32)
    return X, (nx, ny)


def plot_rd_demo(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    shape_hw: tuple[int, int],
    out_png: str | Path,
) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    h, w = shape_hw
    U0 = X_true[0].reshape(h, w)
    Ut = X_true[-1].reshape(h, w)
    Up = X_pred[-1].reshape(h, w)
    E = np.abs(Ut - Up)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))

    im0 = axs[0].imshow(U0, origin="lower")
    axs[0].set_title("Initial field")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(Ut, origin="lower")
    axs[1].set_title("True final field")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(Up, origin="lower")
    axs[2].set_title("DMD final field")
    plt.colorbar(im2, ax=axs[2], fraction=0.046)

    im3 = axs[3].imshow(E, origin="lower")
    axs[3].set_title("|true - dmd|")
    plt.colorbar(im3, ax=axs[3], fraction=0.046)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return str(out_png)


def run_reaction_diffusion_demo(out_dir: Path, device: torch.device) -> None:
    X, shape_hw = make_reaction_diffusion_2d_data()
    dmd = fit_dmd_on_states(X, device)

    X1 = X[:-1]
    X2 = X[1:]
    X2_hat = predict_one_step_batch(dmd, X1)

    rollout = predict_from_initial(dmd, X[0], steps=X.shape[0] - 1)
    X_roll = np.vstack([X[0:1], rollout])

    one_step_err = rel_l2_err(X2_hat, X2)
    rollout_err = rel_l2_err(X_roll, X)

    eig_dmd, modes_dmd = compute_dmd_eigendecomposition(dmd)

    if rollout_err < 0.20:
        verdict = "GOOD APPROXIMATION FOR RAW DMD"
    elif rollout_err < 0.50:
        verdict = "PARTIAL / MODERATE APPROXIMATION"
    else:
        verdict = "RAW DMD STRUGGLES HERE"

    print("\n============ REACTION-DIFFUSION DMD CHECK ==========\n")
    print("SYSTEM: 2D nonlinear reaction-diffusion field")
    print(f"STATE SHAPE: {X.shape}  (each state = {shape_hw[0]}x{shape_hw[1]} field)")
    print(f"ONE-STEP REL L2 ERROR: {one_step_err:.8e}")
    print(f"ROLLOUT  REL L2 ERROR: {rollout_err:.8e}")
    print(f"VERDICT: {verdict}")

    out_png = plot_rd_demo(X, X_roll, shape_hw, out_dir / "reaction_diffusion_2d_dmd_check.png")
    print(f"PLOT SAVED TO: {out_png}")

    eig_png = plot_eigs_complex_plane(
        eig_dmd,
        out_dir / "reaction_diffusion_2d_dmd_eigenvalues.png",
        title="Reaction-Diffusion 2D DMD Eigenvalues",
    )
    print(f"EIGENVALUE PLOT SAVED TO: {eig_png}")

    modes_png = plot_field_modes(
        eig_dmd,
        modes_dmd,
        X[0],
        shape_hw,
        out_dir / "reaction_diffusion_2d_dmd_modes.png",
        top_k=6,
    )
    print(f"MODE PLOT SAVED TO: {modes_png}")


# =================================== MAIN ===================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone DMD sanity checks")
    parser.add_argument(
        "--system",
        type=str,
        default="both",
        choices=["linear2d", "rd2d", "both"],
        help="which system to test",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="out/dmd-checker",
        help="where to save plots",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force CPU",
    )
    args = parser.parse_args()

    if (not args.cpu) and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_dir = ensure_dir(args.outdir)

    print("DEVICE:", device)
    print("OUTDIR:", out_dir)

    if args.system in ("linear2d", "both"):
        run_linear_demo(out_dir, device)

    if args.system in ("rd2d", "both"):
        run_reaction_diffusion_demo(out_dir, device)


if __name__ == "__main__":
    main()