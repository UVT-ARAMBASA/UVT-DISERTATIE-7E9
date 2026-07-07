# ============================ quadratic_predictor.py ==========================
from __future__ import annotations  # MODERN TYPE HINTS

import numpy as np  # NUMPY
import torch  # TORCH CORE
import torch.nn as nn  # NN MODULE


# LEARNS z -> (A z)^2 DIRECTLY; c IS READ FROM THE INPUT AND ADDED BACK AS A
# FIXED, NON-LEARNED TERM (NOT A GLOBAL BIAS), SO ONE SET OF WEIGHTS WORKS
# ACROSS THE WHOLE C-GRID -- SAME TRICK AS commonlib.py:split_dataset/torch_predict
class QuadraticPredictor(nn.Module):  # QUADRATIC PREDICTOR CLASS
    def __init__(self, state_dim: int, *, rank: int | None = None, bias: bool = False) -> None:  # INIT
        super().__init__()  # CALL SUPER
        self.state_dim = int(state_dim)  # d

        if rank is None:  # FULL d x d COUPLING MATRIX
            self.A: nn.Module = nn.Linear(self.state_dim, self.state_dim, bias=bias)  # LINEAR LAYER
        else:  # LOW-RANK FACTORISATION A = U V, FOR LARGE d
            self.A = nn.Sequential(  # SEQ MODULE
                nn.Linear(self.state_dim, int(rank), bias=False),  # LINEAR LAYER U
                nn.Linear(int(rank), self.state_dim, bias=bias),  # LINEAR LAYER V
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # FORWARD PASS
        d = self.state_dim  # STATE DIM
        zr = x[:, 0:d]  # REAL PART
        zi = x[:, d:2 * d]  # IMAG PART
        cr = x[:, 2 * d:2 * d + 1]  # C REAL, KEPT AS-IS, NOT LEARNED
        ci = x[:, 2 * d + 1:2 * d + 2]  # C IMAG, KEPT AS-IS, NOT LEARNED

        wr = self.A(zr)  # Re(A z)
        wi = self.A(zi)  # Im(A z)

        zr_next = wr * wr - wi * wi + cr  # Re((A z)^2 + c)
        zi_next = 2.0 * wr * wi + ci  # Im((A z)^2 + c)

        return torch.cat([zr_next, zi_next, cr, ci], dim=1)  # SAME LAYOUT AS INPUT


def make_quadratic_predictor_pair(  # BUILD (encoder, decoder) DROP-IN PAIR
    state_dim: int, *, rank: int | None = None, bias: bool = False,
) -> tuple[nn.Module, nn.Module]:
    return QuadraticPredictor(state_dim, rank=rank, bias=bias), nn.Identity()  # DECODER IS IDENTITY


# ======================= SPECTRUM DIAGNOSTIC ==================================
# MIRRORS quadratic-network-torch.py: A_learned WON'T MATCH A ENTRYWISE, BUT
# ITS EIGENVALUES/SINGULAR VALUES SHOULD IF THE NETWORK LEARNED THE DYNAMICS
def plot_learned_vs_true_matrix_spectrum(  # SAVE EIG/SVD COMPARISON PNG
    A_true: np.ndarray,  # TRUE d x d MATRIX
    A_learned: torch.Tensor,  # LEARNED WEIGHT, E.G. q_enc.A.weight
    out_png,  # OUTPUT PATH
) -> str:
    import matplotlib.pyplot as plt  # PLOT
    from pathlib import Path  # PATH

    out_png = Path(out_png)  # PATH
    out_png.parent.mkdir(parents=True, exist_ok=True)  # MKDIR

    A = np.asarray(A_true, dtype=np.float64)  # TRUE
    A_hat = A_learned.detach().cpu().numpy().astype(np.float64)  # LEARNED

    A_eigs = np.linalg.eigvals(A)  # TRUE EIGS
    A_hat_eigs = np.linalg.eigvals(A_hat)  # LEARNED EIGS
    A_sigma = np.linalg.svd(A, compute_uv=False)  # TRUE SINGULAR VALUES
    A_hat_sigma = np.linalg.svd(A_hat, compute_uv=False)  # LEARNED SINGULAR VALUES

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # SIDE BY SIDE

    ax1.plot(A_eigs.real, A_eigs.imag, "o", label="A (true)")  # TRUE EIGS
    ax1.plot(A_hat_eigs.real, A_hat_eigs.imag, "x", label="A (learned)")  # LEARNED EIGS
    ax1.set_xlabel(r"$\lambda_r$")  # X LABEL
    ax1.set_ylabel(r"$\lambda_i$")  # Y LABEL
    ax1.set_title("Eigenvalues")  # TITLE
    ax1.legend()  # LEGEND

    ax2.plot(A_sigma, "o", label=r"$\sigma$(A) (true)")  # TRUE SVD
    ax2.plot(A_hat_sigma, "x", label=r"$\sigma$(A) (learned)")  # LEARNED SVD
    ax2.set_title("Singular values")  # TITLE
    ax2.legend()  # LEGEND

    fig.tight_layout()  # TIGHT
    fig.savefig(out_png, dpi=200)  # SAVE
    plt.close(fig)  # CLOSE
    return str(out_png)  # RETURN