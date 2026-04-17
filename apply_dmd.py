# =============================== APPLY DMD ==================================
from __future__ import annotations  # ENABLE PY<3.11 TYPE HINTS

# ================================ IMPORTS ====================================
import numpy as np  # NUMERICAL ARRAYS
import torch  # TORCH CORE

from latent_dynamics import DMDDynamics  # YOUR DMD CLASS

# =============================== FITTING =====================================
def fit_dmd_on_arrays(X1: np.ndarray, X2: np.ndarray, *, device: torch.device) -> DMDDynamics:  # FIT DMD
    dmd = DMDDynamics(device=device)  # INIT MODEL
    dmd.fit(X1, X2)  # RUN FIT
    return dmd  # RETURN FITTED

# ============================== PREDICTION ===================================
@torch.no_grad()  # DISABLE GRADS
def predict_orbit(dmd: DMDDynamics, z0: torch.Tensor, *, steps: int) -> torch.Tensor:  # PREDICT TRAJ
    # returns (steps, dim)  # OUTPUT SHAPE NOTE
    return dmd.predict(z0, steps=steps)  # CALL PREDICT

# =============================== EXPORT ======================================
def export_dmd_matrix(dmd: DMDDynamics, out_npy: str, out_csv: str | None = None) -> None:  # SAVE A
    A = getattr(dmd, "A", None)  # GET MATRIX
    if A is None:  # CHECK FITTED
        raise RuntimeError("DMD is not fitted (A is None).")  # HARD FAIL

    A_cpu = A.detach().cpu().numpy()  # MOVE TO CPU
    np.save(out_npy, A_cpu)  # SAVE NPY
    if out_csv is not None:  # OPTIONAL CSV
        np.savetxt(out_csv, A_cpu, delimiter=",")  # SAVE CSV

def fit_dmd_from_latent_covariances(  # FIT FROM STREAMED COVS
    G: np.ndarray,  # Z1^T Z1
    H: np.ndarray,  # Z1^T Z2
    *,
    device: torch.device,  # DEVICE
) -> DMDDynamics:
    A_t = np.linalg.pinv(G) @ H  # SOLVE Z2 = Z1 A^T
    dmd = DMDDynamics(device=device)  # INIT
    dmd.A = torch.tensor(A_t.T, dtype=torch.float32, device=device)  # STORE A
    return dmd  # RETURN