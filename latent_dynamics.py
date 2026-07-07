import torch  # TORCH IMPORT
import numpy as np  # NUMPY IMPORT

class DMDDynamics:  # DMD CLASS
    def __init__(self, device="cpu"):  # INIT WITH DEVICE
        self.A = None  # DMD MATRIX
        self.device = device  # STORE DEVICE

    def fit(self, Z1, Z2, ridge: float = 1e-2):  # FIT DMD MODEL, RIDGE-REGULARISED
        Z1_t = torch.tensor(Z1, dtype=torch.float32, device=self.device)
        Z2_t = torch.tensor(Z2, dtype=torch.float32, device=self.device)
        N = Z1_t.shape[0]  # NUMBER OF SNAPSHOT PAIRS
        G = (Z1_t.T @ Z1_t) / N  # MEAN, NOT SUM -- KEEPS ridge MEANINGFUL AT ANY DATASET SIZE
        H = (Z1_t.T @ Z2_t) / N
        eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        A_t = torch.linalg.solve(G + ridge * eye, H)
        self.A = A_t.T
        return self.A

    def predict(self, z0, steps=20):  # PREDICT FUTURE STATES
        preds = []  # STORE PREDICTIONS
        z = z0.clone().to(self.device)  # CLONE AND MOVE TO DEVICE

        for _ in range(steps):  # LOOP OVER STEPS
            if z.ndim == 1:  # SINGLE VECTOR CASE
                z = self.A @ z  # APPLY A TO VECTOR
            else:  # BATCH CASE
                z = z @ self.A.T  # APPLY A TO BATCH
            preds.append(z.clone())  # SAVE STEP OUTPUT

        return torch.stack(preds)  # STACK OVER TIME