import torch  # TORCH IMPORT
import numpy as np  # NUMPY IMPORT

class DMDDynamics:  # DMD CLASS
    def __init__(self, device="cpu"):  # INIT WITH DEVICE
        self.A = None  # DMD MATRIX
        self.device = device  # STORE DEVICE

    def fit(self, Z1, Z2):  # FIT DMD MODEL
        Z1_t = torch.tensor(Z1, dtype=torch.float32, device=self.device)  # TO TENSOR ON DEVICE
        Z2_t = torch.tensor(Z2, dtype=torch.float32, device=self.device)  # TO TENSOR ON DEVICE
        self.A = Z2_t.T @ torch.linalg.pinv(Z1_t.T)  # DMD FORMULA A = Z2 * Z1^+
        return self.A  # RETURN MATRIX

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


