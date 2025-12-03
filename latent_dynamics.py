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

    def predict(self, z0, steps=20):  # PREDICT LATENT TRAJECTORY
        preds = []  # STORE RESULTS
        z = z0.clone().to(self.device)  # COPY INITIAL Z TO DEVICE
        for _ in range(steps):  # LOOP STEPS
            z = self.A @ z  # APPLY DMD UPDATE
            preds.append(z.clone())  # SAVE STEP
        return torch.stack(preds)  # RETURN STACKED RESULT
