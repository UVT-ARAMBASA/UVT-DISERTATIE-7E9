# losses.py
import torch  # TORCH IMPORT
import torch.nn.functional as F  # TORCH FUNCTIONS

# --------------------------- LOSS MODES ---------------------------
# 0 = MSE, 1 = MAE, 2 = HUBER, 3 = WEIGHTED_MSE
def make_reconstruction_loss(loss_mode: int = 0, **params):  # LOSS FACTORY
    beta = float(params.get("beta", 0.01))  # HUBER BETA
    eps = float(params.get("eps", 1e-8))  # NUMERIC EPS
    w_pow = float(params.get("w_pow", 1.0))  # WEIGHT POWER

    if loss_mode == 0:  # MSE
        def loss_fn(x_hat, x):  # LOSS
            return torch.mean((x_hat - x) ** 2)  # MSE
        return loss_fn  # RETURN

    if loss_mode == 1:  # MAE
        def loss_fn(x_hat, x):  # LOSS
            return torch.mean(torch.abs(x_hat - x))  # MAE
        return loss_fn  # RETURN

    if loss_mode == 2:  # HUBER / SMOOTHL1
        def loss_fn(x_hat, x):  # LOSS
            return F.smooth_l1_loss(x_hat, x, beta=beta)  # HUBER
        return loss_fn  # RETURN

    if loss_mode == 3:  # WEIGHTED MSE (DOWN-WEIGHT OUTLIERS)
        # WEIGHTS = 1 / (|x|^p + eps)  -> BIG VALUES GET LOWER WEIGHT
        def loss_fn(x_hat, x):  # LOSS
            w = 1.0 / (torch.abs(x) ** w_pow + eps)  # WEIGHTS
            return torch.mean(w * (x_hat - x) ** 2)  # WEIGHTED MSE
        return loss_fn  # RETURN

    raise ValueError(f"UNKNOWN loss_mode={loss_mode}")  # ERROR
