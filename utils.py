import torch  # TORCH IMPORT
import os  # OS IMPORT

def get_device():  # PICK DEVICE
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # RETURN GPU OR CPU

def to_tensor(x, device):  # CONVERT TO TENSOR ON DEVICE
    return torch.tensor(x, dtype=torch.float32, device=device)  # RETURN TENSOR

def save_model(model, path):  # SAVE MODEL WEIGHTS
    os.makedirs(os.path.dirname(path), exist_ok=True)  # MAKE DIR IF NEEDED
    torch.save(model.state_dict(), path)  # SAVE STATE DICT

def load_model(model, path, device):  # LOAD MODEL WEIGHTS
    model.load_state_dict(torch.load(path, map_location=device))  # LOAD WEIGHTS TO DEVICE
    return model  # RETURN MODEL
