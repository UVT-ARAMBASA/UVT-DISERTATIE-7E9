# ================================ defines.py ================================
from __future__ import annotations  # ENABLE MODERN TYPE HINTS

# ================================ PATHS =====================================
from pathlib import Path  # PATH BASE

PROJECT_DIR = Path(__file__).resolve().parent  # PROJECT ROOT
DATASET_OUT_NPZ = str(PROJECT_DIR / "data-set" / "training_data_mandelbrot.npz")  # DATASET FILE
CHECKPOINT_DIR = str(PROJECT_DIR / "checkpoints")  # MODEL DIR
OUTPUT_DIR = str(PROJECT_DIR / "outputs")  # OUTPUT DIR
OUTPUT_IMAGE_PNG = str(PROJECT_DIR / "outputs" / "mandelbrot_reconstructed.png")  # PNG OUTPUT
OUTPUT_IMAGE_PLOT_PNG = str(PROJECT_DIR / "outputs" / "mandelbrot_reconstructed_plot.png")  # PLOT OUTPUT

# ============================== MANDELBROT BOX ==============================
# ZOOM OUT 4X  # KEEP GRID_N
C_RE_MIN = -0.03
C_RE_MAX =  0.03
C_IM_MIN = -0.03
C_IM_MAX =  0.03

GRID_N = 512  # IMAGE RESOLUTION (GRID_N x GRID_N)
MAX_ITERS = 80  # ITERATIONS PER C
ESCAPE_R = 2.0  # ESCAPE RADIUS  # ALSO USED AS CLAMP UPPER BOUND IN TRAINING

# ========================== TRAINING DATA SETTINGS ===========================
TRAIN_MAX_ITERS = 40  # TRAIN ITERS (T)
TRAIN_SEED = 0  # KEEP (NOT USED IN GRID MODE BUT OK)

# YOU ASKED: RESOLUTION ON C GRID FOR TRAINING
TRAIN_C_RE_N = 256  # RESOLUTION OF PARAMETER C ON REAL SCALE
TRAIN_C_IM_N = 256  # RESOLUTION OF PARAMETER C ON IMAGINARY SCALE

# =============================== AE SETTINGS =================================
LATENT_DIM = 8  # LATENT SIZE
AE_EPOCHS = 60  # TRAIN EPOCHS
AE_BATCH_SIZE = 256  # BATCH SIZE
AE_LR = 1e-3  # LEARNING RATE

# =============================== DMD SETTINGS ================================
DMD_RANK = None  # OPTIONAL TRUNCATION (NONE = FULL)

# ============================== RUNTIME FLAGS ================================
USE_CUDA_IF_AVAILABLE = True  # GPU FLAG
SHOW_PLOT = True  # DISPLAY FIGURE
SAVE_PLOT = True  # SAVE PLOT PNG
SAVE_IMAGE = True  # SAVE PIL PNG

USE_A_SEQUENCES = False  # TOGGLE DATA SOURCE
A_DATA_DIR = str(PROJECT_DIR / "data")  # FOLDER WITH task-emotion.npz + task-rest.npz