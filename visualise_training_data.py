# ========================= VISUALISE TRAINING DATA ==========================
from __future__ import annotations  # TYPE HINTS

# ================================= IMPORTS ==================================
from pathlib import Path  # PATH
import numpy as np  # NUMPY

try:  # PIL IMPORT
    from PIL import Image  # IMAGE
except Exception as e:  # FAIL
    raise RuntimeError("Pillow required: pip install pillow") from e  # ERROR

# ================================ HELPERS ===================================
def _ensure_dir(p: str | Path):  # MKDIR
    Path(p).parent.mkdir(parents=True, exist_ok=True)  # MAKE

def _cap_rows_z2(X: np.ndarray, r: float, re_idx: int = 0, im_idx: int = 1) -> np.ndarray:  # CAP Z
    Zr = X[:, re_idx].astype(np.float32, copy=False)  # RE
    Zi = X[:, im_idx].astype(np.float32, copy=False)  # IM
    mag = np.sqrt(Zr * Zr + Zi * Zi).astype(np.float32)  # |Z|
    bad = (~np.isfinite(mag)) | (mag > float(r))  # BAD
    if np.any(bad):  # APPLY
        mag_safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)  # SAFE
        s = (float(r) / mag_safe).astype(np.float32)  # SCALE
        Zr = np.where(bad, Zr * s, Zr).astype(np.float32)  # RE
        Zi = np.where(bad, Zi * s, Zi).astype(np.float32)  # IM
        X[:, re_idx] = Zr  # WRITE
        X[:, im_idx] = Zi  # WRITE
    return X  # OUT

# ============================ TEXT DUMP (HEAD) ===============================
def dump_training_text(  # TXT DUMP
    X: np.ndarray,  # DATA
    out_txt: str | Path,  # FILE
    *,  # KWONLY
    n_rows: int = 200,  # HOW MANY
    decimals: int = 6,  # PRECISION
):
    out_txt = Path(out_txt)  # PATH
    _ensure_dir(out_txt)  # MKDIR

    X = np.asarray(X, dtype=np.float32)  # FP32
    n = min(int(n_rows), int(X.shape[0]))  # LIMIT

    with open(out_txt, "w", encoding="utf-8") as f:  # OPEN
        f.write(f"#SHAPE {X.shape}\n")  # HEADER
        f.write(f"#DTYPE {X.dtype}\n")  # HEADER
        f.write("#FIRST_ROWS\n")  # HEADER
        fmt = f"{{:.{int(decimals)}f}}"  # FORMAT

        for i in range(n):  # ROWS
            row = ",".join(fmt.format(float(v)) for v in X[i])  # CSV LINE
            f.write(row + "\n")  # WRITE

    print(f"[OK] WROTE {out_txt}")  # LOG

# ============================ CSV DUMP (FULL) ================================
def dump_training_csv(  # CSV DUMP
    X: np.ndarray,  # DATA
    out_csv: str | Path,  # FILE
):
    out_csv = Path(out_csv)  # PATH
    _ensure_dir(out_csv)  # MKDIR
    X = np.asarray(X, dtype=np.float32)  # FP32
    np.savetxt(out_csv, X, delimiter=",", fmt="%.8f")  # WRITE
    print(f"[OK] WROTE {out_csv}")  # LOG

# ========================= QUICK IMAGE (Z-PLANE) =============================
def render_z_scatter_png(  # Z SCATTER
    X: np.ndarray,  # DATA
    out_png: str | Path,  # FILE
    *,  # KWONLY
    escape_r: float = 2.0,  # LIMIT
    max_points: int = 200000,  # SPEED
    re_idx: int = 0,  # WHERE IS ZR
    im_idx: int = 1,  # WHERE IS ZI
    size: int = 1024,  # IMAGE SIZE
):
    out_png = Path(out_png)  # PATH
    _ensure_dir(out_png)  # MKDIR

    X = np.asarray(X, dtype=np.float32)  # FP32
    X = X.copy()  # SAFE
    X = _cap_rows_z2(X, float(escape_r), re_idx, im_idx)  # CAP

    # SUBSAMPLE  # SPEED
    N = int(X.shape[0])  # COUNT
    if N > int(max_points):  # LIMIT
        idx = np.random.default_rng(0).choice(N, size=int(max_points), replace=False)  # PICK
        # d.p.?
        Xs = X[idx]  # SUB
    else:
        Xs = X  # ALL

    zr = Xs[:, re_idx]  # RE
    zi = Xs[:, im_idx]  # IM

    # MAP TO PIXELS  # [-R,R] -> [0,S-1]
    R = float(escape_r)  # R
    S = int(size)  # SIZE
    img = np.zeros((S, S), dtype=np.uint8)  # CANVAS

    x = ((zr + R) / (2.0 * R) * (S - 1)).astype(np.int32)  # X
    y = ((zi + R) / (2.0 * R) * (S - 1)).astype(np.int32)  # Y

    good = (x >= 0) & (x < S) & (y >= 0) & (y < S)  # IN BOUNDS
    x = x[good]  # FILTER
    y = y[good]  # FILTER

    img[y, x] = 255  # PLOT POINTS
    im = Image.fromarray(img, mode="L")  # GRAY
    im.save(out_png)  # SAVE
    print(f"[OK] WROTE {out_png}")  # LOG

# ====================== OPTIONAL: ORBIT INSPECTION ===========================
def dump_one_orbit(  # SINGLE ORBIT
    X: np.ndarray,  # DATA
    out_txt: str | Path,  # FILE
    *,  # KWONLY
    steps: int = 80,  # LENGTH
    stride: int = 1,  # PICK EVERY K
):
    out_txt = Path(out_txt)  # PATH
    _ensure_dir(out_txt)  # MKDIR
    X = np.asarray(X, dtype=np.float32)  # FP32

    # TAKE FIRST STEPS  # SIMPLE
    T = min(int(steps), int(X.shape[0]))  # LIMIT
    with open(out_txt, "w", encoding="utf-8") as f:  # OPEN
        f.write("#i,zr,zi\n")  # HEADER
        for i in range(0, T, int(stride)):  # LOOP
            zr = float(X[i, 0])  # RE
            zi = float(X[i, 1])  # IM
            f.write(f"{i},{zr:.8f},{zi:.8f}\n")  # LINE

    print(f"[OK] WROTE {out_txt}")  # LOG
def render_final_iteration_png(  # FINAL ITER ONLY
    X_grid: np.ndarray,  # (T,Ni,Nr,D)
    out_png: str | Path,  # FILE
    *,  # KWONLY
    escape_r: float = 2.0,  # LIMIT
    re_idx: int = 0,  # ZR
    im_idx: int = 1,  # ZI
    size: int = 1024,  # IMAGE SIZE
    max_points: int = 200000,  # SPEED
):
    # TAKE LAST TIME SLICE  # FINAL ITER
    X_last = X_grid[-1].reshape(-1, int(X_grid.shape[-1])).astype(np.float32, copy=False)  # (P,D)
    render_z_scatter_png(  # REUSE YOUR SCATTER
        X_last,
        out_png,
        escape_r=float(escape_r),
        max_points=int(max_points),
        re_idx=int(re_idx),
        im_idx=int(im_idx),
        size=int(size),
    )