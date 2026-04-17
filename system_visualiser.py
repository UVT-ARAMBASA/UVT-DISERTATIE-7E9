from __future__ import annotations  # MODERN TYPES

import os  # OS
from pathlib import Path  # PATH

import numpy as np  # NUMPY

from data_loader import load_one_A_matrix  # LOAD A


def _fmt_float(x: float) -> str:  # FLOAT TEXT
    return f"{float(x):.12g}"  # SHORT FLOAT


def _safe_name(x: float) -> str:  # FILE PART
    s = f"{float(x):+.6f}"  # FIXED TEXT
    s = s.replace("+", "p").replace("-", "m").replace(".", "_")  # SAFE NAME
    return s  # RETURN


def _formula_text(cr: float, ci: float, d: int) -> str:  # FORMULA TEXT
    c_text = f"({_fmt_float(cr)} + {_fmt_float(ci)}j)"  # C TEXT

    lines = []  # OUT LINES
    lines.append("SYSTEM FOR THIS PARAMETER C")  # TITLE
    lines.append("")  # BLANK
    lines.append(f"d = {d}")  # DIM
    lines.append(f"c = {c_text}")  # C VALUE
    lines.append("")  # BLANK
    lines.append("ITERATION RULE")  # TITLE
    lines.append("Az = A @ z_n")  # MATRIX STEP
    lines.append(f"z_(n+1) = (Az)^2 + {c_text} * e1")  # ACTUAL RULE
    lines.append("")  # BLANK
    lines.append("COMPONENT FORM")  # TITLE
    lines.append(f"z_(n+1)[0] = (sum_j A[0,j] * z_n[j])^2 + {c_text}")  # FIRST COMP
    lines.append("z_(n+1)[k] = (sum_j A[k,j] * z_n[j])^2,  FOR k = 1,...,d-1")  # OTHER COMPS
    lines.append("")  # BLANK
    lines.append("INITIAL CONDITION")  # TITLE
    lines.append(f"z_0 = [{c_text}, 0, 0, ..., 0]^T")  # INIT
    lines.append("")  # BLANK
    lines.append("PLOTTING HELP")  # TITLE
    lines.append("Use the trajectory TXT/CSV file to plot Re(z_i), Im(z_i), |z_i|, or max_i |z_i| over n.")  # HELP
    return "\n".join(lines) + "\n"  # RETURN


def export_system_for_each_c(  # EXPORT ALL C
    *,  # KWONLY
    data_dir: str,  # DATA DIR
    source: str,  # SOURCE
    index: int,  # MATRIX INDEX
    c_re_min: float,  # RE MIN
    c_re_max: float,  # RE MAX
    c_im_min: float,  # IM MIN
    c_im_max: float,  # IM MAX
    c_re_n: int,  # RE RES
    c_im_n: int,  # IM RES
    max_iters: int,  # ITERS
    out_dir: str = "out/system_visualiser",  # OUT DIR
) -> None:  # NO RETURN
    A = load_one_A_matrix(data_dir=data_dir, source=source, index=index).astype(np.float32)  # LOAD A
    d = int(A.shape[0])  # DIM

    cr = np.linspace(float(c_re_min), float(c_re_max), int(c_re_n), dtype=np.float32)  # RE GRID
    ci = np.linspace(float(c_im_min), float(c_im_max), int(c_im_n), dtype=np.float32)  # IM GRID
    C_re, C_im = np.meshgrid(cr, ci, indexing="xy")  # GRID

    base = Path(out_dir)  # BASE
    traj_dir = base / "per_c_values"  # VALUES DIR
    form_dir = base / "per_c_formula"  # FORMULA DIR
    traj_dir.mkdir(parents=True, exist_ok=True)  # MAKE DIR
    form_dir.mkdir(parents=True, exist_ok=True)  # MAKE DIR

    for row in range(int(c_im_n)):  # ROW LOOP
        for col in range(int(c_re_n)):  # COL LOOP
            crv = float(C_re[row, col])  # THIS CR
            civ = float(C_im[row, col])  # THIS CI
            c = np.complex64(crv + 1j * civ)  # THIS C

            z = np.zeros((d,), dtype=np.complex64)  # INIT
            z[0] = c  # SEED FIRST

            stem = f"c_re_{_safe_name(crv)}__c_im_{_safe_name(civ)}"  # STEM
            txt_path = traj_dir / f"{stem}.txt"  # TXT PATH
            csv_path = traj_dir / f"{stem}.csv"  # CSV PATH
            formula_path = form_dir / f"{stem}_formula.txt"  # FORMULA PATH

            with open(txt_path, "w", encoding="utf-8") as f_txt, open(csv_path, "w", encoding="utf-8") as f_csv:  # OPEN FILES
                f_txt.write(f"# c = ({_fmt_float(crv)} + {_fmt_float(civ)}j)\n")  # HEADER
                f_txt.write("# n,max_abs_z,first_re,first_im\n")  # HEADER
                f_csv.write("n,max_abs_z,first_re,first_im\n")  # HEADER

                for n in range(int(max_iters)):  # TIME LOOP
                    abs_z = np.abs(z)  # ABS
                    max_abs_z = float(np.max(abs_z))  # MAX ABS
                    first_re = float(np.real(z[0]))  # FIRST RE
                    first_im = float(np.imag(z[0]))  # FIRST IM

                    line = f"{n},{max_abs_z:.9e},{first_re:.9e},{first_im:.9e}\n"  # LINE
                    f_txt.write(line)  # WRITE TXT
                    f_csv.write(line)  # WRITE CSV

                    Az = (A @ z).astype(np.complex64)  # APPLY A
                    z = (Az * Az).astype(np.complex64)  # SQUARE
                    z[0] = np.complex64(z[0] + c)  # ADD C FIRST ONLY

            with open(formula_path, "w", encoding="utf-8") as f_formula:  # OPEN FORMULA
                f_formula.write(_formula_text(crv, civ, d))  # WRITE FORMULA