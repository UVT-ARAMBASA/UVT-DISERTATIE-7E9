# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import numpy.linalg as la
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
This script converts MATLAB ``.mat`` files to the more Python-friendly numpy
``.npz``.

Example:

    > {SCRIPT_PATH.name} --variable-name matrices --outfile result.npz data.mat
"""

Array = np.ndarray[Any, np.dtype[Any]]


# {{{ convert MATLAB file

NORMALIZE_TO_ORD = {
    "1": 1,
    "2": 2,
    "inf": np.inf,
    "fro": "fro",
    "nuc": "nuc",
    "max": "max",
}


def mat_normalize(
    mat: Array,
    axis: tuple[int, ...],
    *,
    normalize: str | None = "1",
) -> Array:
    if normalize == "max":
        norm = np.max(mat, axis=axis, keepdims=True)
    elif normalize in NORMALIZE_TO_ORD:
        norm = la.norm(mat, axis=axis, ord=NORMALIZE_TO_ORD[normalize], keepdims=True)
    else:
        return mat

    return mat / norm


def convert_matlab(
    filename: pathlib.Path,
    outfile: pathlib.Path | None = None,
    *,
    mat_variable_names: list[str] | None = None,
    clip: tuple[float, float] | None = None,
    transpose: bool = False,
    normalize: str | None = None,
    absolute: bool = False,
    overwrite: bool = False,
) -> int:
    # {{{ sanitize inputs

    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    if outfile is None:
        outfile = filename.with_suffix(".npz")

    if not overwrite and outfile.exists():
        log.error("Output file exists (use --force): '%s'.", outfile)
        return 1

    if mat_variable_names is None:
        mat_variable_names = []

    if not mat_variable_names:
        log.error("Must pass in at least one variable (with --variable-name).")
        return 1

    if normalize is not None and normalize not in NORMALIZE_TO_ORD:
        log.error("Unknown normalize value '%s'.", normalize)
        return 1

    # }}}

    # {{{ read matrices

    from scipy.io import loadmat

    result = loadmat(filename)

    ret = 0
    matrices = {}
    for name in mat_variable_names:
        old_name = new_name = name
        if ":" in name:
            old_name, new_name = [part.strip() for part in name.split(":")]

        if old_name not in result:
            ret += 1
            log.error("File does not contain '%s' variable: '%s'.", old_name, filename)
            log.info(
                "Known variables are: '%s'",
                "', '".join(key for key in result if not key.startswith("__")),
            )
            continue

        mat = result[old_name]
        if not isinstance(mat, np.ndarray):
            ret += 1
            log.error("Object '%s' is not an ndarray: '%s'", old_name, type(mat).__name__)
            continue

        if mat.ndim == 2:
            if transpose:
                mat = mat.T

            matrices[new_name] = mat
        elif mat.ndim == 3:
            if transpose:
                mat = np.transpose(mat, (2, 0, 1))

            matrices[new_name] = mat
        else:
            # NOTE: should be easy to generalize, but it's not needed right now
            ret += 1
            log.error("Object '%s' has unsupported shape: %s", old_name, mat.shape)
            continue

        log.info("Read a matrix of size '%s' from '%s'.", mat.shape, old_name)

    if not matrices:
        log.warning("Failed to read any matrices from '%s'.", filename)
        return int(bool(ret))

    if ret:
        log.error("%d errors encountered.", ret)

    # }}}

    result = {}
    for name, mat in matrices.items():
        new_mat = mat

        if clip:
            new_mat = np.cli(new_mat, a_min=clip[0], a_max=clip[1])

        if absolute:
            new_mat = np.abs(new_mat)

        if mat.ndim == 2:
            new_mat = mat_normalize(new_mat, (1,), normalize=normalize)
        elif mat.ndim == 3:
            new_mat = mat_normalize(new_mat, (1, 2), normalize=normalize)
        else:
            raise AssertionError

        result[name] = new_mat

    np.savez(outfile, **result)
    log.info("Saved matries to file: '%s'.", outfile)

    return int(bool(ret))


# }}}


if __name__ == "__main__":
    import argparse

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=SCRIPT_LONG_HELP,
    )
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files (named '{basename}-XX')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing files",
    )
    parser.add_argument(
        "-n",
        "--variable-name",
        action="append",
        help="Name of the variable containing matrices in the .mat file: 'name[:new_name]'",
    )
    parser.add_argument(
        "-t",
        "--transpose",
        action="store_true",
        help="Transpose the matrix that is read from the .mat file",
    )
    parser.add_argument(
        "-z",
        "--normalize",
        choices=("1", "2", "inf", "fro", "nuc", "max"),
        default=None,
        help="Normalize the matrices by their norm",
    )
    parser.add_argument(
        "-a",
        "--abs",
        action="store_true",
        help="Take the absolute value of all matrix entries",
    )
    parser.add_argument(
        "--clip",
        type=float,
        nargs=2,
        default=None,
        help="Clip matrix entries to the maximum and minimum values given",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(
        convert_matlab(
            args.filename,
            mat_variable_names=args.variable_name,
            outfile=args.outfile,
            transpose=args.transpose,
            normalize=args.normalize,
            absolute=args.abs,
            clip=args.clip,
            overwrite=args.force,
        )
    )
