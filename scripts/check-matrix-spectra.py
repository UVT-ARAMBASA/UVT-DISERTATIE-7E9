# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Look at the properties of the adjacency matrices.

Example:

    > {SCRIPT_PATH.name} task-emotion.npz
"""

Array1D = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
Array2D = np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]
DTypeLike = np.typing.DTypeLike


# {{{ utils


def set_recommended_matplotlib() -> None:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    defaults: dict[str, dict[str, Any]] = {
        "figure": {
            "figsize": (10, 10),
            "dpi": 300,
            "constrained_layout.use": True,
        },
        "text": {"usetex": True},
        "legend": {"fontsize": 20},
        "lines": {"linewidth": 2, "markersize": 5},
        "axes": {
            "labelsize": 28,
            "titlesize": 28,
            "grid": True,
            "grid.axis": "both",
            "grid.which": "both",
            # NOTE: preserve existing colors (the ones in "science" are ugly)
            "prop_cycle": mp.rcParams["axes.prop_cycle"],
        },
        "xtick": {"labelsize": 20, "direction": "inout"},
        "ytick": {"labelsize": 20, "direction": "inout"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    from contextlib import suppress

    with suppress(ImportError):
        import scienceplots  # noqa: F401

        mp.style.use(["science", "ieee"])

    for group, params in defaults.items():
        mp.rc(group, **params)


# }}}


# {{{ graph measures


def compute_weighted_degree(
    mat: Array2D,
) -> Array1D:
    """Compute the weighted degree (or strength) of each node in the graph."""
    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    return np.sum(mat, axis=1)


def compute_weighted_clustering_coefficient(
    mat: Array2D,
    *,
    eps: float | None = None,
    dtype: DTypeLike | None = None,
) -> Array1D:
    r"""Compute a per-node weighted clustering coefficient from [Barrat2004]_.

    .. math::

        c_i = \frac{1}{s_i (d_i - 1)} \sum_{j, k}^n
            \frac{1}{2} (W_{ij} + W_{ik}) A_{ij} A_{ik} A_{jk}

    .. [Barrat2004] A. Barrat, M. Barthélemy, R. Pastor-Satorras, A. Vespignani,
        *The Architecture of Complex Weighted Networks*,
        Proceedings of the National Academy of Sciences, Vol. 101, pp. 3747--3752, 2004,
        `doi:10.1073/pnas.0400087101 <https://doi.org/10.1073/pnas.0400087101>`__.
    """
    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    if eps is None:
        try:
            eps = np.sqrt(np.finfo(mat.dtype).eps)
        except ValueError:
            eps = 1.0e-8

    if eps <= 0.0:
        raise ValueError(f"'eps' must be positive: {eps}")

    A = (np.abs(mat) > eps).astype(dtype)
    strength = compute_weighted_degree(mat)
    degree = np.sum(A, axis=1)

    result = np.sum((mat * A) * (A @ A), axis=1)

    mask = (degree >= 2) & (np.abs(strength) >= eps)
    wcc = np.zeros(n, dtype=dtype)
    wcc[mask] = result[mask] / (strength[mask] * (degree[mask] - 1))

    return wcc


def compute_graph_disparity(
    mat: Array2D,
    *,
    eps: float | None = None,
    dtype: DTypeLike | None = None,
) -> Array1D:
    r"""Compute a per-node disparity measure from [Serrano2009]_.

    .. math::

        Y_i = \frac{1}{s_i^2} \sum_{j}^n W_{ij}^2,

    where :math:`s_i` is the weighted degree (see :func:`compute_weighted_degree`).
    This measure is similar to the Inverse Participation Ratio.

    .. [Serrano2009] M. Á. Serrano, M. Boguñá, A. Vespignani,
        *Extracting the Multiscale Backbone of Complex Weighted Networks*,
        Proceedings of the National Academy of Sciences, Vol. 106, pp. 6483--6488, 2009,
        `doi:10.1073/pnas.0808904106 <https://doi.org/10.1073/pnas.0808904106>`__.
    """
    n, m = mat.shape
    if n != m:
        raise ValueError(f"matrix not square: {mat.shape}")

    if eps is None:
        try:
            eps = np.sqrt(np.finfo(mat.dtype).eps)
        except ValueError:
            eps = 1.0e-8

    if eps <= 0.0:
        raise ValueError(f"'eps' must be positive: {eps}")

    strength = compute_weighted_degree(mat)
    mask = np.abs(strength) < eps
    strength[mask] = 1.0

    disparity = np.sum(mat**2, axis=1, dtype=dtype) / strength**2
    disparity[mask] = 0.0

    return disparity


# }}}


# {{{ main


def main(
    filename: pathlib.Path,
    *,
    datakey: str = "matrices",
    basename: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if not filename.exists():
        log.error("File does not exist: %s.", filename)
        return 1

    try:
        data = np.load(filename)
    except Exception:
        log.error("Failed to load data from file: '%s'.", filename)
        return 1

    if datakey not in data:
        log.error("File does not contain '%s' dataset: '%s'.", datakey, filename)
        return 1

    if basename is None:
        basename = filename.with_suffix("")

    matrices = data[datakey]
    if matrices.ndim != 3:
        log.error("Expected a 3 dimensional array: %s.", matrices.shape)
        return 1

    import matplotlib.pyplot as mp
    from matplotlib.colors import SymLogNorm

    set_recommended_matplotlib()
    ext = ".{}".format(mp.rcParams["savefig.format"])

    for i in range(matrices.shape[0]):
        mat = matrices[i]
        outfile = basename.with_stem(f"{basename.stem}-{i:02d}").with_suffix(ext)
        if not overwrite and outfile.exists():
            log.error("Output file exists (use --force to overwrite): %s", outfile)
            return 1

        # spectrum
        eigs = np.linalg.eigvals(mat)
        kappa = np.linalg.cond(mat)

        order = np.argsort(eigs.real)[::-1]
        eigs = eigs[order]

        # graph
        eps = min(2.0 * np.min(np.abs(eigs)), 1.0e-2)
        amat = np.abs(mat)

        strength = compute_weighted_degree(amat)
        clustering = compute_weighted_clustering_coefficient(amat, eps=eps)
        disparity = compute_graph_disparity(amat)
        degree = np.sum(amat > eps, axis=1, dtype=np.float64)

        nrows = 2
        ncols = 3
        fig, axes = mp.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        (ax11, ax12, ax13), (ax21, ax22, ax23) = axes

        # {{{ plot

        # matrix
        ax11.imshow(mat, norm=SymLogNorm(0.25), cmap="seismic")

        # eigenvalues: real
        ax12.plot(eigs.real, "o")
        ax12.axhline(0.0, color="k", lw=1, ls="--")

        ax12.set_ylabel(r"$\lambda_r$")
        ax12.set_ylim([-2, 40])
        ax12.set_title(rf"$\kappa = {kappa:.5e}$")

        # eigenvalues: magnitude
        ax13.semilogy(np.abs(eigs), "o")

        ax13.set_ylabel(r"$|\lambda|$")
        ax13.set_title(rf"$\lambda_{{min}} = {np.min(np.abs(eigs)):.5e}$")

        # node strength
        ax21.plot(strength, "o")
        ax21.set_ylim([0, 60])
        ax21.set_title("Strength")

        # node clustering coefficient
        ax22.plot(clustering, "o")
        ax22.set_ylim([0.8, 1.0])
        ax22.set_title("Clustering Coefficient")

        # node disparity
        ax23.plot(disparity, "o")
        ax23.plot(1.0 / degree, "kv", lw=1)
        ax23.set_ylim([0.0, 0.02])
        ax23.set_title("Disparity")

        # }}}

        fig.savefig(outfile)
        mp.close(fig)
        log.info("Saving matrix to file '%s'.", outfile)

    return 0


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
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(
        main(
            args.filename,
            basename=args.outfile,
            overwrite=args.force,
        )
    )
