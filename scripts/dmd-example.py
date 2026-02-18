# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import math
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TypeAlias, Any

import numpy as np
import rich.logging

import torch

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Uses a classic DMD to learn the dynamics of a quadratic iteration

    z <- (A z)^2 + c

Example:

    > {SCRIPT_PATH.name}
"""

# FIXME: does torch.Tensor have any shape / dtype typing? couldn't find anything..
Matrix: TypeAlias = torch.Tensor
Array1D: TypeAlias = torch.Tensor
Array2D: TypeAlias = torch.Tensor
Array3D: TypeAlias = torch.Tensor
Array4D: TypeAlias = torch.Tensor

# {{{ utils


def set_recommended_matplotlib() -> None:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    defaults: dict[str, dict[str, Any]] = {
        "figure": {
            "figsize": (16, 8),
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


@contextmanager
def axis(filename: pathlib.Path) -> Iterator[Any]:
    import matplotlib.pyplot as mp

    if not filename.suffix:
        ext = mp.rcParams["savefig.format"]
        filename = filename.with_suffix(f".{ext}")

    fig = mp.figure(num=1)
    ax = fig.gca()

    try:
        yield ax
    finally:
        log.info("Saving figure in '%s'.", filename)
        fig.savefig(filename)
        mp.close(fig)


@dataclass
class TicTocTimer:
    t_wall_start: float = field(default=0.0, init=False)
    t_wall: float = field(default=0.0, init=False)

    def tic(self) -> None:
        import time

        self.t_wall = 0.0
        self.t_wall_start = time.perf_counter()

    def toc(self) -> float:
        import time

        self.t_wall = time.perf_counter() - self.t_wall_start
        return self.t_wall

    def __str__(self) -> str:
        # NOTE: this matches how MATLAB shows the time from `toc`.
        return f"Elapsed time is {self.t_wall:.5f} seconds."


@contextmanager
def timeit(text: str) -> Iterator[None]:
    tt = TicTocTimer()
    tt.tic()

    try:
        yield None
    finally:
        tt.toc()
        log.info("%s (%s)", text, tt)


# }}}


# {{{ generate data


def determine_escape_radius(A: Matrix) -> float:
    assert A.shape[0] == A.shape[1]

    n = A.shape[0]
    sigma = torch.linalg.svdvals(A)

    return (2.0 * math.sqrt(n) / torch.min(sigma) ** 2).item()


def generate_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    nx: int = 128,
    ny: int = 128,
    dtype: Any = None,
) -> Array2D:
    x = torch.linspace(xmin, xmax, nx)
    y = torch.linspace(ymin, ymax, ny)

    result = x[None, :] + 1j * y[:, None]
    if dtype is not None and dtype != result.dtype:
        result = result.to(dtype)

    return result


def generate_dataset(
    A: Matrix,
    *,
    bbox: tuple[float, float, float, float],
    escape_radius: float = 100.0,
    resolution: int = 128,
    maxit: int = 128,
) -> tuple[Array4D, Array2D]:
    assert A.shape[0] == A.shape[1]

    # NOTE: promoting A to complex because einsum doesn't handle heterogeneous dtypes
    A = A.to(A.dtype.to_complex())
    d = A.shape[0]

    # generate grid
    c = generate_grid(*bbox, nx=resolution, ny=resolution, dtype=A.dtype)
    c = c[:, :, None]

    # iterate
    escape_radius_sqr = escape_radius**2
    result = torch.zeros((maxit, resolution, resolution, d), dtype=A.dtype)

    for i in range(1, maxit):
        # evolve
        z_prev = result[i - 1]
        z_new = torch.einsum("ij,klj->kli", A, z_prev) ** 2 + c

        # clip escaped values over escape radius
        # NOTE: we set it to `R + 1 / sqrt(d)` so that `norm2(z) == R + 1`
        escaped = torch.sum(z_new.real**2 + z_new.imag**2, dim=-1) > escape_radius_sqr

        z_new[escaped, :] = (escape_radius + 1) / math.sqrt(d)

        result[i] = z_new

    return result, c.squeeze(-1)


# }}}


# {{{ predict


@dataclass(frozen=True)
class DMD:
    Ahat: Array2D

    U: Array2D
    S: Array1D
    V: Array2D

    @property
    def dtype(self) -> torch.dtype:
        return self.Ahat.dtype

    @property
    def input_size(self) -> int:
        return self.U.shape[0]

    @property
    def rank(self) -> int:
        return self.Ahat.shape[0]

    def __call__(self, x: Array1D) -> Array1D:
        assert x.shape[0] == self.rank

        x = torch.einsum("ij,j...->i...", self.Ahat, x)
        return x


def dmd(
    X1: Array2D,
    X2: Array2D,
    *,
    rank: int | None = None,
    eps: float | None = None,
) -> DMD:
    U, S, Vh = torch.linalg.svd(X1, full_matrices=False)
    S = S.to(X1.dtype)

    if rank is not None:
        U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

    if eps is None:
        eps = 10 * torch.finfo(X1.dtype).eps

    mask = torch.abs(S) > eps
    U, Vh = U[:, mask], Vh[mask, :]

    # construct reduced order model
    Ahat = U.conj().T @ X2 @ Vh.conj().T @ torch.diag(1 / S[mask])
    assert Ahat.ndim == 2
    assert Ahat.shape[0] == Ahat.shape[1]

    return DMD(Ahat, U, S, Vh)


def predict(
    model: DMD,
    Z: Array4D,
    *,
    bbox: tuple[float, float, float, float],
) -> Array3D:
    maxit, nx, ny, d = Z.shape
    c = generate_grid(*bbox, nx=nx, ny=ny, dtype=model.dtype)
    c = c[None, :, :]

    # initialize the state to (z0, c)
    z = torch.zeros((d + 1, nx, ny), dtype=model.dtype)
    z[-1, :, :] = c

    # project the state back to the reduced state space
    zhat = torch.einsum("ij,j...->i...", model.U.conj().T, z)

    # iterate in the reduced state space for speeeeed!!!
    for i in range(1, maxit):
        zhat = model(zhat)

    # NOTE: remove the last element because that corresponds to c theoretically
    return torch.einsum("ij,j...->i...", model.U, zhat)[:-1]


# }}}


def main(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    dataidx: int = 5,
    escape_radius: float | None = 100.0,
    resolution: int = 128,
    maxit: int = 128,
    overwrite: bool = False,
) -> int:
    # {{{ vallidate inputs

    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    try:
        data = np.load(filename)
    except Exception:
        log.error("Failed to load data from file: '%s'.", filename)
        return 1

    if datakey not in data:
        log.error("File does not contain '%s' dataset: '%s'.", datakey, filename)
        return 1

    matrices = data[datakey]
    if matrices.ndim != 3:
        log.error("Expected a 3 dimensional array: %s.", matrices.shape)
        return 1

    mat = torch.from_numpy(matrices[dataidx])
    if mat.shape[0] != mat.shape[1]:
        log.error("Expected a square matrix: %s.", mat.shape)
        return 1

    if escape_radius is None:
        escape_radius = determine_escape_radius(mat)

    if outfile is None:
        outfile = (
            filename.parent
            / f"{filename.stem}-{dataidx:02d}-r{resolution:04d}-n{maxit:04d}.npz"
        )

    set_recommended_matplotlib()
    log.info("DMD")

    # }}}

    # {{{ generate dataset

    d = mat.shape[0]
    bbox = (-0.03, 0.03, -0.03, 0.03)

    with timeit(f"Generated dataset on {bbox}"):
        Z, c = generate_dataset(
            mat,
            bbox=bbox,
            resolution=resolution,
            escape_radius=escape_radius,
            maxit=maxit,
        )

    mask = torch.sum(Z[-1].real ** 2 + Z[-1].imag ** 2, dim=-1) < escape_radius**2
    log.info("Points:")
    log.info("Converged: %d", torch.sum(mask))
    log.info("Max: %.5e (radius %.5e)", torch.max(torch.abs(Z)), escape_radius)
    log.info("Mean: %.5e", torch.mean(torch.abs(Z)))

    with axis(outfile.with_suffix("")) as ax:
        ax.imshow(mask, extent=bbox, cmap="binary")

    # }}}

    # {{{ build dataset

    # NOTE: To build the dataset we
    #   1. Add c as an extra variable (DMD can't learn affine systems, so we help it)
    #   2. Remove all entries that escaped (no point there)
    #   3. Reshape and get the `X1, X2` blocks for DMD

    C = torch.tile(c[None, :, :, None], (maxit, 1, 1, 1))
    X = torch.concatenate([Z, C], axis=3)  # ty: ignore[no-matching-overload]
    X = X[:, mask, :]
    X = X.reshape(maxit, -1)

    X1 = X[:-1].reshape(-1, d + 1).T
    X2 = X[1:].reshape(-1, d + 1).T
    log.info("Dataset: %s", X1.shape)

    # }}}

    # {{{ train

    with timeit("Constructed DMD model"):
        model = dmd(X1, X2, eps=1.0e-6)

    with axis(outfile.with_suffix("").with_stem(f"{outfile.stem}-sigma")) as ax:
        ax.semilogy(model.S.real)
        ax.set_xlabel("n")
        ax.set_ylabel(r"$\sigma$")

    # }}}

    # {{{ forecast

    # FIXME:
    #   - Why is the "prediction" a lot slower than "generate_dataset"?
    #   - What's a good tolerance for the escaped points based on the error?
    #   - bumping the resolution to 128x128 seems to fail (everybody escapes?)

    with timeit("Predicting"):
        Zpredict = predict(model, Z, bbox=bbox)

    # NOTE: this is a linear operator, so we expect everybody to converge
    escaped = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=0) < escape_radius**2
    log.info("Escaped %d (points %d)", torch.sum(escaped), resolution * resolution)

    # NOTE: try to tell who escaped by looking at the error
    error = torch.linalg.norm(Z[-1].permute(2, 0, 1) - Zpredict, dim=0)

    with axis(outfile.with_suffix("").with_stem(f"{outfile.stem}-error")) as ax:
        im = ax.imshow(torch.log10(error), extent=bbox)
        ax.get_figure().colorbar(im, ax=ax)

    with axis(outfile.with_suffix("").with_stem(f"{outfile.stem}-predict")) as ax:
        ax.imshow(error < 1.0e-1, extent=bbox, cmap="binary")

    # }}}

    return 0


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
        "-r",
        "--resolution",
        type=int,
        default=64,
        help="Resolution (in the c complex plane) of the images",
    )
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
            resolution=args.resolution,
            overwrite=args.force,
        )
    )
