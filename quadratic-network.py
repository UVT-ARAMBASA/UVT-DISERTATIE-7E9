# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeAlias, Any

import numpy as np
import rich.logging

import torch
import torch.nn as nn

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Trains a quadratic neural network.

Example:

    > {SCRIPT_PATH.name}
"""

Matrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]
Array2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.complexfloating[Any]]]
Array4D: TypeAlias = np.ndarray[
    tuple[int, int, int, int], np.dtype[np.complexfloating[Any]]
]

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


# }}}


# {{{ model


class QuadraticActivation(nn.Module):
    def forward(self, x):
        return x**2


class BiasLayer(nn.Module):
    def __init__(self, size: int, *, dtype: Any = None) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class QuadraticNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        dtype: Any = None,
    ) -> None:
        if output_size is None:
            output_size = input_size

        super().__init__()

        # NOTE: we know that our function is (A x)^2 + c, so just encode that
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False, dtype=dtype),
            QuadraticActivation(),
            BiasLayer(output_size, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# }}}

# {{{ generate data


def determine_escape_radius(A: Matrix) -> float:
    assert A.shape[0] == A.shape[1]

    n = A.shape[0]
    sigma = np.linalg.svdvals(A)

    return 2.0 * np.sqrt(n) / np.min(sigma) ** 2


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
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    result = x[None, :] + 1j * y[:, None]
    if dtype is not None:
        result = result.astype(dtype)

    return result


def generate_dataset(
    A: Matrix,
    *,
    bbox: tuple[float, float, float, float],
    escape_radius: float = 100.0,
    resolution: int = 128,
    maxit: int = 128,
) -> Array4D:
    assert A.shape[0] == A.shape[1]
    d = A.shape[0]
    dtype = (A.dtype.type(1.0) * 1j).dtype

    # generate grid
    c = generate_grid(*bbox, nx=resolution, ny=resolution, dtype=dtype)
    c = c[None, :, :]

    # make a mask to only update the points that haven't escaped
    mask = np.ones((resolution, resolution), dtype=np.bool)
    escape_radius_sqr = escape_radius**2

    # iterate
    result = np.full(
        (maxit, d, resolution, resolution), escape_radius / np.sqrt(d), dtype=dtype
    )
    result[0] = 0.0

    for i in range(1, maxit):
        if not np.any(mask):
            # NOTE: everybody escaped.. somehow??
            break

        z_prev = result[i - 1][:, mask]
        z_new = np.einsum("ij,jk->ik", A, z_prev) ** 2 + c[:, mask]

        escaped = np.sum(z_new.real**2 + z_new.imag**2, axis=0) > escape_radius_sqr

        # FIXME: only write back values that have no escaped
        result[i][:, mask] = z_new
        mask[mask] = ~escaped

    return result


# }}}


def main(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    datakey: str = "matrices",
    escape_radius: float | None = 100.0,
    resolution: int = 32,
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

    idx = 5
    mat = matrices[idx]
    if mat.shape[0] != mat.shape[1]:
        log.error("Expected a square matrix: %s.", mat.shape)
        return 1

    if escape_radius is None:
        escape_radius = determine_escape_radius(mat)

    if outfile is None:
        outfile = (
            filename.parent
            / f"{filename.stem}-{idx:02d}-r{resolution:04d}-n{maxit:04d}.npz"
        )

    # }}}

    # {{{ generate dataset

    # TODO: make this work directly in torch already, no point in using numpy
    bbox = (-0.03, 0.03, -0.03, 0.03)
    Z = generate_dataset(
        mat,
        bbox=bbox,
        resolution=resolution,
        escape_radius=escape_radius,
        maxit=maxit,
    )

    set_recommended_matplotlib()

    mask = (np.sum(np.abs(Z[-1]), axis=0) < escape_radius).astype(mat.dtype)
    log.info("Points: %d", np.sum(mask))

    with axis(outfile.with_suffix("")) as ax:
        ax.imshow(mask, extent=bbox, cmap="binary")

    # }}}

    # {{{ build torch dataset

    from torch.utils.data import DataLoader, TensorDataset

    # NOTE: we have an array of shape (maxit, d, n, n) and we want to
    #   * move the `d` dimension to the back -> this is the "feature vector"
    #   * stack the real and imaginary parts -> 2d feature vector
    #   * make our input/output be (Z_n, Z_{n + 1}) so that we can learn the dynamics

    # TODO:
    #   * normalize? not clear how.

    d = 2 * Z.shape[0]
    Znn = torch.from_numpy(Z)
    Znn = Znn.permute(0, 2, 3, 1)
    Znn = torch.cat([Znn.real, Znn.imag], dim=-1)

    Xnn = Znn[:-1, :, :, :].reshape(-1, d)
    Ynn = Znn[1:, :, :, :].reshape(-1, d)

    dataset = TensorDataset(Xnn, Ynn)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # }}}

    # {{{ train

    from torch import optim

    model = QuadraticNetwork(d, d, dtype=Znn.dtype)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1.0e-3)

    epochs = 50

    for epoch in range(epochs):
        loss = 0.0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            # compute loss function
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # compute gradient
            loss.backward()

            # update weights
            optimizer.step()

            loss += loss.item()

        log.info("[%03d/%03d] loss %.5e", epoch, epochs, loss)

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
            overwrite=args.force,
        )
    )