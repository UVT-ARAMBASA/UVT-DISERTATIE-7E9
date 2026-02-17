# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
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

# FIXME: does torch.Tensor have any shape / dtype typing? couldn't find anything..
Matrix: TypeAlias = torch.Tensor
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


# }}}


# {{{ model


class ComplexMSELoss(nn.Module):
    def forward(self, yhat, y):
        return (yhat - y).abs().pow(2).mean()


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

        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype

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
    sigma = torch.linalg.svdvals(A)

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
        # NOTE: we set it to `R + 1 / sqrt(d)` so that `norm(z) == R + 1`
        escaped = torch.sum(z_new.real**2 + z_new.imag**2, dim=-1) > escape_radius_sqr
        z_new[escaped, :] = (escape_radius + 1) / np.sqrt(d)

        result[i] = z_new

    return result, c.squeeze(-1)


# }}}


# {{{ predict


def predict(
    model: nn.Module,
    *,
    bbox: tuple[float, float, float, float],
    resolution: int = 128,
    maxit: int = 128,
) -> Array3D:
    with torch.no_grad():
        c = generate_grid(*bbox, nx=resolution, ny=resolution, dtype=model.dtype)
        z = torch.zeros((*c.shape, model.input_size), dtype=model.dtype)  # ty: ignore[no-matching-overload]
        c = c[:, :, None]

        for i in range(1, maxit):
            z = model(z) + c

    return z


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

    # }}}

    # {{{ generate dataset

    d = mat.shape[0]
    bbox = (-0.03, 0.03, -0.03, 0.03)
    log.info("Generating dataset on %s", bbox)

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
    log.info("Max: %.5e (radius %.5e)", Z.abs().max(), escape_radius)
    log.info("Mean: %.5e", Z.abs().mean())

    with axis(outfile.with_suffix("")) as ax:
        ax.imshow(mask, extent=bbox, cmap="binary")

    # }}}

    # {{{ build torch dataset

    from torch.utils.data import DataLoader, TensorDataset

    # NOTE: we have an array of shape (maxit, n, n, d) and we want to
    #   * make our input/output be (Z_n, Z_{n + 1}) so that we can learn the dynamics
    #   * take only the converged points (based on mask from above)
    #   * squish everything down to use d as the "features"

    X = Z[:-1, mask, :].reshape(-1, d)
    C = c.unsqueeze(0).unsqueeze(-1).expand(maxit - 1, -1, -1, d)
    Y = (Z[1:, mask, :] - C[:, mask, :]).reshape(-1, d)
    log.info("Dataset size: %s", X.shape)

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # }}}

    # {{{ train

    from torch import optim
    from torch.nn.utils import clip_grad_norm_

    model = QuadraticNetwork(d, d, dtype=Z.dtype)
    criterion = ComplexMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    weightsfile = outfile.with_suffix(".pth")
    if weightsfile.exists():
        checkpoint = torch.load(weightsfile)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        losses = checkpoint["losses"]
    else:
        epochs = 50
        losses = torch.empty(epochs)

        for epoch in range(epochs):
            loss_epoch = 0.0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                # compute loss function
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)

                # compute gradient
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)

                # update weights
                optimizer.step()

                loss_epoch += loss.item()

            losses[epoch] = loss_epoch / len(dataloader)
            scheduler.step(losses[epoch])
            log.info("[%03d/%03d] loss %.5e", epoch, epochs, loss_epoch)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": epochs,
                "losses": losses,
            },
            weightsfile,
        )

    with axis(outfile.with_suffix("").with_stem(f"{outfile.stem}-loss")) as ax:
        ax.semilogy(losses)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

    # }}}

    # {{{ forecast

    Zpredict = predict(model, bbox=bbox, resolution=resolution, maxit=maxit)
    mask = torch.sum(Zpredict.real**2 + Zpredict.imag**2, dim=-1) < escape_radius**2

    with axis(outfile.with_suffix("").with_stem(f"{outfile.stem}-predict")) as ax:
        ax.imshow(mask, extent=bbox, cmap="binary")

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
