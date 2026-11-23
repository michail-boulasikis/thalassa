#!/usr/bin/env python3
"""Visualize a checkpoint written by ``examples/navier_stokes.py``.

The summary combines vorticity and streamfunction, velocity magnitude and
streamlines, a radial vorticity spectrum, and run diagnostics.  It is entirely
self-contained and does not use the OASIS or Fortran Navier repositories.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm


COLORS = {
    "blue": "#000080",
    "bronze": "#9c6114",
    "bronze_dark": "#894e11",
    "pink": "#e9c4c7",
    "light_blue": "#b9d3dc",
    "green": "#adcab8",
    "beige": "#d6d2c4",
    "grey": "#bfb8af",
    "black": "#2f2b28",
    "white": "#ffffff",
    "paper": "#faf9f5",
}

PLOT_STYLE = {
    "figure.facecolor": COLORS["paper"],
    "savefig.facecolor": COLORS["paper"],
    "axes.facecolor": COLORS["white"],
    "axes.edgecolor": COLORS["grey"],
    "axes.labelcolor": COLORS["black"],
    "axes.titlecolor": COLORS["black"],
    "text.color": COLORS["black"],
    "xtick.color": COLORS["black"],
    "ytick.color": COLORS["black"],
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.0,
    "axes.labelsize": 10.5,
    "axes.linewidth": 0.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="checkpoint produced by examples/navier_stokes.py",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("navier_summary.png"),
        help="output image (default: navier_summary.png)",
    )
    parser.add_argument(
        "--title",
        default="2-D Navier–Stokes vortex-dipole result",
        help="figure title",
    )
    parser.add_argument("--dpi", type=int, default=220, help="output resolution (default: 220)")
    parser.add_argument("--show", action="store_true", help="display the figure after saving")
    return parser.parse_args()


def load_navier_output(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and validate a Thalassa Navier--Stokes checkpoint."""

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.4 has weights_only, but retain a clear fallback.
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "thalassa.navier-stokes.v1":
        raise ValueError(f"{path} is not a supported Thalassa Navier--Stokes checkpoint")
    if payload.get("field_names") != ("omega", "psi"):
        raise ValueError(f"{path} has unexpected field names {payload.get('field_names')!r}")
    if payload.get("axis_order") != ("field", "x", "y"):
        raise ValueError(f"{path} has unexpected axis order {payload.get('axis_order')!r}")

    state = payload.get("state")
    if not isinstance(state, torch.Tensor) or state.ndim != 3 or state.shape[0] != 2:
        raise ValueError(f"{path} does not contain a state with shape (2, nx, ny)")
    if not bool(torch.all(torch.isfinite(state)).item()):
        raise ValueError(f"{path} contains non-finite field values")

    # Solver state is (field, x, y); image and NumPy conventions are (y, x).
    array = state.detach().to(dtype=torch.float64).numpy()
    omega = np.ascontiguousarray(array[0].T)
    psi = np.ascontiguousarray(array[1].T)
    metadata = {key: value for key, value in payload.items() if key != "state"}
    return omega, psi, metadata


def robust_symmetric_limit(field: np.ndarray, percentile: float = 99.5) -> float:
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(finite), percentile))
    if limit <= 0.0:
        limit = float(np.max(np.abs(finite)))
    return limit if limit > 0.0 else 1.0


def robust_upper_limit(field: np.ndarray, percentile: float = 99.5) -> float:
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(finite, percentile))
    if limit <= 0.0:
        limit = float(np.max(finite))
    return limit if limit > 0.0 else 1.0


def signed_contour_levels(field: np.ndarray, count: int = 13) -> np.ndarray:
    finite = field[np.isfinite(field)]
    if finite.size == 0 or float(np.min(finite)) == float(np.max(finite)):
        return np.array([], dtype=np.float64)
    limit = robust_symmetric_limit(field, 98.0)
    levels = np.linspace(-limit, limit, count)
    levels = levels[(levels > np.min(finite)) & (levels < np.max(finite))]
    return levels[np.abs(levels) > np.finfo(np.float64).eps * limit]


def periodic_velocity(
    psi: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``u = psi_y``, ``v = -psi_x``, and speed."""

    u = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * dy)
    v = -(np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * dx)
    return u, v, np.hypot(u, v)


def radial_vorticity_spectrum(
    omega: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = omega.shape
    omega_hat = np.fft.rfft2(omega - omega.mean(dtype=np.float64))
    power = np.abs(omega_hat) ** 2 / (nx * ny) ** 2
    kx = np.fft.rfftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    bins = np.floor(radius).astype(int)
    summed = np.bincount(bins.ravel(), weights=power.ravel())
    counts = np.bincount(bins.ravel())
    indices = np.arange(summed.size)
    valid = (indices > 0) & (counts > 0) & (summed > 0.0) & np.isfinite(summed)
    return indices[valid].astype(np.float64), summed[valid] / counts[valid]


def _colormaps() -> tuple[LinearSegmentedColormap, LinearSegmentedColormap]:
    signed = LinearSegmentedColormap.from_list(
        "lund_signed",
        [
            COLORS["blue"],
            COLORS["light_blue"],
            COLORS["paper"],
            COLORS["pink"],
            COLORS["bronze_dark"],
        ],
    )
    speed = LinearSegmentedColormap.from_list(
        "lund_speed",
        [
            COLORS["paper"],
            COLORS["light_blue"],
            COLORS["green"],
            COLORS["bronze"],
            COLORS["bronze_dark"],
        ],
    )
    return signed, speed


def _style_spatial_axis(axis: plt.Axes) -> None:
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.xaxis.set_major_locator(mticker.MaxNLocator(5))
    axis.yaxis.set_major_locator(mticker.MaxNLocator(5))
    for spine in axis.spines.values():
        spine.set_color(COLORS["grey"])


def _add_colorbar(figure: plt.Figure, axis: plt.Axes, image, label: str) -> None:
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
    colorbar.set_label(label)
    colorbar.formatter = mticker.ScalarFormatter(useMathText=True)
    colorbar.formatter.set_powerlimits((-2, 3))
    colorbar.update_ticks()


def make_figure(
    omega: np.ndarray,
    psi: np.ndarray,
    metadata: dict[str, Any],
    *,
    title: str,
) -> plt.Figure:
    """Create the four-panel field and spectrum summary."""

    if omega.shape != psi.shape or omega.ndim != 2:
        raise ValueError(f"field shapes differ: omega={omega.shape}, psi={psi.shape}")
    ny, nx = omega.shape
    domain = metadata.get("domain", ((0.0, 1.0), (0.0, 1.0)))
    x_bounds, y_bounds = domain
    length_x = float(x_bounds[1] - x_bounds[0])
    length_y = float(y_bounds[1] - y_bounds[0])
    dx = length_x / nx
    dy = length_y / ny
    x = x_bounds[0] + (np.arange(nx) + 0.5) * dx
    y = y_bounds[0] + (np.arange(ny) + 0.5) * dy
    extent = (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])

    u, v, speed = periodic_velocity(psi, dx, dy)
    radial_k, spectrum = radial_vorticity_spectrum(omega, dx, dy)
    omega_mean = float(omega.mean(dtype=np.float64))
    enstrophy = 0.5 * float(np.mean((omega - omega_mean) ** 2))
    kinetic_energy = 0.5 * float(np.mean(speed**2))
    circulation = omega_mean * length_x * length_y
    signed_map, speed_map = _colormaps()

    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
        figure.subplots_adjust(left=0.06, right=0.96, bottom=0.07, top=0.86, wspace=0.28, hspace=0.30)
        ax_omega, ax_psi, ax_speed, ax_spectrum = axes.ravel()
        figure.suptitle(title, y=0.975, fontsize=19, fontweight="bold", color=COLORS["blue"])
        relative_residual = metadata.get("relative_residual")
        if relative_residual is None:
            solve_quality = (
                f"legacy raw residual={float(metadata.get('residual_norm', 0.0)):.2e}"
            )
        else:
            solve_quality = f"equation defect={float(relative_residual):.2e}"
        if metadata.get("precision_limited", False):
            solve_quality += " (precision-limited)"
        subtitle = (
            f"{metadata.get('vortex_case', '?')} case  •  {nx}×{ny} cells  •  "
            f"t={float(metadata.get('time', 0.0)):.4g}  •  ν={float(metadata.get('viscosity', 0.0)):.3g}  •  "
            f"converged={metadata.get('converged', '?')}  •  {solve_quality}"
        )
        figure.text(0.5, 0.925, subtitle, ha="center", color=COLORS["black"], fontsize=10.5)

        omega_limit = robust_symmetric_limit(omega)
        image = ax_omega.imshow(
            omega,
            origin="lower",
            extent=extent,
            cmap=signed_map,
            norm=TwoSlopeNorm(vmin=-omega_limit, vcenter=0.0, vmax=omega_limit),
            interpolation="bilinear",
            aspect="equal",
        )
        levels = signed_contour_levels(psi, 15)
        if levels.size:
            ax_omega.contour(x, y, psi, levels=levels, colors=COLORS["black"], linewidths=0.6, alpha=0.7)
        ax_omega.set_title(r"A  Vorticity $\omega$ with streamfunction contours", loc="left", fontweight="bold")
        _style_spatial_axis(ax_omega)
        _add_colorbar(figure, ax_omega, image, r"$\omega$")

        psi_limit = robust_symmetric_limit(psi)
        image = ax_psi.imshow(
            psi,
            origin="lower",
            extent=extent,
            cmap=signed_map,
            norm=TwoSlopeNorm(vmin=-psi_limit, vcenter=0.0, vmax=psi_limit),
            interpolation="bilinear",
            aspect="equal",
        )
        levels = signed_contour_levels(omega, 11)
        if levels.size:
            ax_psi.contour(x, y, omega, levels=levels, colors=COLORS["black"], linewidths=0.55, alpha=0.75)
        ax_psi.set_title(r"B  Streamfunction $\psi$", loc="left", fontweight="bold")
        _style_spatial_axis(ax_psi)
        _add_colorbar(figure, ax_psi, image, r"$\psi$")

        image = ax_speed.imshow(
            speed,
            origin="lower",
            extent=extent,
            cmap=speed_map,
            norm=Normalize(vmin=0.0, vmax=robust_upper_limit(speed)),
            interpolation="bilinear",
            aspect="equal",
        )
        stride = max(1, min(nx, ny) // 42)
        if x[::stride].size > 1 and y[::stride].size > 1:
            ax_speed.streamplot(
                x[::stride],
                y[::stride],
                u[::stride, ::stride],
                v[::stride, ::stride],
                density=1.1,
                color=COLORS["black"],
                linewidth=0.5,
                arrowsize=0.7,
            )
        ax_speed.set_title(r"C  Velocity speed $|\mathbf{u}|$", loc="left", fontweight="bold")
        _style_spatial_axis(ax_speed)
        _add_colorbar(figure, ax_speed, image, r"$|\mathbf{u}|$")

        if radial_k.size:
            ax_spectrum.loglog(
                radial_k,
                spectrum,
                color=COLORS["blue"],
                linewidth=1.7,
                marker="o",
                markersize=3.2,
                markerfacecolor=COLORS["paper"],
            )
            ax_spectrum.grid(True, which="major", color=COLORS["beige"], alpha=0.7)
            ax_spectrum.grid(True, which="minor", color=COLORS["grey"], alpha=0.25)
        ax_spectrum.set_title("D  Radially averaged vorticity spectrum", loc="left", fontweight="bold")
        ax_spectrum.set_xlabel(r"spatial frequency $|k|$")
        ax_spectrum.set_ylabel(r"mean power $|\hat{\omega}'|^2$")
        metrics = (
            f"mean ω {omega_mean:+.2e}    circulation {circulation:+.2e}\n"
            f"enstrophy {enstrophy:.3e}    kinetic energy {kinetic_energy:.3e}\n"
            f"steps {metadata.get('steps', '?')}    max Krylov iterations {metadata.get('iterations', '?')}"
        )
        if metadata.get("tolerance_ratio") is not None:
            metrics += (
                f"\ntolerance utilization "
                f"{float(metadata['tolerance_ratio']):.3g}x"
            )
        ax_spectrum.text(
            0.98,
            0.98,
            metrics,
            transform=ax_spectrum.transAxes,
            ha="right",
            va="top",
            fontsize=9.2,
            bbox={"facecolor": COLORS["paper"], "edgecolor": COLORS["beige"], "alpha": 0.9},
        )
    return figure


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    omega, psi, metadata = load_navier_output(args.checkpoint)
    figure = make_figure(omega, psi, metadata, title=args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"Wrote {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
