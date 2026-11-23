"""Solve the 2D heat equation implicitly and check convergence.

On the unit square, consider

    u_t = kappa (u_xx + u_yy),
    du/dn = 0,
    u(0, x, y) = 1 + cos(pi x) + 0.5 cos(2 pi x) cos(pi y).

The cosine factors are eigenfunctions of the Laplacian with homogeneous
Neumann boundary conditions.  The analytical solution is therefore

    u(t, x, y) = 1
        + exp(-kappa pi^2 t) cos(pi x)
        + 0.5 exp(-5 kappa pi^2 t) cos(2 pi x) cos(pi y).

The numerical method uses backward Euler in time and centered second
differences in space.  Running this module performs a grid-refinement study
against the analytical solution.  The time step is refined like h^2 so that
the first-order temporal and second-order spatial errors both decrease at
second order under refinement.
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from sympy import Derivative as D
from sympy import Eq, Function, symbols

from thalassa import (
    Axis,
    CartesianDomain,
    CompileOptions,
    HomogeneousNeumann,
    KrylovOptions,
    PDEEquation,
    PDEProblem,
    ScalarParameter,
    TwoLevelDiscretization,
    UniformGrid,
    centered,
    compile_solver,
    new,
)


DEFAULT_CONVERGENCE_CELLS = (16, 32, 64, 128, 256, 512, 1024)


def _validate_cells(cells: int) -> None:
    if isinstance(cells, bool) or not isinstance(cells, int):
        raise TypeError("cells must be an integer")
    if cells < 4:
        raise ValueError("cells must be at least 4")


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def make_solver(cells: int = 64, dt: float = 1e-3):
    """Compile a backward-Euler heat solver on the unit square."""

    _validate_cells(cells)
    _validate_positive(dt, "dt")

    t, x, y = symbols("t x y")
    kappa = symbols("kappa", positive=True)
    u = Function("u")(t, x, y)
    laplacian = D(u, (x, 2)) + D(u, (y, 2))
    problem = PDEProblem(
        equations=PDEEquation(u, Eq(D(u, t), new(kappa * laplacian))),
        time=t,
        domain=CartesianDomain(Axis(x, 0, 1), Axis(y, 0, 1)),
        boundaries={u: HomogeneousNeumann()},
        parameters=ScalarParameter(kappa),
        name="implicit-heat-neumann",
        discretization=TwoLevelDiscretization(
            dt=dt,
            differences=(
                centered(D(u, (x, 2))),
                centered(D(u, (y, 2))),
            ),
        ),
    )
    return compile_solver(
        problem,
        UniformGrid((cells, cells)),
        options=CompileOptions(
            krylov=KrylovOptions(
                max_iterations=max(16, cells),
                rtol_float64=1e-11,
                rtol_float32=1e-6,
            )
        ),
    )


def analytical_solution(
    cells: int,
    time: float,
    diffusivity: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample the analytical solution at cell centers."""

    _validate_cells(cells)
    if not math.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and nonnegative")
    _validate_positive(diffusivity, "diffusivity")

    generation_dtype = (
        torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    )
    coordinates = (
        torch.arange(cells, device=device, dtype=generation_dtype) + 0.5
    ) / cells
    x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")
    first_mode = math.exp(-diffusivity * math.pi**2 * time) * torch.cos(
        math.pi * x
    )
    mixed_mode = 0.5 * math.exp(
        -5.0 * diffusivity * math.pi**2 * time
    ) * torch.cos(2.0 * math.pi * x) * torch.cos(math.pi * y)
    return (1.0 + first_mode + mixed_mode).unsqueeze(0).to(dtype=dtype)


def run_heat(
    cells: int = 64,
    *,
    dt: float = 1e-3,
    steps: int = 10,
    diffusivity: float = 0.1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Evolve exact initial data with the implicit heat solver."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    _validate_positive(diffusivity, "diffusivity")
    solver = make_solver(cells, dt).to(device=device, dtype=dtype)
    initial = analytical_solution(
        cells, 0.0, diffusivity, device=device, dtype=dtype
    )
    return solver(initial, diffusivity, steps=steps)


@dataclass(frozen=True)
class ConvergenceSample:
    cells: int
    dt: float
    steps: int
    relative_error: float
    observed_order: float | None


def convergence_study(
    cells: Sequence[int] = DEFAULT_CONVERGENCE_CELLS,
    *,
    final_time: float = 1e-2,
    diffusivity: float = 0.1,
    time_step_scale: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[ConvergenceSample, ...]:
    """Compare grid-refined numerical states with the analytical solution.

    Each grid uses ``steps = ceil(final_time / (time_step_scale * h^2))`` and
    shortens the actual step slightly so that every run ends at exactly
    ``final_time``.
    """

    if not cells:
        raise ValueError("cells must contain at least one grid size")
    _validate_positive(final_time, "final_time")
    _validate_positive(diffusivity, "diffusivity")
    _validate_positive(time_step_scale, "time_step_scale")

    samples: list[ConvergenceSample] = []
    previous_error: float | None = None
    previous_cells: int | None = None
    for count in cells:
        _validate_cells(count)
        if previous_cells is not None and count <= previous_cells:
            raise ValueError("convergence grid sizes must be strictly increasing")

        target_dt = time_step_scale / count**2
        steps = math.ceil(final_time / target_dt)
        dt = final_time / steps
        result = run_heat(
            count,
            dt=dt,
            steps=steps,
            diffusivity=diffusivity,
            device=device,
            dtype=dtype,
        )
        if not bool(result.converged.detach().cpu().item()):
            raise RuntimeError(
                f"implicit heat solve did not converge on {count}x{count} cells"
            )

        expected = analytical_solution(
            count,
            final_time,
            diffusivity,
            device=device,
            dtype=torch.float64,
        )
        error = torch.linalg.vector_norm(
            result.state.to(dtype=torch.float64) - expected
        ) / torch.linalg.vector_norm(expected)
        relative_error = float(error.detach().cpu().item())
        observed_order = (
            None
            if previous_error is None or previous_cells is None
            else math.log(previous_error / relative_error)
            / math.log(count / previous_cells)
        )
        samples.append(
            ConvergenceSample(count, dt, steps, relative_error, observed_order)
        )
        previous_error = relative_error
        previous_cells = count

    return tuple(samples)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=int,
        nargs="+",
        default=DEFAULT_CONVERGENCE_CELLS,
        help="increasing grid sizes (default: 16 32 64 128)",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=1e-2,
        help="common comparison time (default: 1e-2)",
    )
    parser.add_argument(
        "--diffusivity",
        type=float,
        default=0.1,
        help="heat diffusivity (default: 0.1)",
    )
    parser.add_argument(
        "--time-step-scale",
        type=float,
        default=1.0,
        help="target dt / h^2 ratio (default: 1)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="solver precision (default: float64)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, or 'auto' to select CUDA when available (default: auto)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    device_name = (
        "cuda"
        if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu" if arguments.device == "auto" else arguments.device
    )
    selected_device = torch.device(device_name)
    selected_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[arguments.dtype]

    print(
        f"Implicit heat convergence at t={arguments.final_time:g} "
        f"on {selected_device}"
    )
    print("cells      steps           dt    relative L2 error    order")
    for sample in convergence_study(
        arguments.cells,
        final_time=arguments.final_time,
        diffusivity=arguments.diffusivity,
        time_step_scale=arguments.time_step_scale,
        device=selected_device,
        dtype=selected_dtype,
    ):
        order = "-" if sample.observed_order is None else f"{sample.observed_order:.3f}"
        print(
            f"{sample.cells:4d}x{sample.cells:<4d} {sample.steps:5d}  "
            f"{sample.dt:11.4e}  {sample.relative_error:17.9e}  {order}"
        )
