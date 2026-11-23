"""Benchmark the 2D porous-medium equation with a Barenblatt--Pattle solution.

The equation and exact solution are

    u_t = kappa Laplacian(u**m),                         m > 1,
    u(r, t) = tau**(-1/m) [C - a r**2 tau**(-1/m)]_+**(1/(m - 1)),
    a = (m - 1) / (4 kappa m**2),                       tau = t + t_s.

See ``examples/BarenblattPattle.md`` for the derivation.  This executable
example evolves the time-shifted profile, reports field, conservation, front,
peak, positivity, and implicit-solver diagnostics, and can
run a grid-refinement experiment with ``--convergence``.
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from sympy import Abs, Derivative as D
from sympy import Eq, Function, symbols

from thalassa import (
    Axis,
    CartesianDomain,
    CompileOptions,
    KrylovOptions,
    PDEEquation,
    PDEProblem,
    Periodic,
    ScalarParameter,
    TwoLevelDiscretization,
    UniformGrid,
    centered,
    compile_solver,
    new,
    old,
)
from thalassa.runtime import SolveResult


# The defaults are deliberately a moderate rather than a nearly linear test:
#
# * m=2 makes the diffusion genuinely degenerate while giving a Lipschitz exact
#   profile at the moving front.  It is the standard first porous-medium case,
#   and users can still select any real m>1 with --exponent.
# * kappa=0.025, C=1, and t_s=0.1 put the initial support radius at 0.356 and
#   the peak at 3.162.  Values are therefore comfortably scaled in FP64/FP32.
# * on [-1, 1]^2 the radius reaches only 0.423 at T=0.1.  The exact solution is
#   consequently zero for at least 0.577 before every boundary; the periodic
#   closure cannot interact with the physical solution during the benchmark.
# * dt=5e-4 resolves the initial diffusion time scale with dt/t_s=0.005.  Over
#   200 steps the front expands by 18.9% and the peak falls by 29.3%, large
#   enough that front and peak diagnostics test dynamics rather than roundoff.
#   The one-step refinement study instead uses dt=1e-4 so its smooth-interior
#   spatial rate remains visible for more grids before temporal error dominates.
DEFAULT_EXPONENT = 2.0
DEFAULT_DIFFUSIVITY = 2.5e-2
DEFAULT_C = 1.0
DEFAULT_TIME_SHIFT = 1.0e-1
DEFAULT_HALF_WIDTH = 1.0
DEFAULT_DT = 5.0e-6
DEFAULT_CONVERGENCE_DT = 1.0e-6
DEFAULT_STEPS = 20000
DEFAULT_CELLS = 128
DEFAULT_CONVERGENCE_CELLS = (16, 32, 64, 128, 256, 512)
# Error over the complete domain measures how the Cartesian stencil captures
# the circular free boundary.  To measure the scheme where the exact solution
# is smooth, we also report errors on r <= 0.75 R(t).  A fixed fraction of the
# analytical support gives every refinement the same physical comparison region;
# an h-dependent exclusion band would quietly change the norm as the grid moves.
INTERIOR_RADIUS_FRACTION = 0.75


@dataclass(frozen=True)
class BarenblattParameters:
    """Physical and geometric parameters for the exact benchmark solution."""

    exponent: float = DEFAULT_EXPONENT
    diffusivity: float = DEFAULT_DIFFUSIVITY
    constant: float = DEFAULT_C
    time_shift: float = DEFAULT_TIME_SHIFT
    half_width: float = DEFAULT_HALF_WIDTH
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self) -> None:
        finite_values = (
            self.exponent,
            self.diffusivity,
            self.constant,
            self.time_shift,
            self.half_width,
            self.center_x,
            self.center_y,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("all Barenblatt parameters must be finite")
        if self.exponent <= 1.0:
            raise ValueError("exponent must be greater than one")
        if self.diffusivity <= 0.0:
            raise ValueError("diffusivity must be positive")
        if self.constant <= 0.0:
            raise ValueError("constant must be positive")
        if self.time_shift <= 0.0:
            raise ValueError("time_shift must be positive")
        if self.half_width <= 0.0:
            raise ValueError("half_width must be positive")
        if abs(self.center_x) >= self.half_width or abs(self.center_y) >= self.half_width:
            raise ValueError("the solution center must lie strictly inside the domain")

    @property
    def profile_coefficient(self) -> float:
        return (self.exponent - 1.0) / (
            4.0 * self.diffusivity * self.exponent**2
        )

    @property
    def exact_mass(self) -> float:
        return (
            4.0
            * math.pi
            * self.diffusivity
            * self.exponent
            * self.constant ** (self.exponent / (self.exponent - 1.0))
        )

    def support_radius(self, time: float) -> float:
        tau = time + self.time_shift
        if not math.isfinite(time) or tau <= 0.0:
            raise ValueError("time + time_shift must be finite and positive")
        return math.sqrt(self.constant / self.profile_coefficient) * tau ** (
            1.0 / (2.0 * self.exponent)
        )

    def peak(self, time: float) -> float:
        tau = time + self.time_shift
        if not math.isfinite(time) or tau <= 0.0:
            raise ValueError("time + time_shift must be finite and positive")
        return self.constant ** (1.0 / (self.exponent - 1.0)) * tau ** (
            -1.0 / self.exponent
        )

    def boundary_clearance(self, time: float) -> float:
        closest_boundary = min(
            self.half_width - abs(self.center_x),
            self.half_width - abs(self.center_y),
        )
        return closest_boundary - self.support_radius(time)


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Accuracy and physical-quality metrics for one numerical state."""

    relative_l1_error: float
    relative_l2_error: float
    relative_linf_error: float
    relative_interior_l1_error: float
    relative_interior_l2_error: float
    relative_interior_linf_error: float
    interior_radius: float
    numerical_mass: float
    relative_mass_error: float
    relative_mass_drift: float
    moment_radius: float
    relative_radius_error: float
    numerical_peak: float
    relative_peak_error: float
    minimum: float
    negative_mass_fraction: float


@dataclass(frozen=True)
class SolverMetrics:
    """Worst implicit-solve diagnostics over all blocks and time steps."""

    converged: bool
    breakdown: bool
    residual_norm: float
    relative_residual: float
    tolerance_ratio: float
    precision_limited: bool
    maximum_iterations: int
    worst_block: str | None
    worst_step: int | None


@dataclass(frozen=True)
class ConvergenceSample:
    cells: int
    metrics: BenchmarkMetrics
    observed_l2_order: float | None
    observed_interior_l2_order: float | None
    solver: SolverMetrics


def _validate_cells(cells: int) -> None:
    if isinstance(cells, bool) or not isinstance(cells, int):
        raise TypeError("cells must be an integer")
    if cells < 8:
        raise ValueError("cells must be at least 8")


def _validate_time_step(dt: float) -> None:
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")


def make_solver(
    cells: int = DEFAULT_CELLS,
    dt: float = DEFAULT_DT,
    *,
    exponent: float = DEFAULT_EXPONENT,
    half_width: float = DEFAULT_HALF_WIDTH,
):
    """Compile the linearly implicit porous-medium solver.

    At each step, the nonlinear flux potential is replaced by its tangent at
    the old state,

        (u_new)**m = m (u_old)**(m-1) u_new - (m-1) (u_old)**m + O(dt**2).

    This makes each backward-Euler step linear while retaining first-order
    temporal consistency.  It is substantially more robust under refinement
    than asking an unpreconditioned Newton/Krylov iteration to resolve the
    increasingly stiff, degenerate free-boundary system.  The positive-part
    extension equals ``u`` for the physical solution and keeps fractional
    powers real if roundoff creates a negative old value.

    Applying centered second derivatives to the complete linearized potential
    gives a conservative flux divergence, so periodic summation preserves
    discrete mass to Krylov tolerance.  Backward Euler avoids the explicit
    ``dt=O(h**2)`` stability restriction; the spatial stencil is second order
    wherever the weak solution is smooth.
    """

    _validate_cells(cells)
    _validate_time_step(dt)
    if not math.isfinite(exponent) or exponent <= 1.0:
        raise ValueError("exponent must be finite and greater than one")
    if not math.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("half_width must be finite and positive")

    t, x, y = symbols("t x y")
    kappa = symbols("kappa", positive=True)
    u = Function("u")(t, x, y)
    positive_old_u = (old(u) + Abs(old(u))) / 2
    linearized_power = (
        exponent * positive_old_u ** (exponent - 1.0) * new(u)
        - (exponent - 1.0) * positive_old_u**exponent
    )
    power_xx = D(linearized_power, (x, 2))
    power_yy = D(linearized_power, (y, 2))

    problem = PDEProblem(
        equations=PDEEquation(
            u, Eq(D(u, t), new(kappa * (power_xx + power_yy)))
        ),
        time=t,
        domain=CartesianDomain(
            Axis(x, -half_width, half_width),
            Axis(y, -half_width, half_width),
        ),
        # Periodicity is chosen over Dirichlet data because the exact solution
        # is identically zero near every boundary and periodic summation makes
        # discrete mass conservation exact.  It therefore behaves like both
        # homogeneous Dirichlet and no-flux data before the front reaches a wall.
        boundaries={u: Periodic()},
        parameters=(ScalarParameter(kappa),),
        name="barenblatt-pattle-porous-medium",
        discretization=TwoLevelDiscretization(
            dt=dt,
            differences=(
                centered(power_xx, accuracy=2),
                centered(power_yy, accuracy=2),
            ),
        ),
    )
    return compile_solver(
        problem,
        UniformGrid((cells, cells)),
        options=CompileOptions(
            # Each step uses matrix-free BiCGStab.  A grid-scaled cap is
            # generous for the Jacobi-preconditioned systems without hiding
            # pathological convergence behind an effectively unlimited solve.
            krylov=KrylovOptions(
                max_iterations=8 * cells,
                rtol_float64=1e-11,
                rtol_float32=5e-7,
            ),
        ),
    )


def cell_center_coordinates(
    cells: int,
    parameters: BarenblattParameters,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cell-center coordinate mesh in Thalassa's ``(x, y)`` order."""

    _validate_cells(cells)
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("the implicit benchmark supports float32 and float64")
    spacing = 2.0 * parameters.half_width / cells
    coordinates = (
        torch.arange(cells, device=device, dtype=dtype) + 0.5
    ) * spacing - parameters.half_width
    return torch.meshgrid(coordinates, coordinates, indexing="ij")


def barenblatt_state(
    cells: int,
    time: float,
    parameters: BarenblattParameters = BarenblattParameters(),
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample the exact ``u`` field at cell centers."""

    tau = time + parameters.time_shift
    if not math.isfinite(time) or tau <= 0.0:
        raise ValueError("time + time_shift must be finite and positive")
    x, y = cell_center_coordinates(
        cells, parameters, device=device, dtype=dtype
    )
    radius_squared = (x - parameters.center_x).square() + (
        y - parameters.center_y
    ).square()
    bracket = torch.clamp(
        parameters.constant
        - parameters.profile_coefficient
        * radius_squared
        * tau ** (-1.0 / parameters.exponent),
        min=0.0,
    )
    u = tau ** (-1.0 / parameters.exponent) * bracket.pow(
        1.0 / (parameters.exponent - 1.0)
    )
    return u.unsqueeze(0)


def _relative_error(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    value = numerator / denominator
    return float(value.detach().cpu().item())


def benchmark_metrics(
    state: torch.Tensor,
    *,
    time: float,
    parameters: BarenblattParameters,
    initial_mass: float,
) -> BenchmarkMetrics:
    """Compare a numerical state with every available analytical diagnostic.

    The front is estimated from the positive solution's second moment.  For a
    2D Barenblatt profile, ``<r^2> = R^2 / (q + 2)`` with
    ``q=1/(m-1)``.  This moment-equivalent radius is much less sensitive than a
    thresholded last-positive-cell estimate to cell-center sampling and tiny
    implicit-solve residuals at the free boundary.

    ``relative_mass_error`` compares cell-center quadrature with the analytical
    whole-space mass, whereas ``relative_mass_drift`` compares with the sampled
    initial mass.  Keeping both separates quadrature error from conservation.
    """

    if state.ndim != 3 or state.shape[0] != 1 or state.shape[1] != state.shape[2]:
        raise ValueError("state must have shape (1, cells, cells)")
    if not state.dtype.is_floating_point:
        raise TypeError("state must have a floating-point dtype")
    if not math.isfinite(initial_mass) or initial_mass <= 0.0:
        raise ValueError("initial_mass must be finite and positive")

    cells = int(state.shape[1])
    exact = barenblatt_state(
        cells,
        time,
        parameters,
        device=state.device,
        dtype=state.dtype,
    )
    numerical_u = state[0]
    exact_u = exact[0]
    error = numerical_u - exact_u
    relative_l1 = _relative_error(error.abs().sum(), exact_u.abs().sum())
    relative_l2 = _relative_error(
        torch.linalg.vector_norm(error), torch.linalg.vector_norm(exact_u)
    )
    relative_linf = _relative_error(error.abs().max(), exact_u.abs().max())

    # This fixed physical core excludes the deliberately nonsmooth free front.
    # It complements rather than replaces the whole-domain norm: the former
    # checks formal stencil accuracy and the latter checks front capture.
    x, y = cell_center_coordinates(
        cells, parameters, device=state.device, dtype=state.dtype
    )
    radius_squared = (x - parameters.center_x).square() + (
        y - parameters.center_y
    ).square()
    interior_radius = INTERIOR_RADIUS_FRACTION * parameters.support_radius(time)
    interior = radius_squared <= interior_radius**2
    if not bool(torch.any(interior).detach().cpu().item()):
        raise ValueError("the grid has no cell centers in the interior error region")
    interior_error = error[interior]
    interior_exact = exact_u[interior]
    relative_interior_l1 = _relative_error(
        interior_error.abs().sum(), interior_exact.abs().sum()
    )
    relative_interior_l2 = _relative_error(
        torch.linalg.vector_norm(interior_error),
        torch.linalg.vector_norm(interior_exact),
    )
    relative_interior_linf = _relative_error(
        interior_error.abs().max(), interior_exact.abs().max()
    )

    spacing = 2.0 * parameters.half_width / cells
    cell_area = spacing**2
    numerical_mass = float((cell_area * numerical_u.sum()).detach().cpu().item())
    relative_mass_error = abs(numerical_mass - parameters.exact_mass) / parameters.exact_mass
    relative_mass_drift = abs(numerical_mass - initial_mass) / initial_mass

    positive_u = torch.clamp(numerical_u, min=0.0)
    positive_mass_sum = positive_u.sum()
    if float(positive_mass_sum.detach().cpu().item()) <= 0.0:
        raise ValueError("the numerical state has no positive mass")
    q = 1.0 / (parameters.exponent - 1.0)
    moment_radius_tensor = torch.sqrt(
        (q + 2.0) * (positive_u * radius_squared).sum() / positive_mass_sum
    )
    moment_radius = float(moment_radius_tensor.detach().cpu().item())
    exact_radius = parameters.support_radius(time)
    relative_radius_error = abs(moment_radius - exact_radius) / exact_radius

    numerical_peak = float(numerical_u.max().detach().cpu().item())
    exact_peak = parameters.peak(time)
    relative_peak_error = abs(numerical_peak - exact_peak) / exact_peak
    minimum = float(numerical_u.min().detach().cpu().item())
    negative_mass_fraction = _relative_error(
        torch.clamp(-numerical_u, min=0.0).sum(), numerical_u.abs().sum()
    )

    return BenchmarkMetrics(
        relative_l1_error=relative_l1,
        relative_l2_error=relative_l2,
        relative_linf_error=relative_linf,
        relative_interior_l1_error=relative_interior_l1,
        relative_interior_l2_error=relative_interior_l2,
        relative_interior_linf_error=relative_interior_linf,
        interior_radius=interior_radius,
        numerical_mass=numerical_mass,
        relative_mass_error=relative_mass_error,
        relative_mass_drift=relative_mass_drift,
        moment_radius=moment_radius,
        relative_radius_error=relative_radius_error,
        numerical_peak=numerical_peak,
        relative_peak_error=relative_peak_error,
        minimum=minimum,
        negative_mass_fraction=negative_mass_fraction,
    )


def solver_metrics(result: SolveResult) -> SolverMetrics:
    """Reduce retained block diagnostics to worst-case scalar values."""

    if not result.diagnostics:
        return SolverMetrics(
            result.converged, False, 0.0, 0.0, 0.0, False, 0, None, None
        )

    def scalar_max(diagnostic, name: str) -> float:
        value = getattr(diagnostic, name)
        return float(torch.max(value).detach().cpu().item())

    worst = max(
        result.diagnostics,
        key=lambda diagnostic: scalar_max(diagnostic, "relative_residual"),
    )
    return SolverMetrics(
        converged=result.converged,
        breakdown=any(
            bool(torch.any(item.breakdown).detach().cpu().item())
            for item in result.diagnostics
        ),
        residual_norm=max(
            scalar_max(item, "residual_norm") for item in result.diagnostics
        ),
        relative_residual=scalar_max(worst, "relative_residual"),
        tolerance_ratio=max(
            scalar_max(item, "tolerance_ratio") for item in result.diagnostics
        ),
        precision_limited=any(
            bool(torch.any(item.precision_limited).detach().cpu().item())
            for item in result.diagnostics
        ),
        maximum_iterations=max(
            int(torch.max(item.iterations).detach().cpu().item())
            for item in result.diagnostics
        ),
        worst_block=",".join(worst.targets),
        worst_step=worst.step_index,
    )


def run_benchmark(
    cells: int = DEFAULT_CELLS,
    *,
    dt: float = DEFAULT_DT,
    steps: int = DEFAULT_STEPS,
    parameters: BarenblattParameters = BarenblattParameters(),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> SolveResult:
    """Evolve an exact initial profile and retain every solve diagnostic."""

    _validate_cells(cells)
    _validate_time_step(dt)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("the implicit solve supports float32 and float64")
    final_time = dt * steps
    if parameters.boundary_clearance(final_time) <= 0.0:
        raise ValueError(
            "the exact support reaches the boundary; enlarge the domain or shorten the run"
        )

    compiled = make_solver(
        cells,
        dt,
        exponent=parameters.exponent,
        half_width=parameters.half_width,
    ).to(device=device, dtype=dtype)
    initial = barenblatt_state(
        cells, 0.0, parameters, device=device, dtype=dtype
    )
    return compiled.solve(
        initial,
        parameters={"kappa": parameters.diffusivity},
        steps=steps,
        raise_on_nonconvergence=False,
    )


def convergence_study(
    cells: Sequence[int] = DEFAULT_CONVERGENCE_CELLS,
    *,
    dt: float = DEFAULT_CONVERGENCE_DT,
    parameters: BarenblattParameters = BarenblattParameters(),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[ConvergenceSample, ...]:
    """Repeat one fixed backward-Euler step on successively refined grids.

    This mirrors the Navier--Stokes Taylor--Green experiment: every grid starts
    from exact cell-center data and advances the same physical time.  Keeping
    dt fixed exposes spatial convergence until the common first-order temporal
    error becomes dominant; reporting all diagnostics makes that plateau
    visible instead of folding it into a single order estimate.  Unlike the
    smooth Taylor--Green field, the Barenblatt solution is only Lipschitz at the
    m=2 free boundary, so its global field order is not expected to equal the
    stencil's second-order accuracy; radius and peak convergence provide useful
    complementary checks.
    """

    if not cells:
        raise ValueError("cells must contain at least one grid size")
    samples: list[ConvergenceSample] = []
    previous_error: float | None = None
    previous_interior_error: float | None = None
    previous_cells: int | None = None
    for count in cells:
        _validate_cells(count)
        if previous_cells is not None and count <= previous_cells:
            raise ValueError("convergence grid sizes must be strictly increasing")
        result = run_benchmark(
            count,
            dt=dt,
            steps=1,
            parameters=parameters,
            device=device,
            dtype=dtype,
        )
        if not result.converged:
            raise RuntimeError(
                f"Barenblatt--Pattle solve did not converge on {count}x{count} cells"
            )
        initial = barenblatt_state(
            count, 0.0, parameters, device=device, dtype=dtype
        )
        spacing = 2.0 * parameters.half_width / count
        initial_mass = float((spacing**2 * initial[0].sum()).detach().cpu().item())
        metrics = benchmark_metrics(
            result.state,
            time=dt,
            parameters=parameters,
            initial_mass=initial_mass,
        )
        order = (
            None
            if previous_error is None or previous_cells is None
            else math.log(previous_error / metrics.relative_l2_error)
            / math.log(count / previous_cells)
        )
        interior_order = (
            None
            if previous_interior_error is None or previous_cells is None
            else math.log(
                previous_interior_error / metrics.relative_interior_l2_error
            )
            / math.log(count / previous_cells)
        )
        samples.append(
            ConvergenceSample(
                count,
                metrics,
                order,
                interior_order,
                solver_metrics(result),
            )
        )
        previous_error = metrics.relative_l2_error
        previous_interior_error = metrics.relative_interior_l2_error
        previous_cells = count
    return tuple(samples)


def _parse_cells(value: str) -> tuple[int, ...]:
    try:
        cells = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid sizes must be comma-separated integers") from error
    if not cells:
        raise argparse.ArgumentTypeError("at least one grid size is required")
    try:
        for count in cells:
            _validate_cells(count)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return cells


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="run the one-step grid-refinement experiment",
    )
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            "time step; defaults to 5e-4 for the full run and 1e-4 for "
            "--convergence"
        ),
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--exponent", type=float, default=DEFAULT_EXPONENT)
    parser.add_argument("--diffusivity", type=float, default=DEFAULT_DIFFUSIVITY)
    parser.add_argument("--constant", type=float, default=DEFAULT_C)
    parser.add_argument("--time-shift", type=float, default=DEFAULT_TIME_SHIFT)
    parser.add_argument("--half-width", type=float, default=DEFAULT_HALF_WIDTH)
    parser.add_argument(
        "--convergence-cells",
        type=_parse_cells,
        default=DEFAULT_CONVERGENCE_CELLS,
        metavar="N,N,...",
        help="grid sizes for --convergence (default: 32,64,128,256,512)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="implicit solver precision (default: float64)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, or 'auto' to select CUDA when available",
    )
    return parser.parse_args()


def _print_convergence(samples: Sequence[ConvergenceSample]) -> None:
    print(
        "cells      rel L2 all order  rel L2 interior order  "
        "mass drift  radius err    peak err  iters  tol/limit"
    )
    for sample in samples:
        metrics = sample.metrics
        order = "-" if sample.observed_l2_order is None else f"{sample.observed_l2_order:5.2f}"
        interior_order = (
            "-"
            if sample.observed_interior_l2_order is None
            else f"{sample.observed_interior_l2_order:5.2f}"
        )
        print(
            f"{sample.cells:4d}x{sample.cells:<4d} "
            f"{metrics.relative_l2_error:10.3e} {order:>5s} "
            f"{metrics.relative_interior_l2_error:15.3e} {interior_order:>5s} "
            f"{metrics.relative_mass_drift:10.3e} "
            f"{metrics.relative_radius_error:10.3e} "
            f"{metrics.relative_peak_error:10.3e} "
            f"{sample.solver.maximum_iterations:6d} "
            f"{sample.solver.tolerance_ratio:10.3e}"
        )


def _print_benchmark(
    metrics: BenchmarkMetrics,
    implicit: SolverMetrics,
    parameters: BarenblattParameters,
    final_time: float,
) -> None:
    exact_radius = parameters.support_radius(final_time)
    exact_peak = parameters.peak(final_time)
    location = (
        "none"
        if implicit.worst_block is None
        else f"{implicit.worst_block}, step {implicit.worst_step}"
    )
    print("Field error against exact cell-center samples")
    print(f"  relative L1                         {metrics.relative_l1_error:.9e}")
    print(f"  relative L2                         {metrics.relative_l2_error:.9e}")
    print(f"  relative Linf                       {metrics.relative_linf_error:.9e}")
    print(f"Interior field error (r <= {metrics.interior_radius:.6g})")
    print(f"  relative L1                         {metrics.relative_interior_l1_error:.9e}")
    print(f"  relative L2                         {metrics.relative_interior_l2_error:.9e}")
    print(f"  relative Linf                       {metrics.relative_interior_linf_error:.9e}")
    print("Physical diagnostics")
    print(f"  mass (numerical / analytical)       {metrics.numerical_mass:.9e} / {parameters.exact_mass:.9e}")
    print(f"  relative analytical mass error      {metrics.relative_mass_error:.9e}")
    print(f"  relative mass drift from t=0        {metrics.relative_mass_drift:.9e}")
    print(f"  moment radius (numerical / exact)   {metrics.moment_radius:.9e} / {exact_radius:.9e}")
    print(f"  relative radius error               {metrics.relative_radius_error:.9e}")
    print(f"  peak (numerical / exact)            {metrics.numerical_peak:.9e} / {exact_peak:.9e}")
    print(f"  relative peak error                 {metrics.relative_peak_error:.9e}")
    print(f"  minimum u                           {metrics.minimum:.9e}")
    print(f"  negative mass fraction              {metrics.negative_mass_fraction:.9e}")
    print("Implicit solver diagnostics")
    print(f"  converged / breakdown               {implicit.converged} / {implicit.breakdown}")
    print(f"  worst residual norm                 {implicit.residual_norm:.9e}")
    print(f"  worst relative residual             {implicit.relative_residual:.9e}")
    print(f"  maximum tolerance utilization       {implicit.tolerance_ratio:.9e}x")
    print(f"  precision limited                   {implicit.precision_limited}")
    print(f"  maximum Krylov iterations           {implicit.maximum_iterations}")
    print(f"  worst relative-residual location    {location}")


if __name__ == "__main__":
    arguments = _parse_args()
    selected_device = torch.device(
        "cuda" if arguments.device == "auto" and torch.cuda.is_available() else (
            "cpu" if arguments.device == "auto" else arguments.device
        )
    )
    selected_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[arguments.dtype]
    selected_parameters = BarenblattParameters(
        exponent=arguments.exponent,
        diffusivity=arguments.diffusivity,
        constant=arguments.constant,
        time_shift=arguments.time_shift,
        half_width=arguments.half_width,
    )
    selected_dt = (
        arguments.dt
        if arguments.dt is not None
        else (DEFAULT_CONVERGENCE_DT if arguments.convergence else DEFAULT_DT)
    )

    if arguments.convergence:
        print(
            f"Barenblatt--Pattle one-step convergence at dt={selected_dt:g} "
            f"on {selected_device} ({arguments.dtype})"
        )
        _print_convergence(
            convergence_study(
                arguments.convergence_cells,
                dt=selected_dt,
                parameters=selected_parameters,
                device=selected_device,
                dtype=selected_dtype,
            )
        )
    else:
        final_time = selected_dt * arguments.steps
        clearance = selected_parameters.boundary_clearance(final_time)
        print(
            f"Evolving m={arguments.exponent:g} on {arguments.cells}x{arguments.cells} "
            f"to t={final_time:g} on {selected_device} ({arguments.dtype}); "
            f"exact boundary clearance={clearance:.6f}"
        )
        result = run_benchmark(
            arguments.cells,
            dt=selected_dt,
            steps=arguments.steps,
            parameters=selected_parameters,
            device=selected_device,
            dtype=selected_dtype,
        )
        initial = barenblatt_state(
            arguments.cells,
            0.0,
            selected_parameters,
            device=selected_device,
            dtype=selected_dtype,
        )
        spacing = 2.0 * selected_parameters.half_width / arguments.cells
        initial_mass = float(
            (spacing**2 * initial[0].sum()).detach().cpu().item()
        )
        _print_benchmark(
            benchmark_metrics(
                result.state,
                time=final_time,
                parameters=selected_parameters,
                initial_mass=initial_mass,
            ),
            solver_metrics(result),
            selected_parameters,
            final_time,
        )
