"""Solve the 2D incompressible Navier--Stokes equations.

The vorticity--streamfunction formulation is

    omega_t + psi_y omega_x - psi_x omega_y = nu Laplacian(omega),
    Laplacian(psi) = -omega,

with velocity ``(u, v) = (psi_y, -psi_x)``.  The vorticity transport uses an
IMEX step: the full advection expression is discretized with the conservative
Arakawa Jacobian while diffusion is implicit.  The Poisson equation then
recovers the new streamfunction from the new vorticity.

The Taylor--Green vortex on the unit torus is

    psi   = (U / k) sin(k x) sin(k y) exp(-2 nu k^2 t),
    omega = 2 U k sin(k x) sin(k y) exp(-2 nu k^2 t),

where ``k = 2 pi``.  Its nonlinear advection vanishes, so it provides an exact
solution for checking the complete transport and periodic Poisson solve.  The
example also contains a deterministic, periodic vortex-dipole initial
condition with easy and hard variants.  Running this module evolves that field
and writes a self-describing PyTorch checkpoint that can be inspected with
``scripts/visualize_navier_output.py``.  Pass ``--convergence`` to run the
Taylor--Green convergence study instead.
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeAlias

import torch
from sympy import Derivative as D
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
    arakawa_jacobian,
    centered,
    compile_solver,
    new,
    old,
)


GridSize: TypeAlias = int | tuple[int, int]

DEFAULT_VORTEX_DIPOLE_SEED = 0x6A09E667F3BCC909
_PERTURBATION_SEED_XOR = 0xD1B54A32D192ED03
_UINT64_MASK = (1 << 64) - 1


class VortexDipoleCase(Enum):
    """Available vortex-dipole initial conditions."""

    EASY = 2
    HARD = 3


@dataclass(frozen=True)
class Vortex:
    x0: float
    y0: float
    sigma_x: float
    sigma_y: float
    amplitude: float


class _MT19937_64:
    """Small implementation of the C++ ``std::mt19937_64`` engine."""

    _N = 312
    _M = 156
    _MATRIX_A = 0xB5026F5AA96619E9
    _UPPER_MASK = 0xFFFFFFFF80000000
    _LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int) -> None:
        self._state = [0] * self._N
        self._state[0] = seed & _UINT64_MASK
        for index in range(1, self._N):
            previous = self._state[index - 1]
            self._state[index] = (
                6364136223846793005 * (previous ^ (previous >> 62)) + index
            ) & _UINT64_MASK
        self._index = self._N

    def _twist(self) -> None:
        for index in range(self._N):
            following = self._state[(index + 1) % self._N]
            value = (self._state[index] & self._UPPER_MASK) | (
                following & self._LOWER_MASK
            )
            twisted = value >> 1
            if value & 1:
                twisted ^= self._MATRIX_A
            self._state[index] = self._state[(index + self._M) % self._N] ^ twisted
        self._index = 0

    def next_uint64(self) -> int:
        if self._index >= self._N:
            self._twist()
        value = self._state[self._index]
        self._index += 1
        value ^= (value >> 29) & 0x5555555555555555
        value ^= (value << 17) & 0x71D67FFFEDA60000
        value ^= (value << 37) & 0xFFF7EEE000000000
        value ^= value >> 43
        return value & _UINT64_MASK

    def uniform(self, low: float, high: float) -> float:
        # libstdc++'s generate_canonical<long double, 64> consumes one result
        # from a full-range 64-bit engine.  The final fields are float tensors,
        # so Python's binary64 conversion is the relevant precision here.
        unit = self.next_uint64() / float(1 << 64)
        return low + (high - low) * unit


def make_solver(cells: GridSize = 32, dt: float = 5e-4):
    """Compile an IMEX vorticity--streamfunction solver on the unit square."""

    t, x, y = symbols("t x y")
    nu = symbols("nu", positive=True)
    omega = Function("omega")(t, x, y)
    psi = Function("psi")(t, x, y)

    omega_laplacian = D(omega, (x, 2)) + D(omega, (y, 2))
    psi_laplacian = D(psi, (x, 2)) + D(psi, (y, 2))
    advection = arakawa_jacobian(omega, psi, axes=(x, y))

    problem = PDEProblem(
        equations=(
            PDEEquation(
                omega,
                Eq(
                    D(omega, t) + old(advection.target),
                    new(nu * omega_laplacian),
                ),
            ),
            PDEEquation(psi, Eq(new(psi_laplacian + omega), 0)),
        ),
        time=t,
        domain=CartesianDomain(Axis(x, 0, 1), Axis(y, 0, 1)),
        boundaries={
            omega: Periodic(),
            psi: Periodic(),
        },
        parameters=(ScalarParameter(nu),),
        name="navier-stokes-vorticity-streamfunction",
        discretization=TwoLevelDiscretization(
            dt=dt,
            differences=(
                centered(D(omega, (x, 2))),
                centered(D(omega, (y, 2))),
                centered(D(psi, (x, 2))),
                centered(D(psi, (y, 2))),
            ),
            expression_rules=advection,
        ),
    )
    grid_shape = (cells, cells) if isinstance(cells, int) else cells
    return compile_solver(
        problem,
        UniformGrid(grid_shape),
        options=CompileOptions(
            krylov=KrylovOptions(
                max_iterations=4 * cells,
                rtol_float64=1e-9,
                rtol_float32=1e-4,
                rtol_float16=1e-2,
                rtol_bfloat16=1e-1,
            )
        )
    )


def _normalize_grid_size(cells: GridSize) -> tuple[int, int]:
    if isinstance(cells, int):
        shape = (cells, cells)
    elif isinstance(cells, tuple) and len(cells) == 2:
        shape = cells
    else:
        raise TypeError("cells must be an integer or an (nx, ny) tuple")
    nx, ny = shape
    if isinstance(nx, bool) or isinstance(ny, bool) or not isinstance(nx, int) or not isinstance(ny, int):
        raise TypeError("grid dimensions must be integers")
    if nx < 8 or ny < 8:
        raise ValueError("nx and ny must both be at least 8")
    return nx, ny


def _normalize_vortex_case(case: VortexDipoleCase | str | int) -> VortexDipoleCase:
    if isinstance(case, VortexDipoleCase):
        return case
    if isinstance(case, str):
        try:
            return VortexDipoleCase[case.upper()]
        except KeyError as error:
            raise ValueError(f"unsupported vortex-dipole case {case!r}") from error
    try:
        return VortexDipoleCase(case)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported vortex-dipole case {case!r}") from error


def _easy_vortices() -> tuple[Vortex, ...]:
    return (
        Vortex(0.36, 0.52, 0.060, 0.055, +55.0),
        Vortex(0.64, 0.48, 0.060, 0.055, -55.0),
        Vortex(0.45, 0.35, 0.040, 0.035, -20.0),
        Vortex(0.55, 0.65, 0.040, 0.035, +20.0),
        Vortex(0.22, 0.76, 0.035, 0.050, +12.0),
        Vortex(0.78, 0.24, 0.035, 0.050, -12.0),
    )


def _hard_vortices(seed: int) -> tuple[Vortex, ...]:
    strength_scales = (45.0, 70.0, 105.0, 150.0, 220.0)
    size_scales = (0.050, 0.038, 0.028, 0.020, 0.014)
    random_engine = _MT19937_64(seed)
    vortices: list[Vortex] = []
    for strength_index, strength in enumerate(strength_scales):
        for size_index, sigma in enumerate(size_scales):
            for local_index in range(5):
                x_bin = (strength_index + local_index) % len(strength_scales)
                y_bin = (size_index + 2 * local_index) % len(size_scales)
                x0 = (x_bin + random_engine.uniform(0.18, 0.82)) / len(strength_scales)
                y0 = (y_bin + random_engine.uniform(0.18, 0.82)) / len(size_scales)
                anisotropy = random_engine.uniform(0.72, 1.28)
                amplitude_jitter = random_engine.uniform(0.85, 1.15)
                sign = 1.0 if (strength_index + size_index + local_index) % 2 == 0 else -1.0
                vortices.append(
                    Vortex(
                        x0,
                        y0,
                        sigma * anisotropy,
                        sigma / anisotropy,
                        sign * strength * amplitude_jitter,
                    )
                )
    return tuple(vortices)


def vortex_dipole_state(
    cells: GridSize,
    *,
    case: VortexDipoleCase | str | int = VortexDipoleCase.EASY,
    seed: int = DEFAULT_VORTEX_DIPOLE_SEED,
    amplitude_scale: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build the periodic, zero-mean ``(omega, psi)`` vortex-dipole state.

    Values are sampled at cell centers and the state layout is
    ``(field, x, y)``, matching :class:`thalassa.runtime.CompiledSolver`.
    ``psi`` is initially zero, matching the data-generator contract;
    :func:`run_vortex_dipole` initializes it before the first transport step.
    """

    nx, ny = _normalize_grid_size(cells)
    selected_case = _normalize_vortex_case(case)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not math.isfinite(amplitude_scale) or amplitude_scale <= 0.0:
        raise ValueError("amplitude_scale must be finite and positive")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype")

    # Generate on the CPU so the same seed is independent of accelerator RNGs.
    # Build coordinates and transcendental functions in FP32 when the solver
    # dtype is reduced precision; BF16 cannot represent every cell index on a
    # moderately sized grid, and CPU half transcendental coverage varies.
    generation_dtype = (
        torch.float32
        if dtype in (torch.float16, torch.bfloat16)
        else dtype
    )
    x = (torch.arange(nx, dtype=generation_dtype) + 0.5) / nx
    y = (torch.arange(ny, dtype=generation_dtype) + 0.5) / ny
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    omega = torch.zeros((nx, ny), dtype=generation_dtype)

    vortices = _easy_vortices() if selected_case is VortexDipoleCase.EASY else _hard_vortices(seed)
    for vortex in vortices:
        dx = xx - vortex.x0
        dy = yy - vortex.y0
        dx = torch.where(dx >= 0.5, dx - 1.0, dx)
        dx = torch.where(dx < -0.5, dx + 1.0, dx)
        dy = torch.where(dy >= 0.5, dy - 1.0, dy)
        dy = torch.where(dy < -0.5, dy + 1.0, dy)
        radius_squared = (dx / vortex.sigma_x).square() + (dy / vortex.sigma_y).square()
        omega.add_(vortex.amplitude * torch.exp(-0.5 * radius_squared))

    two_pi = 2.0 * math.pi
    background_amplitude = 4.0 if selected_case is VortexDipoleCase.EASY else 8.0
    perturbation_amplitude = 1.5 if selected_case is VortexDipoleCase.EASY else 3.0
    omega.add_(background_amplitude * torch.sin(two_pi * xx) * torch.sin(2.0 * two_pi * yy))

    phase_engine = _MT19937_64(seed ^ _PERTURBATION_SEED_XOR)
    for kx in range(1, 5):
        for ky in range(1, 5):
            phase = phase_engine.uniform(0.0, two_pi)
            weight = perturbation_amplitude / (kx * kx + ky * ky)
            omega.add_(weight * torch.sin(two_pi * (kx * xx + ky * yy) + phase))

    omega.mul_(amplitude_scale)
    omega.sub_(omega.mean())
    # Make the same final-cell correction as OASIS.  Flatten y-major so that
    # the corrected element corresponds to its (iy, ix) generation order.
    y_major = omega.transpose(0, 1).contiguous().view(-1)
    y_major[-1] = -y_major[:-1].sum()
    omega = y_major.view(ny, nx).transpose(0, 1).contiguous()
    psi = torch.zeros_like(omega)
    return torch.stack((omega, psi)).to(device=device, dtype=dtype)


def periodic_streamfunction_from_vorticity(omega: torch.Tensor) -> torch.Tensor:
    """Solve the centered periodic equation ``Laplacian(psi) = -omega``.

    The Fourier eigenvalues are those of the same second-order discrete
    Laplacian used by :func:`make_solver`, rather than the continuous operator.
    The arbitrary constant mode is fixed to zero.
    """

    if omega.ndim != 2:
        raise ValueError("omega must be a two-dimensional tensor")
    if not omega.dtype.is_floating_point:
        raise TypeError("omega must have a floating-point dtype")
    nx, ny = omega.shape
    if nx < 2 or ny < 2:
        raise ValueError("omega must contain at least two cells per axis")

    transform_dtype = omega.dtype
    if transform_dtype in (torch.float16, torch.bfloat16):
        transform_dtype = torch.float32
    working = omega.to(dtype=transform_dtype)
    kx = torch.arange(nx, device=omega.device, dtype=transform_dtype)
    ky = torch.arange(ny, device=omega.device, dtype=transform_dtype)
    eigenvalue_magnitude = (
        4.0 * nx**2 * torch.sin(math.pi * kx / nx).square()[:, None]
        + 4.0 * ny**2 * torch.sin(math.pi * ky / ny).square()[None, :]
    )
    eigenvalue_magnitude[0, 0] = 1.0
    psi_hat = torch.fft.fftn(working) / eigenvalue_magnitude
    psi_hat[0, 0] = 0.0
    return torch.fft.ifftn(psi_hat).real.to(dtype=omega.dtype)


def taylor_green_state(
    cells: int,
    time: float,
    viscosity: float,
    *,
    velocity: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample the exact Taylor--Green ``(omega, psi)`` state at cell centers."""

    generation_dtype = (
        torch.float32
        if dtype in (torch.float16, torch.bfloat16)
        else dtype
    )
    coordinates = (
        torch.arange(cells, device=device, dtype=generation_dtype) + 0.5
    ) / cells
    x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")
    wave_number = 2 * math.pi
    decay = math.exp(-2 * viscosity * wave_number**2 * time)
    mode = torch.sin(wave_number * x) * torch.sin(wave_number * y)
    psi = velocity * mode * decay / wave_number
    omega = 2 * velocity * wave_number * mode * decay
    return torch.stack((omega, psi)).to(dtype=dtype)


@dataclass(frozen=True)
class ConvergenceSample:
    cells: int
    relative_error: float
    observed_order: float | None


def convergence_study(
    cells: Sequence[int] = (16, 32, 64, 128,),
    *,
    dt: float = 5e-4,
    viscosity: float = 1e-2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[ConvergenceSample, ...]:
    """Run one fixed-time step and report worst per-field relative L2 errors."""

    samples: list[ConvergenceSample] = []
    previous_error: float | None = None
    for count in cells:
        solver = make_solver(count, dt).to(device=device, dtype=dtype)
        initial = taylor_green_state(
            count, 0.0, viscosity, device=device, dtype=dtype
        )
        result = solver.step(initial, viscosity)
        if not bool(result.converged.detach().cpu().item()):
            raise RuntimeError(f"Taylor--Green solve did not converge on {count}x{count} cells")

        expected = taylor_green_state(
            count, dt, viscosity, device=device, dtype=torch.float64
        )
        state_for_error = result.state.to(dtype=torch.float64)
        field_errors = torch.linalg.vector_norm(
            state_for_error - expected, dim=(-2, -1)
        ) / torch.linalg.vector_norm(expected, dim=(-2, -1))
        relative_error = float(torch.max(field_errors).detach().cpu().item())
        observed_order = (
            None
            if previous_error is None
            else math.log(previous_error / relative_error, 2)
        )
        samples.append(ConvergenceSample(count, relative_error, observed_order))
        previous_error = relative_error
    return tuple(samples)


def run_vortex_dipole(
    cells: int = 128,
    *,
    dt: float = 5e-4,
    steps: int = 100,
    viscosity: float = 1e-3,
    case: VortexDipoleCase | str | int = VortexDipoleCase.EASY,
    seed: int = DEFAULT_VORTEX_DIPOLE_SEED,
    amplitude_scale: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    retain_diagnostics: bool = False,
):
    """Generate and evolve a vortex dipole, returning the solver output.

    ``retain_diagnostics=True`` selects the eager result containing one
    diagnostic record per solve block and time step. The default remains the
    compile-friendly aggregate output.
    """

    if cells < 8:
        raise ValueError("cells must be at least 8")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if not math.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("viscosity must be finite and positive")

    solver = make_solver(cells, dt).to(device=device, dtype=dtype)
    initial = vortex_dipole_state(
        cells,
        case=case,
        seed=seed,
        amplitude_scale=amplitude_scale,
        device=device,
        dtype=dtype,
    )
    initial[1] = periodic_streamfunction_from_vorticity(initial[0])
    if retain_diagnostics:
        return solver.solve(
            initial,
            parameters={"nu": viscosity},
            steps=steps,
            raise_on_nonconvergence=False,
        )
    return solver(initial, viscosity, steps=steps)


def save_navier_output(
    path: str | Path,
    state: torch.Tensor,
    *,
    dt: float,
    steps: int,
    viscosity: float,
    case: VortexDipoleCase | str | int,
    seed: int,
    amplitude_scale: float,
    converged: bool,
    residual_norm: float,
    iterations: int,
    relative_residual: float | None = None,
    tolerance_ratio: float | None = None,
    precision_limited: bool = False,
    worst_residual_block: str | None = None,
    worst_residual_step: int | None = None,
) -> None:
    """Write a portable, self-describing checkpoint for the plotter."""

    if state.ndim != 3 or state.shape[0] != 2 or state.shape[1] != state.shape[2]:
        raise ValueError("state must have shape (2, cells, cells)")
    selected_case = _normalize_vortex_case(case)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "thalassa.navier-stokes.v1",
        "state": state.detach().to(device="cpu").contiguous(),
        "field_names": ("omega", "psi"),
        "axis_order": ("field", "x", "y"),
        "domain": ((0.0, 1.0), (0.0, 1.0)),
        "grid_shape": (int(state.shape[1]), int(state.shape[2])),
        "dt": float(dt),
        "steps": int(steps),
        "time": float(dt * steps),
        "viscosity": float(viscosity),
        "vortex_case": selected_case.name.lower(),
        "seed": int(seed),
        "amplitude_scale": float(amplitude_scale),
        "converged": bool(converged),
        # Kept for backward compatibility and detailed numerical debugging.
        "residual_norm": float(residual_norm),
        "relative_residual": (
            None if relative_residual is None else float(relative_residual)
        ),
        "tolerance_ratio": (
            None if tolerance_ratio is None else float(tolerance_ratio)
        ),
        "precision_limited": bool(precision_limited),
        "worst_residual_block": worst_residual_block,
        "worst_residual_step": worst_residual_step,
        "iterations": int(iterations),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="run the Taylor--Green convergence study instead of a vortex dipole",
    )
    parser.add_argument("--cells", type=int, default=128, help="cells per axis (default: 128)")
    parser.add_argument("--dt", type=float, default=5e-4, help="time-step size (default: 5e-4)")
    parser.add_argument("--steps", type=int, default=100, help="number of time steps (default: 100)")
    parser.add_argument("--viscosity", type=float, default=1e-2, help="kinematic viscosity (default: 1e-2)")
    parser.add_argument(
        "--case",
        choices=("easy", "hard"),
        default="easy",
        help="vortex-dipole complexity (default: easy)",
    )
    parser.add_argument(
        "--seed",
        type=lambda value: int(value, 0),
        default=DEFAULT_VORTEX_DIPOLE_SEED,
        help="deterministic integer seed; decimal or 0x-prefixed (default: OASIS seed)",
    )
    parser.add_argument(
        "--amplitude-scale",
        type=float,
        default=1.0,
        help="multiplier for the complete initial vorticity field (default: 1)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float64",
        help="solver precision (default: float64)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, or 'auto' to select CUDA when available (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("navier_vortex_dipole.pt"),
        help="output checkpoint (default: navier_vortex_dipole.pt)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = arguments.device
    selected_device = torch.device(device_name)
    selected_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[arguments.dtype]

    if arguments.convergence:
        print(f"Taylor--Green convergence on {selected_device}")
        print("cells      relative L2 error    order")
        for sample in convergence_study(
            device=selected_device, dtype=selected_dtype
        ):
            order = "-" if sample.observed_order is None else f"{sample.observed_order:.3f}"
            print(f"{sample.cells:4d}x{sample.cells:<4d} {sample.relative_error:17.9e}  {order}")
    else:
        print(
            f"Evolving {arguments.case} vortex dipole on {arguments.cells}x{arguments.cells} "
            f"for {arguments.steps} steps on {selected_device}"
        )
        result = run_vortex_dipole(
            arguments.cells,
            dt=arguments.dt,
            steps=arguments.steps,
            viscosity=arguments.viscosity,
            case=arguments.case,
            seed=arguments.seed,
            amplitude_scale=arguments.amplitude_scale,
            device=selected_device,
            dtype=selected_dtype,
            retain_diagnostics=True,
        )
        did_converge = result.converged
        if result.diagnostics:

            def diagnostic_max(diagnostic, field: str) -> float:
                value = getattr(diagnostic, field)
                return float(torch.max(value).detach().cpu().item())

            worst = max(
                result.diagnostics,
                key=lambda diagnostic: diagnostic_max(
                    diagnostic, "relative_residual"
                ),
            )
            residual = max(
                diagnostic_max(diagnostic, "residual_norm")
                for diagnostic in result.diagnostics
            )
            relative_residual = diagnostic_max(worst, "relative_residual")
            tolerance_ratio = max(
                diagnostic_max(diagnostic, "tolerance_ratio")
                for diagnostic in result.diagnostics
            )
            precision_limited = any(
                bool(torch.any(diagnostic.precision_limited).detach().cpu().item())
                for diagnostic in result.diagnostics
            )
            iteration_count = max(
                int(torch.max(diagnostic.iterations).detach().cpu().item())
                for diagnostic in result.diagnostics
            )
            worst_block = ",".join(worst.targets)
            worst_step = worst.step_index
        else:
            residual = 0.0
            relative_residual = 0.0
            tolerance_ratio = 0.0
            precision_limited = False
            iteration_count = 0
            worst_block = None
            worst_step = None
        save_navier_output(
            arguments.output,
            result.state,
            dt=arguments.dt,
            steps=arguments.steps,
            viscosity=arguments.viscosity,
            case=arguments.case,
            seed=arguments.seed,
            amplitude_scale=arguments.amplitude_scale,
            converged=did_converge,
            residual_norm=residual,
            iterations=iteration_count,
            relative_residual=relative_residual,
            tolerance_ratio=tolerance_ratio,
            precision_limited=precision_limited,
            worst_residual_block=worst_block,
            worst_residual_step=worst_step,
        )
        convergence_note = (
            " (precision-limited)" if precision_limited else ""
        )
        location = (
            ""
            if worst_block is None
            else f" [{worst_block}, step {worst_step}]"
        )
        print(
            f"Wrote {arguments.output} at t={arguments.dt * arguments.steps:g}; "
            f"converged={did_converge}{convergence_note}, "
            f"worst equation defect={relative_residual:.3e}{location}, "
            f"tolerance utilization={tolerance_ratio:.3g}x, "
            f"max iterations={iteration_count}"
        )
