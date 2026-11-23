# Thalassa 2.0

Thalassa is a typed finite-difference PDE compiler whose primary artifact is a
reusable `torch.nn.Module`. Version 2 is a clean break from the former source
generator: there is no `PDESystem`, `pde_compile`, phase API, or generated
command-line solver compatibility layer.

The supported numerical core is intentionally focused:

- one to three cell-centered Cartesian axes;
- uniform grids and compile-time time steps;
- uniform scalar Dirichlet, homogeneous Neumann, or periodic conditions per field;
- two-level explicit, IMEX, implicit, and algebraic discretizations;
- exact finite-difference weights and custom exact stencils;
- exact whole-expression rules with locally scoped derivative stencils;
- matrix-free CG, projected CG, BiCGStab, and damped Newton solves;
- CPU/CUDA portability through normal PyTorch device and dtype behavior.

## Installation

```bash
python -m pip install .
```

Only SymPy and PyTorch are runtime dependencies. Test and lint tools are in the
`dev` extra:

```bash
python -m pip install -e '.[dev]'
pytest
```

Thalassa requires Python 3.10 or newer.

### Experimental Google TPU backend

The optional single-device TPU adapter uses matching PyTorch and PyTorch/XLA
2.8.0 packages on Python 3.12:

```bash
python -m pip install -e '.[tpu]'
```

`compile_tpu_solver(...)` defaults to BF16 and `torch_xla.device(0)`. It wraps
the normal v2 `CompiledSolver` in `torch.compile(backend="openxla",
fullgraph=True, dynamic=False)`; compilation remains lazy until the first
invocation. Inputs must already have the selected dtype and XLA device. TPU v1
supports direct, pointwise systems through variable 4x4, CG, projected CG, and
BiCGStab. It excludes nonlinear Newton/JVP, multi-core sharding, StableHLO
export, and domain decomposition. BF16 is recommended because TPUs support it
natively and it has substantially more range than FP16.

See the [v5e-8 runbook](docs/tpu_v5e_runbook.md), the
[benchmark harness](benchmarks/tpu_v5e.py), and PyTorch/XLA's
[static inference guidance](https://docs.pytorch.org/xla/release/r2.8/learn/eager.html).

## Example

This backward-Euler/centered-space heat solver uses a cell-centered grid. State
layout is always `(*batch, fields, *grid_shape)`.

```python
import torch
from sympy import Derivative as D
from sympy import Eq, Function, symbols

from thalassa import (
    Axis,
    CartesianDomain,
    PDEEquation,
    PDEProblem,
    ScalarParameter,
    TwoLevelDiscretization,
    UniformDirichlet,
    UniformGrid,
    centered,
    compile_solver,
    new,
)

t, x = symbols("t x")
nu = symbols("nu", positive=True)
u = Function("u")(t, x)

problem = PDEProblem(
    equations=PDEEquation(u, Eq(D(u, t), new(nu * D(u, (x, 2))))),
    time=t,
    domain=CartesianDomain(Axis(x, 0, 1)),
    boundaries={u: UniformDirichlet(0.0)},
    parameters=ScalarParameter(nu),
    discretization=TwoLevelDiscretization(
        dt=1e-3,
        differences=centered(D(u, (x, 2)), accuracy=2),
    ),
)

solver = compile_solver(problem, UniformGrid(256))
centers = (torch.arange(256, dtype=torch.float64) + 0.5) / 256
initial = torch.sin(torch.pi * centers)[None, :]  # fields, x
output = solver(initial, 0.1, steps=10)
assert output.converged
```

Collection-valued model inputs accept either one item directly or a sequence.
Tuples remain useful for systems with multiple equations, parameters, spatial
rules, or grid axes.

Fields and runtime inputs have deterministic declaration order in
`solver.metadata`. `solver.step(...)` and `solver(...)` return `SolveOutput`,
including convergence masks, residual norms, and iteration counts. The eager
`solver.solve(...)` wrapper accepts mappings by symbolic name, retains detailed
per-block diagnostics, and raises on non-convergence by default.

## Equation and time-level semantics

`old(expr)` evaluates all fields, sampled coefficients, and time references in
`expr` at step `n`; `new(expr)` evaluates them at `n+1`. Wrappers work at any
subexpression, for example `old(u) * new(D(u, x))`. Every first-order time
derivative that is not inside a wrapper is automatically lowered with backward
Euler to `(u[n+1] - u[n]) / dt`. Other time-dependent field, coefficient, and
time expressions require an explicit `old(...)` or `new(...)` annotation.

Each `PDEEquation` is the single source for both the annotated numerical
equation and its continuous form. `continuous_equation` and
`continuous_residual` erase all time-level annotations, and the latter is
serialized into `ProgramIR`. Spatial rules target exact derivative expressions,
so conservative rules such as `D(a * D(u, x), x)` can be declared directly.

Whole-expression rules keep the PDE's continuous expression separate from its
numerical approximation. `ExpressionRule(target, approximation, differences)`
matches an exact SymPy subexpression and lowers its approximation using the
bundled derivative rules, which are local to that approximation. Put the target
behind its own `old(...)` or `new(...)` wrapper so SymPy retains the expression
boundary; matching is exact rather than wildcard or algebraic subset matching.
For example, the classic second-order Arakawa Jacobian is self-contained:

```python
advection = arakawa_jacobian(omega, psi, axes=(x, y))

equation = PDEEquation(
    omega,
    Eq(D(omega, t) + old(advection.target), new(nu * omega_laplacian)),
)
discretization = TwoLevelDiscretization(
    dt=dt,
    differences=(
        centered(D(omega, (x, 2))),
        centered(D(omega, (y, 2))),
    ),
    expression_rules=advection,
)
```

`arakawa_jacobian(a, b, axes=(x, y))` uses the convention
`J(a, b) = a_x*b_y - a_y*b_x`. Its circulation, energy, and enstrophy
conservation guarantees apply to the uniform periodic closure. Other supported
boundary conditions still use the normal ghost-cell semantics but do not carry
those conservation guarantees.

For custom `ExpressionRule` objects, Thalassa validates symbolic support,
declarations, exact coverage, and every required derivative stencil. The rule
author remains responsible for consistency and formal accuracy of the supplied
approximation.

Compilation is immutable and deterministic. Its passes verify, discretize,
canonicalize residual blocks, schedule strongly connected components, characterize
operators, and lower to versioned `ProgramIR`. No randomized symbolic probing is
used.

## Boundaries and implicit solves

Values live at cell centers with spacing `(upper - lower) / cells`. Ghost cells
use odd reflection for Dirichlet data (`2g - u_mirror`) and even reflection for
homogeneous Neumann data. `Periodic()` instead wraps logical cell index `i` to
`i % cells` independently on every axis; the state contains no duplicated
endpoint. Extensions preserve axis order and output shape in 1D, 2D, and 3D.

A state boundary applies to every axis of that field. Sampled coefficients
default to homogeneous Neumann extension and can opt into matching periodic
semantics explicitly with `CoefficientField(a, boundary=Periodic())`. Analytic
coordinate-dependent expressions used under periodic stencils must themselves
be periodic across the declared bounds.

CG is selected only for operators whose symmetry and positive definiteness are
proven. Recognized Neumann and periodic graph-Laplacian blocks use projected CG
on the mean-zero subspace and reject incompatible right-hand sides. Every other
affine implicit block uses BiCGStab. Non-affine blocks use damped Newton with
`torch.func.jvp` Jacobian-vector products.

Iterative implicit paths support `float16`, `bfloat16`, `float32`, and
`float64`. Reduced-precision matvecs and solutions retain their dtype while
Krylov recurrences and convergence tests use FP32. Nonlinear implicit solves
remain restricted to FP32/FP64. V2 is forward-first and does not guarantee
gradient behavior.

Solver outputs retain `residual_norm` as a raw debugging quantity and expose
two dimensionless summaries. `relative_residual` is the worst equation defect
`||Ax-b|| / (||Ax|| + ||b||)`, evaluated independently for every block and
step. `tolerance_ratio` is the worst residual divided by that block's effective
convergence threshold, so values at or below one satisfy the stopping test.
`precision_limited` reports whether an FP16/BF16 solve required its roundoff
floor rather than satisfying the requested absolute/relative tolerance alone.

## Deterministic export

```python
from thalassa import export_python

export_python(solver, "heat_export.py")
```

The exported module contains serialized `ProgramIR`, imports
`thalassa.runtime`, and exposes both `solver` and `load_solver()`. It does not
duplicate numerical solver implementations.

See [`examples/burgers.py`](examples/burgers.py) for a 1D IMEX example and
[`examples/navier_stokes.py`](examples/navier_stokes.py) for the 2D
vorticity--streamfunction Navier--Stokes equations. The latter checks the
periodic solver against the Taylor--Green vortex and reports spatial convergence
from 32x32 through 4096x4096 cells. It can also generate and evolve seeded
periodic vortex-dipole cases:

```bash
python examples/navier_stokes.py --cells 128 --steps 100 --case easy \
  --output navier_vortex_dipole.pt
```

For the Navier--Stokes example, use `--dtype float16` or `--dtype bfloat16` to
run the matrix-free implicit solves on reduced-precision hardware. Krylov
reductions, recurrence scalars, and convergence diagnostics remain FP32, while
operator applications and the returned state use the selected dtype.

Install the `viz` extra and render the self-describing checkpoint with:

```bash
python -m pip install -e '.[viz]'
python scripts/visualize_navier_output.py navier_vortex_dipole.pt \
  --output navier_summary.png
```

[`examples/implicit_heat.py`](examples/implicit_heat.py) solves the 2D heat
equation with backward Euler and homogeneous Neumann conditions. Its initial
condition is a sum of cosine modes with a closed-form analytical evolution;
running the example reports a space--time refinement study against that
solution:

```bash
python examples/implicit_heat.py
```

[`examples/barenblatt_pattle.py`](examples/barenblatt_pattle.py) benchmarks the
nonlinear porous-medium equation against its compactly supported 2D
Barenblatt--Pattle solution. It reports whole-domain and smooth-interior field
errors, mass conservation, moving front and peak errors, positivity, and implicit-solver
quality. Run its grid-refinement study with:

```bash
python examples/barenblatt_pattle.py --convergence
```

## Deliberate v2 limits

V2 excludes multistage/higher-order time integration, adaptive or nonuniform
grids, per-axis or per-face mixed boundaries, Robin conditions, unstructured
or distributed meshes, complex dtypes, reduced-precision nonlinear implicit
solves, arbitrary runtime callbacks, and a differentiability guarantee.
