"""Source-tree runtime benchmarks with warm-up and NVTX measurement ranges.

Examples::

    python benchmarks/runtime_workloads.py --workload burgers --mode eager
    python benchmarks/runtime_workloads.py --workload explicit_heat_2d --mode compiled
    python benchmarks/runtime_workloads.py --workload navier_stokes_2d --mode eager
    nsys profile --trace=cuda,nvtx -o thalassa-burgers --gpu-metrics-devices=cuda-visible \
        python benchmarks/runtime_workloads.py --workload burgers --mode eager
    nsys profile --trace=cuda,nvtx -o thalassa-burgers --gpu-metrics-devices=cuda-visible --force-overwrite=true \
        python -c 'import os, runpy, sys; os.environ.pop("LD_PRELOAD", None); sys.argv[0] = "benchmarks/runtime_workloads.py"; runpy.run_path("benchmarks/runtime_workloads.py", run_name="__main__")' \
        --workload explicit_heat_2d --mode compiled

The ``thalassa:<workload>:<mode>:measure`` range is the steady-state range to use
for CUDA kernel-launch and synchronization counts.  Run Nsight Compute against
the same range when collecting DRAM-byte metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import torch
from sympy import Derivative as D
from sympy import Eq, Function, symbols

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE.parent))

import thalassa  # noqa: E402
from examples.navier_stokes import (  # noqa: E402
    make_solver as make_navier_stokes_solver,
    taylor_green_state,
)

Workload = tuple[torch.nn.Module, torch.Tensor, tuple[torch.Tensor, ...]]
COMPILED_STEPS_PER_CALL = 4


class _StateOnlySolver(torch.nn.Module):
    """Expose only solver state so compilation can discard diagnostics."""

    def __init__(self, solver: torch.nn.Module) -> None:
        super().__init__()
        self.solver = solver

    def forward(
        self,
        state: torch.Tensor,
        *inputs: torch.Tensor,
        t0: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        return self.solver(state, *inputs, t0=t0, steps=steps).state


def _domain(*axes):
    return thalassa.CartesianDomain(
        *(thalassa.Axis(axis, 0, 1) for axis in axes)
    )


def _isotropic_cells(cells: int, dimensions: int) -> int:
    """Return the side length nearest to a requested total cell count."""

    if cells < 1:
        raise ValueError("cells must be positive")
    return max(1, round(cells ** (1 / dimensions)))


def _heat(
    cells: int,
    batch: int,
    dtype: torch.dtype,
    *,
    dimensions: int,
    implicit: bool,
) -> Workload:
    t = symbols("t")
    axes = symbols(f"x:{dimensions}")
    u = Function("u")(t, *axes)
    side = _isotropic_cells(cells, dimensions)
    laplacian_terms = tuple(D(u, (axis, 2)) for axis in axes)
    laplacian = sum(laplacian_terms)
    if implicit:
        dt = min(1e-3, 0.25 / (dimensions * side**2))
        right_hand_side = thalassa.new(laplacian)
    else:
        dt = min(1e-5, 10.0 / (dimensions * side**2))
        right_hand_side = thalassa.old(0.01 * laplacian)
    problem = thalassa.PDEProblem(
        (thalassa.PDEEquation(u, Eq(D(u, t), right_hand_side)),),
        t,
        _domain(*axes),
        {u: thalassa.UniformDirichlet()},
        discretization=thalassa.TwoLevelDiscretization(
            dt,
            tuple(thalassa.centered(term) for term in laplacian_terms),
        ),
    )
    shape = (side,) * dimensions
    solver = thalassa.compile_solver(problem, thalassa.UniformGrid(shape))
    return solver, torch.rand((batch, 1, *shape), dtype=dtype), ()


def explicit_heat(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    return _heat(cells, batch, dtype, dimensions=1, implicit=False)


def explicit_heat_2d(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    return _heat(cells, batch, dtype, dimensions=2, implicit=False)


def explicit_heat_3d(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    return _heat(cells, batch, dtype, dimensions=3, implicit=False)


def burgers(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    t, x = symbols("t x")
    nu = symbols("nu", positive=True)
    u = Function("u")(t, x)
    problem = thalassa.PDEProblem(
        (
            thalassa.PDEEquation(
                u,
                Eq(
                    D(u, t) + thalassa.old(u * D(u, x)),
                    thalassa.new(nu * D(u, (x, 2))),
                ),
            ),
        ),
        t,
        _domain(x),
        {u: thalassa.UniformDirichlet()},
        (thalassa.ScalarParameter(nu),),
        discretization=thalassa.TwoLevelDiscretization(
            2e-4,
            (
                thalassa.centered(D(u, x)),
                thalassa.centered(D(u, (x, 2))),
            ),
        ),
    )
    solver = thalassa.compile_solver(problem, thalassa.UniformGrid((cells,)))
    return (
        solver,
        torch.rand((batch, 1, cells), dtype=dtype),
        (torch.tensor(1e-3, dtype=dtype),),
    )


def implicit_heat(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    return _heat(cells, batch, dtype, dimensions=1, implicit=True)


def implicit_heat_2d(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    return _heat(cells, batch, dtype, dimensions=2, implicit=True)


def implicit_heat_3d(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    return _heat(cells, batch, dtype, dimensions=3, implicit=True)


def navier_stokes_2d(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    side = _isotropic_cells(cells, 2)
    solver = make_navier_stokes_solver(side)
    state = taylor_green_state(
        side, 0.0, 1e-2, dtype=dtype
    ).unsqueeze(0).expand(batch, -1, -1, -1).clone()
    return solver, state, (torch.tensor(1e-2, dtype=dtype),)


def implicit_advection(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    t, x = symbols("t x")
    u = Function("u")(t, x)
    problem = thalassa.PDEProblem(
        (
            thalassa.PDEEquation(
                u, Eq(D(u, t) + thalassa.new(D(u, x)), 0)
            ),
        ),
        t,
        _domain(x),
        {u: thalassa.UniformDirichlet()},
        discretization=thalassa.TwoLevelDiscretization(
            1e-3, (thalassa.centered(D(u, x)),)
        ),
    )
    solver = thalassa.compile_solver(problem, thalassa.UniformGrid((cells,)))
    return solver, torch.rand((batch, 1, cells), dtype=dtype), ()


def pointwise(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    t, x = symbols("t x")
    u = Function("u")(t, x)
    v = Function("v")(t, x)
    problem = thalassa.PDEProblem(
        (
            thalassa.PDEEquation(
                u, Eq(thalassa.new(u) + thalassa.new(v), 3)
            ),
            thalassa.PDEEquation(
                v, Eq(thalassa.new(u) - thalassa.new(v), 1)
            ),
        ),
        t,
        _domain(x),
        {u: thalassa.UniformDirichlet(), v: thalassa.UniformDirichlet()},
        discretization=thalassa.TwoLevelDiscretization(1, ()),
    )
    solver = thalassa.compile_solver(problem, thalassa.UniformGrid((cells,)))
    return solver, torch.zeros((batch, 2, cells), dtype=dtype), ()


def pointwise_variable_4x4(
    cells: int, batch: int, dtype: torch.dtype
) -> Workload:
    t, x = symbols("t x")
    fields = tuple(Function(f"u{index}")(t, x) for index in range(4))
    coefficients = tuple(
        tuple(Function(f"a{row}{column}")(x) for column in range(4))
        for row in range(4)
    )
    equations = tuple(
        thalassa.PDEEquation(
            fields[row],
            Eq(
                sum(
                    coefficients[row][column] * thalassa.new(fields[column])
                    for column in range(4)
                ),
                thalassa.old(fields[row]),
            ),
        )
        for row in range(4)
    )
    problem = thalassa.PDEProblem(
        equations,
        t,
        _domain(x),
        {field: thalassa.UniformDirichlet() for field in fields},
        coefficient_fields=tuple(
            thalassa.CoefficientField(coefficient)
            for row in coefficients
            for coefficient in row
        ),
        discretization=thalassa.TwoLevelDiscretization(1, ()),
    )
    solver = thalassa.compile_solver(problem, thalassa.UniformGrid((cells,)))
    coordinate = torch.linspace(0.0, 1.0, cells, dtype=dtype)
    matrix = torch.empty(cells, 4, 4, dtype=dtype)
    for row in range(4):
        for column in range(4):
            if row == column:
                matrix[:, row, column] = 5.0 + 0.2 * torch.sin(
                    (row + 1) * coordinate
                )
            else:
                matrix[:, row, column] = 0.15 * torch.cos(
                    (row + column + 1) * coordinate
                )
    inputs = tuple(
        matrix[:, row, column] for row in range(4) for column in range(4)
    )
    return solver, torch.rand((batch, 4, cells), dtype=dtype), inputs


def nonlinear(cells: int, batch: int, dtype: torch.dtype) -> Workload:
    t, x = symbols("t x")
    u = Function("u")(t, x)
    problem = thalassa.PDEProblem(
        (thalassa.PDEEquation(u, Eq(thalassa.new(u**3), 1)),),
        t,
        _domain(x),
        {u: thalassa.UniformDirichlet()},
        discretization=thalassa.TwoLevelDiscretization(1, ()),
    )
    solver = thalassa.compile_solver(problem, thalassa.UniformGrid((cells,)))
    return solver, torch.full((batch, 1, cells), 0.1, dtype=dtype), ()


BUILDERS: dict[str, Callable[[int, int, torch.dtype], Workload]] = {
    "explicit_heat": explicit_heat,
    "explicit_heat_2d": explicit_heat_2d,
    "explicit_heat_3d": explicit_heat_3d,
    "burgers": burgers,
    "implicit_heat": implicit_heat,
    "implicit_heat_2d": implicit_heat_2d,
    "implicit_heat_3d": implicit_heat_3d,
    "navier_stokes_2d": navier_stokes_2d,
    "implicit_advection": implicit_advection,
    "pointwise": pointwise,
    "pointwise_variable_4x4": pointwise_variable_4x4,
    "nonlinear": nonlinear,
}


def _range(label: str, enabled: bool) -> None:
    if enabled:
        torch.cuda.nvtx.range_push(label)


def _range_end(enabled: bool) -> None:
    if enabled:
        torch.cuda.nvtx.range_pop()


def _run_steps(
    runner: torch.nn.Module,
    state: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    steps: int,
    *,
    timestep: float,
    compiled: bool,
) -> torch.Tensor:
    """Run a sample, limiting compiled graphs to four time steps."""

    if steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if not compiled:
        return runner(state, *inputs, steps=steps).state
    if steps == 0:
        return state

    current = state
    time = state.new_zeros(())
    completed = 0
    while completed < steps:
        chunk = min(COMPILED_STEPS_PER_CALL, steps - completed)
        current = runner(current, *inputs, t0=time, steps=chunk)
        completed += chunk
        time = time + chunk * timestep
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=BUILDERS, default="burgers")
    parser.add_argument("--mode", choices=("eager", "compiled"), default="eager")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32", "float64"), default="float32")
    parser.add_argument("--cells", type=int, default=256*256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=25)
    arguments = parser.parse_args()

    dtype = getattr(torch, arguments.dtype)
    solver, state, inputs = BUILDERS[arguments.workload](
        arguments.cells, arguments.batch, dtype
    )
    solver = solver.to(device=arguments.device, dtype=dtype)
    state = state.to(arguments.device)
    inputs = tuple(value.to(arguments.device) for value in inputs)
    runner = (
        torch.compile(_StateOnlySolver(solver), fullgraph=True)
        if arguments.mode == "compiled"
        else solver
    )
    compiled = arguments.mode == "compiled"
    cuda = state.is_cuda

    _range(f"thalassa:{arguments.workload}:{arguments.mode}:warmup", cuda)
    for _ in range(arguments.warmup):
        _run_steps(
            runner,
            state,
            inputs,
            arguments.steps,
            timestep=solver._dt,
            compiled=compiled,
        )
    if cuda:
        torch.cuda.synchronize()
    _range_end(cuda)

    durations: list[float] = []
    _range(f"thalassa:{arguments.workload}:{arguments.mode}:measure", cuda)
    for _ in range(arguments.samples):
        start = perf_counter()
        _run_steps(
            runner,
            state,
            inputs,
            arguments.steps,
            timestep=solver._dt,
            compiled=compiled,
        )
        if cuda:
            torch.cuda.synchronize()
        durations.append(perf_counter() - start)
    _range_end(cuda)
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "workload": arguments.workload,
                "mode": arguments.mode,
                "device": str(state.device),
                "dtype": str(dtype),
                "cells": arguments.cells,
                "batch": arguments.batch,
                "steps": arguments.steps,
                "samples": arguments.samples,
                "median_seconds": statistics.median(durations),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
