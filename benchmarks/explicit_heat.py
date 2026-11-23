"""Minimal explicit-heat smoke benchmark using the current source tree.

Use ``runtime_workloads.py`` for eager/compiled comparisons and NVTX ranges.

nsys profile --trace=cuda,nvtx -o explicit-heat \
    python -c 'import os, runpy; os.environ.pop("LD_PRELOAD", None); runpy.run_path("benchmarks/explicit_heat.py", run_name="__main__")'

"""

import sys
from pathlib import Path
from time import perf_counter

import torch
from sympy import Derivative as D
from sympy import Eq, Function, symbols

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thalassa import (  # noqa: E402
    Axis,
    CartesianDomain,
    PDEEquation,
    PDEProblem,
    TwoLevelDiscretization,
    UniformDirichlet,
    UniformGrid,
    centered,
    compile_solver,
    old,
)


def build(cells: int):
    t, x = symbols("t x")
    u = Function("u")(t, x)
    problem = PDEProblem(
        (PDEEquation(u, Eq(D(u, t), old(0.01 * D(u, (x, 2))))),),
        t,
        CartesianDomain(Axis(x, 0, 1)),
        {u: UniformDirichlet()},
        name="explicit-heat-benchmark",
        discretization=TwoLevelDiscretization(
            1e-6,
            (centered(D(u, (x, 2))),),
        ),
    )
    return compile_solver(problem, UniformGrid((cells,)))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = build(4096).to(device)
    state = torch.rand((256, 1, 4096), device=device)
    compiled = torch.compile(solver)
    compiled(state, steps=10)
    if device == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(100):
        state = compiled(state, steps=10).state
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"{device}: {perf_counter() - start:.6f}s")
