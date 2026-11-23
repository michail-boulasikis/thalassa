"""Run a small IMEX Burgers problem with the Thalassa 2.0 API."""

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
    old,
)


def make_solver(cells: int = 256):
    t, x = symbols("t x")
    nu = symbols("nu", positive=True)
    u = Function("u")(t, x)
    eq = PDEEquation(u, Eq(D(u,t) + old(u * D(u,x)), new(nu * D(u, (x, 2)))))
    problem = PDEProblem(
        equations=(eq,),
        time=t,
        domain=CartesianDomain(Axis(x, 0, 1)),
        boundaries={u: UniformDirichlet(0.0)},
        parameters=ScalarParameter(nu),
        name="burgers",
        discretization=TwoLevelDiscretization(
            dt=2e-4,
            differences=(
                centered(D(u, x), accuracy=2),
                centered(D(u, (x, 2)), accuracy=2),
            ),
        ),
    )
    return compile_solver(problem, UniformGrid(cells))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    cells = 256

    solver = make_solver(cells).to(device=device, dtype=dtype)

    x = (torch.arange(cells, device=device, dtype=dtype) + 0.5) / cells
    initial = torch.exp(-100 * (x - 0.5) ** 2)[None, :]

    result = solver.solve(
        initial,
        parameters={"nu": 1e-3},
        steps=100,
    )

    print(result.state.shape, result.state.device, result.converged)
