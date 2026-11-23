"""Print every compiler phase for a small mixed Thalassa problem.

Run from the repository root with::

    python examples/compiler_phases.py

The example calls the same pure passes used by ``compile_program`` one at a
time.  It deliberately contains a pointwise coupled block followed by a
spatially implicit block, so coefficient extraction, canonical orientation,
dependency scheduling, and block classification are all visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp
import torch
from sympy import Derivative as D
from sympy import Eq, Function, symbols

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

import thalassa
from thalassa.compiler import (
    canonicalize,
    characterize,
    discretize,
    schedule,
    verify,
)
from thalassa.ir import (
    CoefficientRef,
    CoordinateRef,
    GridRef,
    TimeRef,
    expression_to_dict,
)


def heading(number: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{number}. {title}\n{'=' * 78}")


def expression_json(expression) -> str:
    return json.dumps(
        expression_to_dict(expression),
        indent=2,
        sort_keys=True,
    )


def reference_name(reference, fields: tuple[str, ...]) -> str:
    if isinstance(reference, GridRef):
        return (
            f"state field={fields[reference.field]!r}, "
            f"level={reference.level.value!r}, offset={reference.offset}"
        )
    if isinstance(reference, CoefficientRef):
        return (
            f"coefficient={reference.name!r}, "
            f"level={reference.level.value!r}, offset={reference.offset}"
        )
    if isinstance(reference, CoordinateRef):
        return f"coordinate axis={reference.axis}, offset={reference.offset}"
    if isinstance(reference, TimeRef):
        return f"time level={reference.level.value!r}"
    return repr(reference)


def build_problem():
    t, x = symbols("t x")
    beta, kappa = symbols("beta kappa", positive=True)
    u = Function("u")(t, x)
    v = Function("v")(t, x)
    w = Function("w")(t, x)
    a = Function("a")(x)

    problem = thalassa.PDEProblem(
        equations=(
            # The negative target coefficient is flipped by canonicalization.
            thalassa.PDEEquation(
                u,
                Eq(
                    -2 * thalassa.new(u) - a * thalassa.new(v),
                    thalassa.old(u),
                ),
            ),
            thalassa.PDEEquation(
                v,
                Eq(
                    beta * thalassa.new(u) + 3 * thalassa.new(v),
                    thalassa.old(v),
                ),
            ),
            # This block depends on the completed (u, v) block and contains a
            # nonzero-offset new-state stencil, so it is linear implicit.
            thalassa.PDEEquation(
                w,
                Eq(
                    D(w, t),
                    thalassa.new(kappa * D(w, (x, 2)) + u),
                ),
            ),
        ),
        time=t,
        domain=thalassa.CartesianDomain(thalassa.Axis(x, 0, 1)),
        boundaries={
            u: thalassa.Periodic(),
            v: thalassa.Periodic(),
            w: thalassa.Periodic(),
        },
        parameters=(
            thalassa.ScalarParameter(beta),
            thalassa.ScalarParameter(kappa),
        ),
        coefficient_fields=(
            thalassa.CoefficientField(a, boundary=thalassa.Periodic()),
        ),
        name="compiler-phases",
        discretization=thalassa.TwoLevelDiscretization(
            sp.Rational(1, 10),
            (thalassa.centered(D(w, (x, 2))),),
        ),
    )
    return problem, thalassa.UniformGrid((5,)), (u, v, w)


def main() -> None:
    torch.set_printoptions(precision=5, linewidth=100)
    problem, grid, source_fields = build_problem()
    field_names = tuple(field.func.__name__ for field in source_fields)

    heading(0, "Source model")
    print(f"problem: {problem.name!r}")
    print(f"grid: {grid.shape}")
    print(f"time step: {problem.discretization.dt}")
    for index, equation in enumerate(problem.equations):
        print(f"equation {index} ({field_names[index]}): {equation.equation}")
        print(f"  residual: {equation.residual}")
    for rule in problem.discretization.differences:
        print(f"difference rule: {rule.derivative}")
        print(f"  exact stencil: {rule.stencil.entries}")

    heading(1, "verify(problem, grid)")
    verified = verify(problem, grid)
    print("field indices:")
    for field, index in verified.field_indices.items():
        print(f"  {field} -> {index}")
    print("resolved finite-difference rules:")
    for derivative, entries in verified.resolved_rules.items():
        print(f"  {derivative} -> {entries}")
    print(f"normalized options: {verified.options}")

    heading(2, "discretize(verified)")
    discrete = discretize(verified)
    print("typed atom table (the symbolic residuals refer to these names):")
    for symbol, reference in sorted(
        discrete.atoms.items(), key=lambda item: item[0].name
    ):
        print(f"  {symbol} -> {reference_name(reference, field_names)}")
    for index, equation in enumerate(discrete.equations):
        print(f"\nequation {index} target={field_names[equation.target]!r}")
        print(f"symbolic residual: {sp.sstr(equation.residual)}")
        print("execution expression IR:")
        print(expression_json(equation.execution_residual))

    heading(3, "canonicalize(discrete)")
    canonical = canonicalize(discrete)
    for index, equation in enumerate(canonical.equations):
        before = discrete.equations[index].residual
        changed = before != equation.residual
        dependencies = tuple(
            field_names[field] for field in canonical.dependencies[index]
        )
        print(f"equation {index} target={field_names[equation.target]!r}")
        print(f"  orientation changed: {changed}")
        print(f"  before: {sp.sstr(before)}")
        print(f"  after:  {sp.sstr(equation.residual)}")
        print(f"  new-state dependencies: {dependencies}")

    heading(4, "schedule(canonical)")
    scheduled = schedule(canonical)
    for layer_index, layer in enumerate(scheduled.layers):
        named_blocks = tuple(
            tuple(field_names[field] for field in block) for block in layer
        )
        print(f"layer {layer_index}: {named_blocks}")

    heading(5, "characterize(scheduled) -> ProgramIR")
    program = characterize(scheduled)
    for equation_index, equation in enumerate(program.equations):
        print(
            f"equation {equation_index} target={program.fields[equation.target]!r}"
        )
        print("  residual IR:")
        print(expression_json(equation.residual))
        print("  diagonal IR:")
        print(
            "null"
            if equation.diagonal is None
            else expression_json(equation.diagonal)
        )
    for layer_index, layer in enumerate(program.schedule):
        for block in layer.blocks:
            targets = tuple(program.fields[target] for target in block.targets)
            print(
                f"layer {layer_index}, block {block.id}: targets={targets}, "
                f"kind={block.kind.value!r}, "
                f"temporal={block.temporal_treatment.value!r}"
            )
            print(f"  properties: {block.properties}")
            if block.pointwise_coefficients is not None:
                coefficients = [
                    [expression_to_dict(expression) for expression in row]
                    for row in block.pointwise_coefficients
                ]
                print("  extracted pointwise coefficient matrix:")
                print(json.dumps(coefficients, indent=2, sort_keys=True))

    print("\ncomplete deterministic ProgramIR serialization:")
    print(json.dumps(program.to_dict(), indent=2, sort_keys=True))

    heading(6, "CompiledSolver construction and one runtime step")
    solver = thalassa.CompiledSolver(program)
    print("selected pointwise runtime strategies:")
    for block_id, strategy in solver._pointwise_strategies.items():
        print(f"  block {block_id}: {strategy}")

    coordinate = (torch.arange(5, dtype=torch.float64) + 0.5) / 5
    initial = torch.stack(
        (
            torch.sin(2 * torch.pi * coordinate),
            torch.cos(2 * torch.pi * coordinate),
            torch.zeros_like(coordinate),
        )
    )
    coefficient = 0.5 + 0.1 * coordinate
    result = solver.solve(
        initial,
        parameters={"beta": 0.25, "kappa": 0.02},
        coefficient_fields={"a": coefficient},
        raise_on_nonconvergence=False,
    )
    print(f"initial state:\n{initial}")
    print(f"final state:\n{result.state}")
    print(f"converged: {result.converged}")
    print("block diagnostics:")
    for diagnostic in result.diagnostics:
        print(f"  {diagnostic}")


if __name__ == "__main__":
    main()
