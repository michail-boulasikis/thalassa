from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest import mock

import torch
from sympy import Eq, Function, symbols

import thalassa
from thalassa.ir import Call, GridRef, Literal, NaryOp, StencilApply, TimeLevel
from thalassa.runtime import (
    bicgstab,
    conjugate_gradient,
    extend_ghosts,
    projected_conjugate_gradient,
)


def _reference_region(tensor, offset, halo, kind, value):
    extension = tuple(
        requested + abs(shift)
        for requested, shift in zip(halo, offset, strict=True)
    )
    extended = extend_ghosts(tensor, extension, kind=kind, value=value)
    slices = [slice(None)] * extended.ndim
    spatial_ndim = len(offset)
    for axis, (radius, requested, shift) in enumerate(
        zip(extension, halo, offset, strict=True)
    ):
        dimension = extended.ndim - spatial_ndim + axis
        cells = tensor.shape[tensor.ndim - spatial_ndim + axis]
        start = radius - requested + shift
        slices[dimension] = slice(start, start + cells + 2 * requested)
    return extended[tuple(slices)]


def _region_solver(ndim):
    symbols_all = symbols("t x y z")
    time = symbols_all[0]
    axes = symbols_all[1 : ndim + 1]
    field = Function(f"u{ndim}")(time, *axes)
    domain = thalassa.CartesianDomain(
        *(thalassa.Axis(axis, 0, 1) for axis in axes)
    )
    problem = thalassa.PDEProblem(
        (thalassa.PDEEquation(field, Eq(thalassa.new(field), 0)),),
        time,
        domain,
        {field: thalassa.UniformDirichlet()},
        discretization=thalassa.TwoLevelDiscretization(1, ()),
    )
    grid_shape = (5, 6, 7)[:ndim]
    base = thalassa.compile_solver(problem, thalassa.UniformGrid(grid_shape))
    offsets = []
    for axis in range(ndim):
        left = [0] * ndim
        right = [0] * ndim
        left[axis] = -1
        right[axis] = 1
        offsets.extend((tuple(left), tuple(right)))
    entries = tuple((offset, "1") for offset in offsets)
    shifted = tuple(-1 if axis % 2 == 0 else 1 for axis in range(ndim))
    expression = NaryOp(
        "add",
        (
            GridRef(0, TimeLevel.OLD, (0,) * ndim),
            GridRef(0, TimeLevel.OLD, shifted),
            StencilApply(
                entries,
                StencilApply(
                    entries,
                    GridRef(0, TimeLevel.OLD, shifted),
                ),
            ),
        ),
    )
    equation = replace(base.program.equations[0], residual=expression, diagonal=None)
    return thalassa.CompiledSolver(
        replace(base.program, equations=(equation,))
    )


class RuntimeOptimizationTests(unittest.TestCase):
    def test_constant_subtrees_are_folded_and_broadcast_without_scalar_tensors(self):
        t, x = symbols("t x")
        u = Function("u")(t, x)
        problem = thalassa.PDEProblem(
            (thalassa.PDEEquation(u, Eq(thalassa.new(u), 0)),),
            t,
            thalassa.CartesianDomain(thalassa.Axis(x, 0, 1)),
            {u: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        base = thalassa.compile_solver(problem, thalassa.UniformGrid((6,)))
        constant = NaryOp(
            "mul",
            (
                NaryOp("add", (Literal("2"), Literal("3"))),
                Call("sin", (NaryOp("mul", (Literal("1/2"), Literal("pi"))),)),
            ),
        )
        residual = NaryOp(
            "add",
            (GridRef(0, TimeLevel.NEW, (0,)), constant),
        )
        equation = replace(
            base.program.equations[0],
            residual=residual,
            diagonal=Literal("1"),
        )
        solver = thalassa.CompiledSolver(
            replace(base.program, equations=(equation,))
        )
        self.assertEqual(solver._constant_values[constant], 5.0)
        self.assertFalse(
            any(name.startswith("_literal") for name, _ in solver.named_buffers())
        )
        state = torch.zeros(2, 1, 6, dtype=torch.float64)
        result = solver.step(state)
        self.assertTrue(torch.equal(result.state, torch.full_like(state, -5.0)))

    def test_precomputed_regions_match_flip_cat_reference_in_1d_to_3d(self):
        for ndim in range(1, 4):
            solver = _region_solver(ndim)
            values = torch.arange(
                6 * math.prod(solver.program.grid_shape), dtype=torch.float64
            ).reshape(2, 3, *solver.program.grid_shape)
            requests = tuple(solver._region_index_names)
            self.assertIn(((0,) * ndim, (0,) * ndim), requests)
            self.assertTrue(any(any(halo) for _, halo in requests))
            for offset, halo in requests:
                for kind, boundary_value in (
                    ("neumann", 0.0),
                    ("dirichlet", 2.75),
                    ("periodic", 0.0),
                ):
                    with self.subTest(
                        ndim=ndim, offset=offset, halo=halo, kind=kind
                    ):
                        actual = solver._shifted_region(
                            values, offset, halo, kind, boundary_value
                        )
                        expected = _reference_region(
                            values, offset, halo, kind, boundary_value
                        )
                        self.assertTrue(torch.equal(actual, expected))

    def test_single_field_stencil_hot_path_does_not_build_cat_ghosts(self):
        t, x = symbols("t x")
        u = Function("u")(t, x)
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    u, Eq(u.diff(t), thalassa.old(u.diff(x, 2)))
                ),
            ),
            t,
            thalassa.CartesianDomain(thalassa.Axis(x, 0, 1)),
            {u: thalassa.UniformDirichlet(1.5)},
            discretization=thalassa.TwoLevelDiscretization(
                0.01,
                (thalassa.centered(u.diff(x, 2)),),
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((16,)))
        with mock.patch(
            "thalassa.runtime.extend_ghosts",
            side_effect=AssertionError("hot path called extend_ghosts"),
        ):
            output = solver.step(torch.rand(3, 1, 16))
        self.assertTrue(torch.all(output.converged))

    def test_krylov_reference_dtypes_breakdown_and_fullgraph_capture(self):
        for dtype in (torch.float32, torch.float64):
            tolerance = 2e-5 if dtype == torch.float32 else 1e-11
            spd = torch.tensor(
                [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
                dtype=dtype,
            )
            general = torch.tensor(
                [[3.0, 2.0, 0.0], [0.0, 2.0, -1.0], [1.0, 0.0, 1.0]],
                dtype=dtype,
            )
            rhs = torch.tensor([1.0, -2.0, 3.0], dtype=dtype)
            cg = conjugate_gradient(
                lambda value, matrix=spd: matrix @ value,
                rhs,
                batch_ndim=0,
                max_iterations=20,
                rtol=1e-7 if dtype == torch.float32 else 1e-12,
            )
            bi = bicgstab(
                lambda value, matrix=general: matrix @ value,
                rhs,
                batch_ndim=0,
                max_iterations=30,
                rtol=1e-6 if dtype == torch.float32 else 1e-12,
            )
            self.assertTrue(cg.converged)
            self.assertTrue(bi.converged)
            self.assertTrue(
                torch.allclose(cg.solution, torch.linalg.solve(spd, rhs), atol=tolerance)
            )
            self.assertTrue(
                torch.allclose(
                    bi.solution, torch.linalg.solve(general, rhs), atol=tolerance
                )
            )

            laplacian = torch.tensor(
                [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]],
                dtype=dtype,
            )
            projected = projected_conjugate_gradient(
                lambda value, matrix=laplacian: matrix @ value,
                torch.tensor([1.0, -1.0, 0.0], dtype=dtype),
                spatial_ndim=1,
                batch_ndim=0,
                max_iterations=10,
            )
            self.assertTrue(projected.converged)
            self.assertLess(float(torch.abs(projected.solution.mean())), tolerance)

        def compiled_cg(value):
            matrix = torch.tensor(
                [[4.0, 1.0], [1.0, 3.0]],
                dtype=value.dtype,
                device=value.device,
            )
            return conjugate_gradient(
                lambda trial: matrix @ trial,
                value,
                batch_ndim=0,
                max_iterations=10,
            )

        rhs = torch.tensor([1.0, 2.0])
        eager = compiled_cg(rhs)
        compiled = torch.compile(
            compiled_cg, backend="eager", fullgraph=True
        )(rhs)
        self.assertTrue(torch.allclose(compiled.solution, eager.solution))
        self.assertTrue(torch.equal(compiled.converged, eager.converged))
        self.assertTrue(torch.equal(compiled.breakdown, eager.breakdown))

        broken = conjugate_gradient(
            lambda value: torch.zeros_like(value),
            torch.ones(4),
            batch_ndim=0,
            max_iterations=2,
        )
        self.assertTrue(broken.breakdown)
        self.assertFalse(broken.converged)

    def test_pointwise_coefficients_avoid_basis_trial_batch(self):
        t, x = symbols("t x")
        u = Function("u")(t, x)
        v = Function("v")(t, x)
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    u, Eq(2 * thalassa.new(u) + thalassa.new(v), 5)
                ),
                thalassa.PDEEquation(
                    v, Eq(thalassa.new(u) - 3 * thalassa.new(v), -5)
                ),
            ),
            t,
            thalassa.CartesianDomain(thalassa.Axis(x, 0, 1)),
            {u: thalassa.UniformDirichlet(), v: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((7,)))
        state = torch.zeros(3, 2, 7, dtype=torch.float64)
        with (
            mock.patch.object(
                solver, "_residuals", wraps=solver._residuals
            ) as residuals,
            mock.patch(
                "thalassa.runtime.torch.eye",
                side_effect=AssertionError("pointwise solve built a basis"),
            ),
            mock.patch(
                "thalassa.runtime.torch.linalg.solve_ex",
                side_effect=AssertionError("pointwise solve used solve_ex"),
            ),
        ):
            eager = solver.step(state)
        self.assertEqual(residuals.call_count, 2)
        expected = torch.tensor([10 / 7, 15 / 7], dtype=torch.float64)
        self.assertTrue(
            torch.allclose(eager.state, expected.reshape(1, 2, 1).expand_as(state))
        )
        compiled = torch.compile(
            solver, backend="eager", fullgraph=True
        )(state)
        self.assertTrue(torch.allclose(compiled.state, eager.state))

    def test_batched_newton_backtracking_and_fullgraph_match(self):
        t, x = symbols("t x")
        u = Function("u")(t, x)
        problem = thalassa.PDEProblem(
            (thalassa.PDEEquation(u, Eq(thalassa.new(u**3), 1)),),
            t,
            thalassa.CartesianDomain(thalassa.Axis(x, 0, 1)),
            {u: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((5,)))
        for dtype in (torch.float32, torch.float64):
            state = torch.tensor([0.1, 0.2], dtype=dtype).reshape(2, 1, 1)
            state = state.expand(2, 1, 5).clone()
            eager = solver.step(state)
            self.assertTrue(torch.all(eager.converged))
            self.assertTrue(
                torch.allclose(eager.state, torch.ones_like(state), atol=2e-6)
            )
            compiled = torch.compile(
                solver, backend="eager", fullgraph=True
            )(state)
            self.assertTrue(torch.equal(compiled.converged, eager.converged))
            self.assertTrue(
                torch.allclose(compiled.state, eager.state, atol=2e-6)
            )


if __name__ == "__main__":
    unittest.main()
