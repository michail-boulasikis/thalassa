import unittest
from unittest import mock

import torch
from sympy import Derivative as D
from sympy import Eq, Function, symbols

import thalassa
from thalassa.ir import Call, NaryOp, Power, StencilApply
from thalassa.runtime import ghost_shift


def _stencils(expression):
    if isinstance(expression, StencilApply):
        return (expression, *_stencils(expression.expression))
    if isinstance(expression, NaryOp):
        return tuple(
            stencil
            for argument in expression.arguments
            for stencil in _stencils(argument)
        )
    if isinstance(expression, Power):
        return (*_stencils(expression.base), *_stencils(expression.exponent))
    if isinstance(expression, Call):
        return tuple(
            stencil
            for argument in expression.arguments
            for stencil in _stencils(argument)
        )
    return ()


def _burgers_solver(cells):
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
        thalassa.CartesianDomain(thalassa.Axis(x, 0, 1)),
        {u: thalassa.UniformDirichlet()},
        (thalassa.ScalarParameter(nu),),
        discretization=thalassa.TwoLevelDiscretization(
            2e-4,
            (thalassa.centered(D(u, x)), thalassa.centered(D(u, (x, 2)))),
        ),
    )
    return thalassa.compile_solver(problem, thalassa.UniformGrid((cells,)))


class StencilConvolutionTests(unittest.TestCase):
    def test_burgers_preserves_stencils_and_compile_captures_convolution(self):
        solver = _burgers_solver(8)
        residual = solver.program.equations[0].residual
        stencils = _stencils(residual)
        self.assertEqual(len(stencils), 2)
        self.assertEqual(
            {tuple(offset for offset, _ in stencil.entries) for stencil in stencils},
            {((-1,), (1,)), ((-1,), (0,), (1,))},
        )

        original = torch.nn.functional.conv1d
        with mock.patch(
            "thalassa.runtime.torch_functional.conv1d", wraps=original
        ) as convolution:
            output = solver.step(torch.rand(1, 8), 1e-3)
        self.assertTrue(output.converged)
        self.assertGreater(convolution.call_count, 0)

        # Inspect a fresh graph. Earlier compile tests exercise enough scalar and
        # dtype specializations to exhaust Dynamo's process-wide recompile cache.
        torch._dynamo.reset()
        graphs = []

        def backend(graph, _inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(solver, backend=backend)
        compiled(torch.rand(1, 8), torch.tensor(1e-3))
        convolution_nodes = [
            node
            for graph in graphs
            for node in graph.graph.nodes
            if "conv1d" in str(node.target)
        ]
        self.assertTrue(convolution_nodes)

    def test_sibling_stencils_are_one_grouped_convolution(self):
        t, x = symbols("t x")
        u = Function("u")(t, x)
        v = Function("v")(t, x)
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    u,
                    Eq(D(u, t) + thalassa.old(D(u, x) * D(v, (x, 2))), 0),
                ),
                thalassa.PDEEquation(v, Eq(D(v, t), 0)),
            ),
            t,
            thalassa.CartesianDomain(thalassa.Axis(x, 0, 1)),
            {u: thalassa.UniformDirichlet(), v: thalassa.HomogeneousNeumann()},
            discretization=thalassa.TwoLevelDiscretization(
                0.01,
                (
                    thalassa.centered(D(u, x)),
                    thalassa.centered(D(v, (x, 2))),
                ),
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((8,)))
        product_stencils = _stencils(solver.program.equations[0].residual)
        self.assertEqual(len(product_stencils), 2)

        original = torch.nn.functional.conv1d
        with mock.patch(
            "thalassa.runtime.torch_functional.conv1d", wraps=original
        ) as convolution:
            result = solver.step(torch.rand(2, 8))
        self.assertTrue(result.converged)
        self.assertEqual(convolution.call_count, 1)
        self.assertEqual(convolution.call_args.kwargs["groups"], 2)

    def test_nested_composite_stencil_matches_shift_semantics(self):
        t, x = symbols("t x")
        u = Function("u")(t, x)
        coefficient = Function("a")(x)
        flux = D(coefficient * D(u, x), x)
        problem = thalassa.PDEProblem(
            (thalassa.PDEEquation(u, Eq(D(u, t), thalassa.old(flux))),),
            t,
            thalassa.CartesianDomain(thalassa.Axis(x, 0, 1)),
            {u: thalassa.UniformDirichlet(2)},
            coefficient_fields=(thalassa.CoefficientField(coefficient),),
            discretization=thalassa.TwoLevelDiscretization(
                0.01,
                (thalassa.centered(D(u, x)), thalassa.centered(flux)),
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((8,)))
        state = torch.linspace(0, 1, 8)[None, :]
        sampled = torch.linspace(1, 2, 8)
        result = solver.step(state, sampled)

        def inner(offset):
            return 4 * (
                ghost_shift(state[0], (offset + 1,), "dirichlet", 2)
                - ghost_shift(state[0], (offset - 1,), "dirichlet", 2)
            )

        outer = 4 * (
            ghost_shift(sampled, (1,), "neumann") * inner(1)
            - ghost_shift(sampled, (-1,), "neumann") * inner(-1)
        )
        expected = state[0] + 0.01 * outer
        self.assertTrue(torch.allclose(result.state[0], expected, atol=2e-7))

        with self.assertRaises(thalassa.DiscretizationError):
            thalassa.compile_solver(problem, thalassa.UniformGrid((1,)))

    def test_periodic_sampled_coefficient_wraps_in_conservative_stencil(self):
        t, x = symbols("t x")
        u = Function("u")(t, x)
        coefficient = Function("a")(x)
        flux = D(coefficient * D(u, x), x)
        problem = thalassa.PDEProblem(
            thalassa.PDEEquation(u, Eq(D(u, t), thalassa.old(flux))),
            t,
            thalassa.CartesianDomain(thalassa.Axis(x, 0, 1)),
            {u: thalassa.Periodic()},
            coefficient_fields=thalassa.CoefficientField(
                coefficient, boundary=thalassa.Periodic()
            ),
            discretization=thalassa.TwoLevelDiscretization(
                0.01,
                (thalassa.centered(D(u, x)), thalassa.centered(flux)),
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid(8))
        state = torch.linspace(-0.5, 1.0, 8)[None, :]
        sampled = torch.linspace(1.0, 2.5, 8)
        result = solver.step(state, sampled)

        def inner(offset):
            return 4 * (
                ghost_shift(state[0], (offset + 1,), "periodic")
                - ghost_shift(state[0], (offset - 1,), "periodic")
            )

        outer = 4 * (
            ghost_shift(sampled, (1,), "periodic") * inner(1)
            - ghost_shift(sampled, (-1,), "periodic") * inner(-1)
        )
        expected = state[0] + 0.01 * outer
        self.assertTrue(torch.allclose(result.state[0], expected, atol=2e-7))
        self.assertTrue(
            torch.allclose(result.state.mean(), state.mean(), atol=1e-7)
        )

    def test_multidimensional_stencils_dispatch_to_matching_convolution(self):
        t, x, y, z = symbols("t x y z")
        cases = (
            ((x, y), (4, 5), "conv2d"),
            ((x, y, z), (3, 4, 5), "conv3d"),
        )
        for axes, shape, function_name in cases:
            with self.subTest(ndim=len(axes)):
                field = Function(f"u{len(axes)}")(t, *axes)
                laplacian = sum(D(field, (axis, 2)) for axis in axes)
                problem = thalassa.PDEProblem(
                    (
                        thalassa.PDEEquation(
                            field, Eq(D(field, t) + thalassa.old(laplacian), 0)
                        ),
                    ),
                    t,
                    thalassa.CartesianDomain(
                        *(thalassa.Axis(axis, 0, 1) for axis in axes)
                    ),
                    {field: thalassa.HomogeneousNeumann()},
                    discretization=thalassa.TwoLevelDiscretization(
                        0.01,
                        tuple(
                            thalassa.centered(D(field, (axis, 2)))
                            for axis in axes
                        ),
                    ),
                )
                solver = thalassa.compile_solver(problem, thalassa.UniformGrid(shape))
                self.assertEqual(
                    len(_stencils(solver.program.equations[0].residual)), 1
                )
                direct_solution = solver.program.equations[0].direct_solution
                self.assertIsInstance(direct_solution, StencilApply)
                self.assertEqual(len(_stencils(direct_solution)), 1)
                self.assertIn(
                    (0,) * len(axes),
                    tuple(offset for offset, _ in direct_solution.entries),
                )
                original = getattr(torch.nn.functional, function_name)
                with mock.patch(
                    f"thalassa.runtime.torch_functional.{function_name}",
                    wraps=original,
                ) as convolution:
                    result = solver.step(torch.rand((1, *shape)))
                self.assertEqual(result.state.shape, (1, *shape))
                self.assertEqual(convolution.call_count, 1)


if __name__ == "__main__":
    unittest.main()
