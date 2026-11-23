import unittest

import sympy as sp
import torch
from sympy import Derivative as D
from sympy import Eq, Function, symbols

import thalassa
from thalassa.compiler import canonicalize, discretize, verify
from thalassa.ir import (
    Call,
    CoefficientRef,
    CoordinateRef,
    GridRef,
    Literal,
    NaryOp,
    Power,
    ScalarRef,
    StencilApply,
    TimeLevel,
    TimeRef,
    expression_from_dict,
    expression_to_dict,
)
from thalassa.runtime import bicgstab, extend_ghosts, ghost_shift


class EdgeAndFunctionTests(unittest.TestCase):
    def setUp(self):
        self.t, self.x = symbols("t x")
        self.u = Function("u")(self.t, self.x)
        self.domain = thalassa.CartesianDomain(thalassa.Axis(self.x, 0, 1))

    def test_analytic_coordinate_time_and_function_runtime(self):
        forcing = sp.sin(self.t) + sp.cos(self.x) + sp.exp(-self.x)
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u, Eq(D(self.u, self.t), thalassa.old(forcing))
                ),
            ),
            self.t,
            self.domain,
            {self.u: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(
                sp.Rational(1, 10), ()
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((4,)))
        state = torch.zeros(1, 4, dtype=torch.float64)
        result = solver(state, t0=0.2, steps=2)
        centers = (torch.arange(4, dtype=torch.float64) + 0.5) / 4
        expected = 0.1 * (
            torch.sin(torch.tensor(0.2))
            + torch.sin(torch.tensor(0.3))
            + 2 * torch.cos(centers)
            + 2 * torch.exp(-centers)
        )
        self.assertTrue(torch.allclose(result.state[0], expected, atol=1e-7))
        self.assertIs(result.final_state, result.state)

    def test_batched_scalar_and_sampled_coefficient_mapping(self):
        nu = symbols("nu")
        coefficient = Function("a", positive=True)(self.x)
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u,
                    Eq(
                        D(self.u, self.t)
                        - thalassa.old(nu * coefficient * self.u),
                        0,
                    ),
                ),
            ),
            self.t,
            self.domain,
            {self.u: thalassa.UniformDirichlet()},
            parameters=(thalassa.ScalarParameter(nu),),
            coefficient_fields=(thalassa.CoefficientField(coefficient),),
            discretization=thalassa.TwoLevelDiscretization(0.01, ()),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((5,)))
        initial = torch.ones(2, 1, 5)
        nu_values = torch.tensor([1.0, 2.0])
        sampled = torch.linspace(1, 2, 5)
        result = solver.solve(
            initial,
            parameters={nu: nu_values},
            coefficient_fields={coefficient: sampled},
            steps=1,
        )
        expected = initial + 0.01 * nu_values[:, None, None] * sampled[None, None, :]
        self.assertTrue(torch.allclose(result.final_state, expected))
        self.assertTrue(result.converged)
        self.assertEqual(len(result.diagnostics), 1)

    def test_nonsymmetric_implicit_bicgstab_path(self):
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u,
                    Eq(D(self.u, self.t) + thalassa.new(D(self.u, self.x)), 0),
                ),
            ),
            self.t,
            self.domain,
            {self.u: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(
                0.01, (thalassa.centered(D(self.u, self.x)),)
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((12,)))
        block = solver.program.schedule[0].blocks[0]
        self.assertEqual(block.kind, thalassa.BlockKind.LINEAR_IMPLICIT)
        self.assertEqual(block.properties.symmetry, thalassa.PropertyState.REFUTED)
        output = solver.step(torch.rand(3, 1, 12, dtype=torch.float64))
        self.assertTrue(torch.all(output.converged))
        state = torch.rand(2, 1, 12, dtype=torch.float32)
        eager = solver(state)
        compiled = torch.compile(solver, backend="eager")(state)
        self.assertTrue(torch.all(compiled.converged))
        self.assertTrue(torch.allclose(compiled.state, eager.state))

        limited = thalassa.compile_solver(
            problem,
            thalassa.UniformGrid((12,)),
            thalassa.CompileOptions(
                krylov=thalassa.KrylovOptions(max_iterations=1, rtol_float64=1e-14)
            ),
        )
        difficult = torch.linspace(0, 1, 12, dtype=torch.float64)[None, :]
        failed = limited.step(difficult)
        self.assertFalse(failed.converged)
        with self.assertRaises(thalassa.LinearSolveError):
            limited.solve(difficult)
        retained = limited.solve(difficult, raise_on_nonconvergence=False)
        self.assertFalse(retained.converged)

    def test_periodic_operator_proof_requires_periodic_spatial_coefficients(self):
        coefficient = Function("a", positive=True)(self.x)
        flux = D(coefficient * D(self.u, self.x), self.x)

        def compile_with(boundary):
            problem = thalassa.PDEProblem(
                thalassa.PDEEquation(
                    self.u, Eq(D(self.u, self.t), thalassa.new(flux))
                ),
                self.t,
                self.domain,
                {self.u: thalassa.Periodic()},
                coefficient_fields=thalassa.CoefficientField(
                    coefficient, boundary=boundary
                ),
                discretization=thalassa.TwoLevelDiscretization(
                    0.01,
                    (
                        thalassa.centered(D(self.u, self.x)),
                        thalassa.centered(flux),
                    ),
                ),
            )
            return thalassa.compile_solver(problem, thalassa.UniformGrid(12))

        periodic = compile_with(thalassa.Periodic())
        unproven = compile_with(thalassa.HomogeneousNeumann())
        self.assertEqual(
            periodic.program.schedule[0].blocks[0].properties.symmetry,
            thalassa.PropertyState.PROVEN,
        )
        self.assertEqual(
            unproven.program.schedule[0].blocks[0].properties.symmetry,
            thalassa.PropertyState.UNKNOWN,
        )

    def test_three_dimensional_axis_preservation(self):
        y, z = symbols("y z")
        field = Function("q")(self.t, self.x, y, z)
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    field,
                    Eq(
                        D(field, self.t)
                        + thalassa.old(D(field, self.x) + D(field, y) + D(field, z)),
                        0,
                    ),
                ),
            ),
            self.t,
            thalassa.CartesianDomain(
                thalassa.Axis(self.x, 0, 1),
                thalassa.Axis(y, -1, 1),
                thalassa.Axis(z, 2, 5),
            ),
            {field: thalassa.HomogeneousNeumann()},
            discretization=thalassa.TwoLevelDiscretization(
                0.001,
                (
                    thalassa.backward(D(field, self.x), accuracy=2),
                    thalassa.centered(D(field, y), accuracy=2),
                    thalassa.forward(D(field, z), accuracy=1),
                ),
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((4, 5, 6)))
        state = torch.rand(2, 1, 4, 5, 6)
        self.assertEqual(solver.step(state).state.shape, state.shape)
        self.assertEqual(solver.metadata.spacing, (0.25, 0.4, 0.5))

    def test_custom_stencil_algebra_and_ir_expression_roundtrip(self):
        left = thalassa.Stencil({(-1,): -1, (0,): 1})
        right = thalassa.Stencil({(0,): -1, (1,): 1})
        laplacian = left * right
        self.assertEqual(laplacian.coefficients, {(-1,): 1, (0,): -2, (1,): 1})
        self.assertEqual((laplacian - laplacian).weights, (0,))

        expressions = (
            Literal("1/2"),
            GridRef(0, TimeLevel.NEW, (1,)),
            ScalarRef("a"),
            CoefficientRef("f", TimeLevel.OLD, (0,)),
            CoordinateRef(0, -1),
            TimeRef(TimeLevel.NEW),
            NaryOp("add", (Literal("1"), Literal("2"))),
            Power(Literal("2"), Literal("3")),
            Call("sin", (Literal("1"),)),
            StencilApply((((-1,), "-1/2"), ((1,), "1/2")), GridRef(0, TimeLevel.OLD, (0,))),
        )
        for expression in expressions:
            self.assertEqual(expression_from_dict(expression_to_dict(expression)), expression)

    def test_validation_and_runtime_error_paths(self):
        base_problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u,
                    Eq(D(self.u, self.t) + thalassa.old(D(self.u, self.x)), 0),
                ),
            ),
            self.t,
            self.domain,
            {self.u: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(0.1, ()),
        )
        with self.assertRaises(thalassa.DiscretizationError):
            verify(base_problem, thalassa.UniformGrid((4,)))

        duplicate = thalassa.centered(D(self.u, self.x))
        duplicate_problem = thalassa.PDEProblem(
            base_problem.equations,
            self.t,
            self.domain,
            base_problem.boundaries,
            discretization=thalassa.TwoLevelDiscretization(
                0.1, (duplicate, duplicate)
            ),
        )
        with self.assertRaises(thalassa.DiscretizationError):
            verify(duplicate_problem, thalassa.UniformGrid((4,)))

        invalid_operator = bicgstab(
            lambda value: torch.zeros_like(value),
            torch.ones(4),
            batch_ndim=0,
            max_iterations=2,
        )
        self.assertTrue(invalid_operator.breakdown)
        with self.assertRaises(ValueError):
            extend_ghosts(torch.ones(2), (3,), kind="neumann")
        self.assertTrue(
            torch.equal(
                ghost_shift(torch.tensor([1.0, 2.0]), (1,), "periodic"),
                torch.tensor([2.0, 1.0]),
            )
        )

        explicit_solver = thalassa.compile_solver(
            thalassa.PDEProblem(
                base_problem.equations,
                self.t,
                self.domain,
                base_problem.boundaries,
                discretization=thalassa.TwoLevelDiscretization(
                    0.1, (duplicate,)
                ),
            ),
            thalassa.UniformGrid((4,)),
        )
        with self.assertRaises(ValueError):
            explicit_solver.step(torch.zeros(2, 4))
        with self.assertRaises(TypeError):
            explicit_solver.step(torch.zeros(1, 4), unknown=1)

    def test_compiler_validation_matrix(self):
        algebraic_problem = thalassa.PDEProblem(
            (thalassa.PDEEquation(self.u, Eq(thalassa.new(self.u), 0)),),
            self.t,
            self.domain,
            {self.u: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        grid = thalassa.UniformGrid((4,))
        with self.assertRaises(thalassa.ModelValidationError):
            verify("not a problem", grid)
        invalid_time = thalassa.PDEProblem(
            algebraic_problem.equations,
            "t",
            self.domain,
            algebraic_problem.boundaries,
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(invalid_time, grid)
        with self.assertRaises(thalassa.ModelValidationError):
            verify(algebraic_problem, thalassa.UniformGrid((2, 2)))
        empty = thalassa.PDEProblem(
            (),
            self.t,
            self.domain,
            {},
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(empty, grid)
        duplicate_target = thalassa.PDEProblem(
            (*algebraic_problem.equations, *algebraic_problem.equations),
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(duplicate_target, grid)
        missing_boundary = thalassa.PDEProblem(
            algebraic_problem.equations,
            self.t,
            self.domain,
            {},
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(missing_boundary, grid)
        invalid_boundary = thalassa.PDEProblem(
            algebraic_problem.equations,
            self.t,
            self.domain,
            {self.u: object()},
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(invalid_boundary, grid)

        parameter = symbols("parameter")
        duplicate_parameters = thalassa.PDEProblem(
            algebraic_problem.equations,
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            (thalassa.ScalarParameter(parameter), thalassa.ScalarParameter(parameter)),
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(duplicate_parameters, grid)
        coefficient = Function("a")(self.x)
        duplicate_coefficients = thalassa.PDEProblem(
            algebraic_problem.equations,
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            coefficient_fields=(
                thalassa.CoefficientField(coefficient),
                thalassa.CoefficientField(coefficient),
            ),
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(duplicate_coefficients, grid)
        same_name = Function("u")(self.x)
        colliding_coefficient = thalassa.PDEProblem(
            algebraic_problem.equations,
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            coefficient_fields=(thalassa.CoefficientField(same_name),),
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(colliding_coefficient, grid)

        other = Function("other")(self.t, self.x)
        wrong_target = thalassa.PDEProblem(
            (thalassa.PDEEquation(self.u, Eq(thalassa.new(other), 0)),),
            self.t,
            self.domain,
            {self.u: thalassa.UniformDirichlet()},
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(wrong_target, grid)

        extra = symbols("extra")
        undeclared_problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u,
                    Eq(
                        thalassa.new(self.u)
                        + thalassa.old(extra)
                        - thalassa.new(extra),
                        0,
                    ),
                ),
            ),
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(undeclared_problem, grid)
        unsupported_problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u,
                    Eq(
                        thalassa.new(self.u)
                        + sp.Heaviside(thalassa.new(self.u)),
                        0,
                    ),
                ),
            ),
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(unsupported_problem, grid)

        derivative_problem = thalassa.PDEProblem(
            (thalassa.PDEEquation(self.u, Eq(D(self.u, self.x), 0)),),
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            discretization=thalassa.TwoLevelDiscretization(
                1, (thalassa.centered(D(self.u, self.x)),)
            ),
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(derivative_problem, grid)
        unlevelled_field = thalassa.PDEProblem(
            (thalassa.PDEEquation(self.u, Eq(self.u, 0)),),
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.ModelValidationError):
            verify(unlevelled_field, grid)
        old_only = thalassa.PDEProblem(
            (thalassa.PDEEquation(self.u, Eq(thalassa.old(self.u), 0)),),
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            discretization=algebraic_problem.discretization,
        )
        with self.assertRaises(thalassa.CanonicalizationError):
            canonicalize(discretize(verify(old_only, grid)))

        time_rule = thalassa.FiniteDifferenceRule(
            D(self.u, self.t),
            thalassa.Stencil({(0,): 1}),
        )
        time_rule_problem = thalassa.PDEProblem(
            algebraic_problem.equations,
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            discretization=thalassa.TwoLevelDiscretization(1, (time_rule,)),
        )
        with self.assertRaises(thalassa.DiscretizationError):
            verify(time_rule_problem, grid)
        wide_rule = thalassa.FiniteDifferenceRule(
            D(self.u, self.x),
            thalassa.Stencil({(-5,): -1, (5,): 1}),
        )
        wide_problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u, Eq(thalassa.old(D(self.u, self.x)), 0)
                ),
            ),
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            discretization=thalassa.TwoLevelDiscretization(1, (wide_rule,)),
        )
        with self.assertRaises(thalassa.DiscretizationError):
            verify(wide_problem, grid)

        mixed_problem = thalassa.PDEProblem(
            (thalassa.PDEEquation(self.u, Eq(D(self.u, self.t, self.x), 0)),),
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            discretization=thalassa.TwoLevelDiscretization(
                1, (thalassa.centered(D(self.u, self.x)),)
            ),
        )
        thalassa.compile_program(mixed_problem, grid)

        imex_problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u,
                    Eq(
                        D(self.u, self.t) + thalassa.old(self.u),
                        thalassa.new(D(self.u, (self.x, 2))),
                    ),
                ),
            ),
            self.t,
            self.domain,
            algebraic_problem.boundaries,
            discretization=thalassa.TwoLevelDiscretization(
                0.1, (thalassa.centered(D(self.u, (self.x, 2))),)
            ),
        )
        imex = thalassa.compile_program(imex_problem, grid)
        self.assertEqual(
            imex.schedule[0].blocks[0].temporal_treatment,
            thalassa.TemporalTreatment.IMEX,
        )


if __name__ == "__main__":
    unittest.main()
