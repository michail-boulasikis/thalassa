import dataclasses
import inspect
import json
import unittest

import sympy as sp
from sympy import Derivative as D
from sympy import Eq, Function, symbols

import thalassa


class ModelAndCompilerTests(unittest.TestCase):
    def setUp(self):
        self.t, self.x = symbols("t x")
        self.u = Function("u")(self.t, self.x)
        self.domain = thalassa.CartesianDomain(thalassa.Axis(self.x, 0, 1))

    def heat(self):
        nu = symbols("nu", positive=True)
        equation = thalassa.PDEEquation(
            self.u,
            Eq(D(self.u, self.t), thalassa.new(nu * D(self.u, (self.x, 2)))),
        )
        problem = thalassa.PDEProblem(
            (equation,),
            self.t,
            self.domain,
            {self.u: thalassa.UniformDirichlet()},
            (thalassa.ScalarParameter(nu),),
            discretization=thalassa.TwoLevelDiscretization(
                sp.Rational(1, 100),
                (thalassa.centered(D(self.u, (self.x, 2))),),
            ),
        )
        return problem

    def test_exact_stencil_moments(self):
        first = thalassa.centered(D(self.u, self.x), accuracy=2).stencil
        second = thalassa.centered(D(self.u, (self.x, 2)), accuracy=4).stencil
        self.assertEqual(first.entries, (((-1,), -sp.Rational(1, 2)), ((1,), sp.Rational(1, 2))))
        self.assertEqual(first.moment((0,)), 0)
        self.assertEqual(first.moment((1,)), 1)
        for degree in (0, 1, 3, 4, 5):
            self.assertEqual(second.moment((degree,)), 0)
        self.assertEqual(second.moment((2,)), 2)

    def test_inputs_are_immutable_and_reusable(self):
        problem = self.heat()
        rules_before = problem.discretization.differences
        equations_before = tuple(item.equation for item in problem.equations)
        first = thalassa.compile_program(problem, thalassa.UniformGrid((16,)))
        second = thalassa.compile_program(problem, thalassa.UniformGrid((16,)))
        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first.to_dict(), sort_keys=True), json.dumps(second.to_dict(), sort_keys=True))
        self.assertIs(rules_before, problem.discretization.differences)
        self.assertEqual(equations_before, tuple(item.equation for item in problem.equations))
        with self.assertRaises((dataclasses.FrozenInstanceError, TypeError)):
            problem.name = "changed"
        with self.assertRaises((dataclasses.FrozenInstanceError, TypeError)):
            problem.discretization.dt = 1

    def test_single_items_are_normalized_to_immutable_tuples(self):
        nu = symbols("nu", positive=True)
        coefficient = thalassa.CoefficientField(
            Function("a")(self.x), boundary=thalassa.Periodic()
        )
        equation = thalassa.PDEEquation(
            self.u,
            Eq(D(self.u, self.t), thalassa.new(nu * D(self.u, (self.x, 2)))),
        )
        rule = thalassa.centered(D(self.u, (self.x, 2)))
        problem = thalassa.PDEProblem(
            equations=equation,
            time=self.t,
            domain=self.domain,
            boundaries={self.u: thalassa.Periodic()},
            parameters=thalassa.ScalarParameter(nu),
            coefficient_fields=coefficient,
            discretization=thalassa.TwoLevelDiscretization(0.01, rule),
        )

        self.assertEqual(problem.equations, (equation,))
        self.assertEqual(problem.parameters, (thalassa.ScalarParameter(nu),))
        self.assertEqual(problem.coefficient_fields, (coefficient,))
        self.assertEqual(problem.discretization.differences, (rule,))
        self.assertEqual(thalassa.UniformGrid(16).shape, (16,))
        program = thalassa.compile_program(problem, thalassa.UniformGrid(16))
        self.assertEqual(program.boundaries[0].kind, "periodic")
        self.assertEqual(program.coefficient_boundaries[0].kind, "periodic")

        default = thalassa.CoefficientField(Function("b")(self.x))
        self.assertIsInstance(default.boundary, thalassa.HomogeneousNeumann)
        with self.assertRaises(thalassa.ModelValidationError):
            thalassa.CoefficientField(Function("bad")(self.x), boundary=object())

    def test_discretization_requires_a_positive_constant_time_step(self):
        for invalid in (0, -1, symbols("dt"), sp.oo):
            with self.subTest(dt=invalid), self.assertRaises(
                thalassa.ModelValidationError
            ):
                thalassa.TwoLevelDiscretization(invalid, ())
        with self.assertRaises(TypeError):
            thalassa.PDEProblem(
                (thalassa.PDEEquation(self.u, Eq(thalassa.new(self.u), 0)),),
                self.t,
                self.domain,
                {self.u: thalassa.UniformDirichlet()},
            )

    def test_continuous_projection_is_exact_and_serialized(self):
        nested = thalassa.old(
            self.u * thalassa.new(D(self.u, self.x) + thalassa.old(self.u))
        )
        equation = thalassa.PDEEquation(
            self.u,
            Eq(D(self.u, self.t) + nested, thalassa.new(D(self.u, (self.x, 2)))),
        )
        expected_equation = Eq(
            D(self.u, self.t) + self.u * (D(self.u, self.x) + self.u),
            D(self.u, (self.x, 2)),
        )
        self.assertEqual(equation.continuous_equation, expected_equation)
        self.assertEqual(
            sp.simplify(equation.continuous_residual - (expected_equation.lhs - expected_equation.rhs)),
            0,
        )
        identity = thalassa.PDEEquation(
            self.u, Eq(thalassa.old(self.u), thalassa.new(self.u))
        )
        self.assertEqual(
            identity.continuous_equation,
            Eq(self.u, self.u, evaluate=False),
        )
        problem = thalassa.PDEProblem(
            (equation,),
            self.t,
            self.domain,
            {self.u: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(
                0.01,
                (
                    thalassa.centered(D(self.u, self.x)),
                    thalassa.centered(D(self.u, (self.x, 2))),
                ),
            ),
        )
        program = thalassa.compile_program(problem, thalassa.UniformGrid((16,)))
        self.assertEqual(program.equations[0].continuous_residual, sp.srepr(equation.continuous_residual))
        self.assertEqual(
            program.to_dict()["equations"][0]["continuous_residual"],
            sp.srepr(equation.continuous_residual),
        )

    def test_signature_time_order_and_time_level_validation(self):
        malformed = Function("q")(self.x, self.t)
        malformed_problem = thalassa.PDEProblem(
            (thalassa.PDEEquation(malformed, Eq(D(malformed, self.t), 0)),),
            self.t,
            self.domain,
            {malformed: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(0.1, ()),
        )
        with self.assertRaises(thalassa.ModelValidationError):
            thalassa.compile_program(malformed_problem, thalassa.UniformGrid((8,)))

        for derivative in (D(self.u, (self.t, 2)), thalassa.old(D(self.u, self.t))):
            problem = thalassa.PDEProblem(
                (thalassa.PDEEquation(self.u, Eq(derivative, 0)),),
                self.t,
                self.domain,
                {self.u: thalassa.UniformDirichlet()},
                discretization=thalassa.TwoLevelDiscretization(0.1, ()),
            )
            with self.subTest(derivative=derivative), self.assertRaises(thalassa.ModelValidationError):
                thalassa.compile_program(problem, thalassa.UniformGrid((8,)))

    def test_schedule_scc_and_topological_layers(self):
        v = Function("v")(self.t, self.x)
        boundaries = {self.u: thalassa.UniformDirichlet(), v: thalassa.UniformDirichlet()}
        coupled = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(self.u, Eq(thalassa.new(self.u) + thalassa.new(v), 3)),
                thalassa.PDEEquation(v, Eq(thalassa.new(self.u) - thalassa.new(v), 1)),
            ),
            self.t,
            self.domain,
            boundaries,
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        solver = thalassa.compile_solver(coupled, thalassa.UniformGrid((4,)))
        self.assertEqual(solver.metadata.schedule, ((("u", "v"),),))
        self.assertEqual(solver.program.schedule[0].blocks[0].kind, thalassa.BlockKind.POINTWISE_LINEAR)

        chain = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(self.u, Eq(thalassa.new(self.u), 1)),
                thalassa.PDEEquation(v, Eq(thalassa.new(v), thalassa.new(self.u))),
            ),
            self.t,
            self.domain,
            boundaries,
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        chain_solver = thalassa.compile_solver(chain, thalassa.UniformGrid((4,)))
        self.assertEqual(chain_solver.metadata.schedule, ((("u",),), (("v",),)))

    def test_removed_api_and_compile_signatures(self):
        for name in (
            "PDESystem",
            "pde_compile",
            "SchemeEquation",
            "TwoLevelScheme",
            "backward_euler",
            "SchemeValidationError",
        ):
            self.assertFalse(hasattr(thalassa, name))
        self.assertEqual(
            tuple(inspect.signature(thalassa.compile_solver).parameters),
            ("problem", "grid", "options"),
        )
        self.assertEqual(
            tuple(inspect.signature(thalassa.verify).parameters),
            ("problem", "grid", "options"),
        )
        with self.assertRaises((TypeError, thalassa.ModelValidationError)):
            thalassa.compile_solver(self.heat(), object(), thalassa.UniformGrid((8,)))


if __name__ == "__main__":
    unittest.main()
