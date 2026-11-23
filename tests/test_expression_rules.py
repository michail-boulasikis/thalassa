import dataclasses
import json
import math
import unittest

import sympy as sp
import torch
from sympy import Derivative as D

import thalassa


def _serialized_stencils(value):
    result = []
    if isinstance(value, dict):
        if value.get("type") == "stencil":
            result.append(value["entries"])
        for item in value.values():
            result.extend(_serialized_stencils(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_serialized_stencils(item))
    return result


def _serialized_expression_types(value):
    result = set()
    if isinstance(value, dict):
        if "type" in value:
            result.add(value["type"])
        for item in value.values():
            result.update(_serialized_expression_types(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_serialized_expression_types(item))
    return result


class ExpressionRuleModelTests(unittest.TestCase):
    def setUp(self):
        self.t, self.x = sp.symbols("t x")
        self.u = sp.Function("u")(self.t, self.x)
        self.du = D(self.u, self.x)
        self.domain = thalassa.CartesianDomain(thalassa.Axis(self.x, 0, 1))

    def problem(self, expression, *, differences=(), expression_rules=()):
        return thalassa.PDEProblem(
            thalassa.PDEEquation(
                self.u,
                sp.Eq(D(self.u, self.t) + thalassa.old(expression), 0),
            ),
            self.t,
            self.domain,
            {self.u: thalassa.Periodic()},
            discretization=thalassa.TwoLevelDiscretization(
                1,
                differences,
                expression_rules=expression_rules,
            ),
        )

    def test_public_api_is_immutable_and_backward_compatible(self):
        target = sp.sin(self.du)
        rule = thalassa.ExpressionRule(target, target, thalassa.centered(self.du))
        discretization = thalassa.TwoLevelDiscretization(1, expression_rules=rule)

        self.assertIs(thalassa.ExpressionRule, type(rule))
        self.assertEqual(discretization.differences, ())
        self.assertEqual(discretization.expression_rules, (rule,))
        self.assertEqual(thalassa.TwoLevelDiscretization(1).expression_rules, ())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rule.target = self.u

    def test_exact_rule_preserves_continuous_residual_and_ir(self):
        target = sp.sin(self.du)
        rule = thalassa.ExpressionRule(target, target, thalassa.centered(self.du))
        problem = self.problem(target, expression_rules=rule)
        first = thalassa.compile_program(problem, thalassa.UniformGrid(8))
        second = thalassa.compile_program(problem, thalassa.UniformGrid(8))

        self.assertEqual(first.version, "2.0")
        self.assertEqual(
            first.equations[0].continuous_residual,
            sp.srepr(problem.equations[0].continuous_residual),
        )
        self.assertEqual(
            json.dumps(first.to_dict(), sort_keys=True),
            json.dumps(second.to_dict(), sort_keys=True),
        )
        self.assertEqual(thalassa.ProgramIR.from_dict(first.to_dict()), first)
        self.assertLessEqual(
            _serialized_expression_types(first.to_dict()),
            {
                "literal",
                "grid_ref",
                "scalar_ref",
                "coefficient_ref",
                "coordinate_ref",
                "time_ref",
                "nary",
                "power",
                "call",
                "stencil",
            },
        )

    def test_local_differences_do_not_override_global_rules(self):
        target = sp.sin(self.du)
        rule = thalassa.ExpressionRule(target, target, thalassa.centered(self.du))
        problem = thalassa.PDEProblem(
            thalassa.PDEEquation(
                self.u,
                sp.Eq(
                    D(self.u, self.t)
                    + thalassa.old(target)
                    + thalassa.old(self.du),
                    0,
                ),
            ),
            self.t,
            self.domain,
            {self.u: thalassa.Periodic()},
            discretization=thalassa.TwoLevelDiscretization(
                1,
                thalassa.forward(self.du),
                expression_rules=rule,
            ),
        )
        program = thalassa.compile_program(problem, thalassa.UniformGrid(8))
        entries = _serialized_stencils(program.to_dict()["equations"][0]["residual"])
        offset_sets = {
            tuple(tuple(entry[0]) for entry in stencil)
            for stencil in entries
        }
        self.assertIn(((-1,), (1,)), offset_sets)
        self.assertIn(((0,), (1,)), offset_sets)

    def test_expression_rule_validation_errors(self):
        target = sp.sin(self.du)
        centered = thalassa.centered(self.du)

        unmatched = thalassa.ExpressionRule(sp.cos(self.du), sp.cos(self.du), centered)
        with self.assertRaisesRegex(thalassa.DiscretizationError, "does not occur exactly"):
            thalassa.verify(
                self.problem(target, differences=centered, expression_rules=unmatched),
                thalassa.UniformGrid(8),
            )

        duplicate = thalassa.ExpressionRule(target, target, centered)
        with self.assertRaisesRegex(thalassa.DiscretizationError, "Duplicate expression rule"):
            thalassa.verify(
                self.problem(target, expression_rules=(duplicate, duplicate)),
                thalassa.UniformGrid(8),
            )

        inner = thalassa.ExpressionRule(self.du, self.du, centered)
        with self.assertRaisesRegex(thalassa.DiscretizationError, "overlap structurally"):
            thalassa.verify(
                self.problem(target, expression_rules=(duplicate, inner)),
                thalassa.UniformGrid(8),
            )

        cosine = sp.cos(self.du)
        dependent = thalassa.ExpressionRule(target, cosine, centered)
        other = thalassa.ExpressionRule(cosine, cosine, centered)
        combined = thalassa.old(target) + thalassa.old(cosine)
        dependency_problem = thalassa.PDEProblem(
            thalassa.PDEEquation(
                self.u,
                sp.Eq(D(self.u, self.t) + combined, 0),
            ),
            self.t,
            self.domain,
            {self.u: thalassa.Periodic()},
            discretization=thalassa.TwoLevelDiscretization(
                1,
                expression_rules=(dependent, other),
            ),
        )
        with self.assertRaisesRegex(thalassa.DiscretizationError, "depends on"):
            thalassa.verify(dependency_problem, thalassa.UniformGrid(8))

        missing = thalassa.ExpressionRule(target, self.du)
        with self.assertRaisesRegex(thalassa.DiscretizationError, "approximation"):
            thalassa.verify(
                self.problem(target, expression_rules=missing),
                thalassa.UniformGrid(8),
            )

        duplicate_local = thalassa.ExpressionRule(
            target,
            target,
            (centered, centered),
        )
        with self.assertRaisesRegex(thalassa.DiscretizationError, "Duplicate local"):
            thalassa.verify(
                self.problem(target, expression_rules=duplicate_local),
                thalassa.UniformGrid(8),
            )

        annotated = thalassa.ExpressionRule(
            target,
            thalassa.old(target),
            centered,
        )
        with self.assertRaisesRegex(thalassa.DiscretizationError, "old/new"):
            thalassa.verify(
                self.problem(target, expression_rules=annotated),
                thalassa.UniformGrid(8),
            )

        time_dependent = thalassa.ExpressionRule(
            target,
            D(self.u, self.t),
        )
        with self.assertRaisesRegex(thalassa.DiscretizationError, "time derivative"):
            thalassa.verify(
                self.problem(target, expression_rules=time_dependent),
                thalassa.UniformGrid(8),
            )

        undeclared = thalassa.ExpressionRule(
            target,
            target + sp.Symbol("z"),
            centered,
        )
        with self.assertRaisesRegex(thalassa.DiscretizationError, "Undeclared"):
            thalassa.verify(
                self.problem(target, expression_rules=undeclared),
                thalassa.UniformGrid(8),
            )

    def test_rule_inherits_new_time_level(self):
        target = sp.sin(self.du)
        rule = thalassa.ExpressionRule(target, target, thalassa.centered(self.du))
        problem = thalassa.PDEProblem(
            thalassa.PDEEquation(
                self.u,
                sp.Eq(
                    D(self.u, self.t)
                    + thalassa.old(target)
                    + thalassa.new(target),
                    0,
                ),
            ),
            self.t,
            self.domain,
            {self.u: thalassa.Periodic()},
            discretization=thalassa.TwoLevelDiscretization(
                1,
                expression_rules=rule,
            ),
        )
        program = thalassa.compile_program(problem, thalassa.UniformGrid(8))
        self.assertEqual(
            program.schedule[0].blocks[0].kind,
            thalassa.BlockKind.NONLINEAR_IMPLICIT,
        )
        self.assertEqual(
            len(_serialized_stencils(program.to_dict()["equations"][0]["residual"])),
            2,
        )


class ArakawaJacobianTests(unittest.TestCase):
    @staticmethod
    def make_solver(cells):
        t, x, y = sp.symbols("t x y")
        omega = sp.Function("omega")(t, x, y)
        psi = sp.Function("psi")(t, x, y)
        rule = thalassa.arakawa_jacobian(omega, psi, axes=(x, y))
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    omega,
                    sp.Eq(D(omega, t) + thalassa.old(rule.target), 0),
                ),
                thalassa.PDEEquation(
                    psi,
                    sp.Eq(thalassa.new(psi), thalassa.old(psi)),
                ),
            ),
            t,
            thalassa.CartesianDomain(
                thalassa.Axis(x, 0, 1),
                thalassa.Axis(y, 0, 1),
            ),
            {omega: thalassa.Periodic(), psi: thalassa.Periodic()},
            discretization=thalassa.TwoLevelDiscretization(
                1,
                expression_rules=rule,
            ),
        )
        return thalassa.compile_solver(problem, thalassa.UniformGrid((cells, cells)))

    def test_helper_convention_and_validation(self):
        t, x, y, z = sp.symbols("t x y z")
        a = sp.Function("a")(t, x, y)
        b = sp.Function("b")(t, x, y)
        rule = thalassa.arakawa_jacobian(a, b, axes=(x, y))
        expected = D(a, x) * D(b, y) - D(a, y) * D(b, x)

        self.assertEqual(rule.target, expected)
        self.assertEqual(sp.simplify(rule.approximation.doit() - expected), 0)
        self.assertEqual(len(rule.differences), 8)
        self.assertTrue(all(item.accuracy == 2 for item in rule.differences))
        with self.assertRaises(thalassa.DiscretizationError):
            thalassa.arakawa_jacobian(a, b, axes=(x, x))
        with self.assertRaises(thalassa.DiscretizationError):
            thalassa.arakawa_jacobian(a, b, axes=(x, y, z))
        with self.assertRaises(thalassa.DiscretizationError):
            thalassa.arakawa_jacobian(a, a, axes=(x, y))

        invalid_axis = thalassa.arakawa_jacobian(a, b, axes=(x, z))
        invalid_problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    a,
                    sp.Eq(D(a, t) + thalassa.old(invalid_axis.target), 0),
                ),
                thalassa.PDEEquation(
                    b,
                    sp.Eq(thalassa.new(b), thalassa.old(b)),
                ),
            ),
            t,
            thalassa.CartesianDomain(
                thalassa.Axis(x, 0, 1),
                thalassa.Axis(y, 0, 1),
            ),
            {a: thalassa.Periodic(), b: thalassa.Periodic()},
            discretization=thalassa.TwoLevelDiscretization(
                1,
                expression_rules=invalid_axis,
            ),
        )
        with self.assertRaisesRegex(thalassa.DiscretizationError, "undeclared axis"):
            thalassa.verify(invalid_problem, thalassa.UniformGrid((8, 8)))

    def test_matches_direct_periodic_reference_and_conserves_invariants(self):
        cells = 24
        solver = self.make_solver(cells).to(dtype=torch.float64)
        torch.manual_seed(7)
        state = torch.randn(2, cells, cells, dtype=torch.float64)
        result = solver.step(state)
        jacobian = state[0] - result.state[0]

        def centered(value, axis):
            return (
                torch.roll(value, -1, dims=axis)
                - torch.roll(value, 1, dims=axis)
            ) * (cells / 2)

        omega, psi = state
        omega_x = centered(omega, 0)
        omega_y = centered(omega, 1)
        psi_x = centered(psi, 0)
        psi_y = centered(psi, 1)
        direct = (
            omega_x * psi_y
            - omega_y * psi_x
            + centered(omega * psi_y, 0)
            - centered(omega * psi_x, 1)
            + centered(psi * omega_x, 1)
            - centered(psi * omega_y, 0)
        ) / 3

        self.assertTrue(result.converged)
        torch.testing.assert_close(jacobian, direct, rtol=1e-12, atol=1e-10)
        self.assertLess(abs(float(jacobian.sum())), 1e-8)
        self.assertLess(abs(float((omega * jacobian).sum())), 1e-8)
        self.assertLess(abs(float((psi * jacobian).sum())), 1e-8)

    def test_second_order_for_nonzero_smooth_jacobian(self):
        errors = []
        for cells in (16, 32, 64):
            solver = self.make_solver(cells).to(dtype=torch.float64)
            centers = (torch.arange(cells, dtype=torch.float64) + 0.5) / cells
            x, y = torch.meshgrid(centers, centers, indexing="ij")
            omega = torch.sin(2 * torch.pi * x)
            psi = torch.sin(2 * torch.pi * y)
            state = torch.stack((omega, psi))
            result = solver.step(state)
            actual = state[0] - result.state[0]
            expected = (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x) * torch.cos(
                2 * torch.pi * y
            )
            errors.append(
                float(
                    torch.linalg.vector_norm(actual - expected)
                    / torch.linalg.vector_norm(expected)
                )
            )

        orders = [
            math.log(coarse / fine, 2)
            for coarse, fine in zip(errors, errors[1:], strict=False)
        ]
        self.assertTrue(all(1.9 < order < 2.1 for order in orders))


if __name__ == "__main__":
    unittest.main()
