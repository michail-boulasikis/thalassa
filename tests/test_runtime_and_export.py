import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import sympy as sp
import torch
from sympy import Derivative as D
from sympy import Eq, Function, symbols

import thalassa


class RuntimeAndExportTests(unittest.TestCase):
    def setUp(self):
        self.t, self.x = symbols("t x")
        self.u = Function("u")(self.t, self.x)

    def test_explicit_neumann_step_and_float16(self):
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u,
                    Eq(D(self.u, self.t) + thalassa.old(D(self.u, self.x)), 0),
                ),
            ),
            self.t,
            thalassa.CartesianDomain(thalassa.Axis(self.x, 0, 1)),
            {self.u: thalassa.HomogeneousNeumann()},
            discretization=thalassa.TwoLevelDiscretization(
                sp.Rational(1, 100),
                (thalassa.centered(D(self.u, self.x)),),
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((8,)))
        state = torch.arange(8, dtype=torch.float16)[None, :]
        output = solver.step(state)
        self.assertEqual(output.state.shape, state.shape)
        self.assertEqual(output.state.dtype, torch.float16)
        self.assertTrue(output.converged)

    def test_implicit_heat_and_reduced_precision(self):
        nu = symbols("nu", positive=True)
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u,
                    Eq(D(self.u, self.t), thalassa.new(nu * D(self.u, (self.x, 2)))),
                ),
            ),
            self.t,
            thalassa.CartesianDomain(thalassa.Axis(self.x, 0, 1)),
            {self.u: thalassa.UniformDirichlet()},
            (thalassa.ScalarParameter(nu),),
            discretization=thalassa.TwoLevelDiscretization(
                sp.Rational(1, 100),
                (thalassa.centered(D(self.u, (self.x, 2))),),
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((16,)))
        centers = (torch.arange(16, dtype=torch.float64) + 0.5) / 16
        state = torch.sin(torch.pi * centers)[None, :]
        output = solver.step(state, 0.1)
        self.assertTrue(output.converged)
        self.assertLess(float(output.residual_norm), 1e-8)
        compiled_output = torch.compile(solver, backend="eager")(state, 0.1)
        self.assertTrue(compiled_output.converged)
        self.assertTrue(torch.allclose(compiled_output.state, output.state))
        for dtype in (torch.float16, torch.bfloat16):
            reduced = solver.step(state.to(dtype), 0.1)
            self.assertTrue(reduced.converged)
            self.assertEqual(reduced.state.dtype, dtype)
            self.assertEqual(reduced.residual_norm.dtype, torch.float32)
            self.assertTrue(torch.all(torch.isfinite(reduced.state)))

    def test_pure_neumann_compatibility_and_projection(self):
        pressure = Function("p")(self.t, self.x)
        forcing = Function("f")(self.x)
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    pressure,
                    Eq(-thalassa.new(D(pressure, (self.x, 2))), thalassa.old(forcing)),
                ),
            ),
            self.t,
            thalassa.CartesianDomain(thalassa.Axis(self.x, 0, 1)),
            {pressure: thalassa.HomogeneousNeumann()},
            coefficient_fields=(thalassa.CoefficientField(forcing),),
            discretization=thalassa.TwoLevelDiscretization(
                1,
                (thalassa.centered(D(pressure, (self.x, 2))),),
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((16,)))
        centers = (torch.arange(16, dtype=torch.float64) + 0.5) / 16
        compatible = torch.sin(2 * torch.pi * centers)
        output = solver.step(torch.zeros(1, 16, dtype=torch.float64), compatible)
        self.assertTrue(output.converged)
        self.assertAlmostEqual(float(output.state.mean()), 0.0, places=12)
        with self.assertRaises(thalassa.BoundaryCompatibilityError):
            solver.step(torch.zeros(1, 16, dtype=torch.float64), torch.ones(16, dtype=torch.float64))
        compiled = torch.compile(solver, backend="eager")
        incompatible = compiled(
            torch.zeros(1, 16, dtype=torch.float64),
            torch.ones(16, dtype=torch.float64),
        )
        self.assertFalse(incompatible.converged)

    def test_periodic_fourier_heat_and_poisson(self):
        cells = 16
        dt = sp.Rational(1, 100)
        nu = symbols("nu", positive=True)
        heat = thalassa.PDEProblem(
            thalassa.PDEEquation(
                self.u,
                Eq(
                    D(self.u, self.t),
                    thalassa.new(nu * D(self.u, (self.x, 2))),
                ),
            ),
            self.t,
            thalassa.CartesianDomain(thalassa.Axis(self.x, 0, 1)),
            {self.u: thalassa.Periodic()},
            parameters=thalassa.ScalarParameter(nu),
            discretization=thalassa.TwoLevelDiscretization(
                dt, thalassa.centered(D(self.u, (self.x, 2)))
            ),
        )
        heat_solver = thalassa.compile_solver(heat, thalassa.UniformGrid(cells))
        centers = (torch.arange(cells, dtype=torch.float64) + 0.5) / cells
        mode = torch.sin(2 * torch.pi * centers)
        state = (2.0 + mode)[None, :]
        viscosity = 0.2
        result = heat_solver.step(state, viscosity)
        eigenvalue = 4 * cells**2 * math.sin(math.pi / cells) ** 2
        expected = 2.0 + mode / (1.0 + viscosity * float(dt) * eigenvalue)
        self.assertTrue(result.converged)
        self.assertTrue(torch.allclose(result.state[0], expected, atol=1e-11))
        self.assertAlmostEqual(float(result.state.mean()), float(state.mean()), places=12)

        pressure = Function("p")(self.t, self.x)
        forcing = Function("f")(self.x)
        poisson = thalassa.PDEProblem(
            thalassa.PDEEquation(
                pressure,
                Eq(-thalassa.new(D(pressure, (self.x, 2))), thalassa.old(forcing)),
            ),
            self.t,
            thalassa.CartesianDomain(thalassa.Axis(self.x, 0, 1)),
            {pressure: thalassa.Periodic()},
            coefficient_fields=thalassa.CoefficientField(
                forcing, boundary=thalassa.Periodic()
            ),
            discretization=thalassa.TwoLevelDiscretization(
                1, thalassa.centered(D(pressure, (self.x, 2)))
            ),
        )
        poisson_solver = thalassa.compile_solver(
            poisson, thalassa.UniformGrid(cells)
        )
        block = poisson_solver.program.schedule[0].blocks[0]
        self.assertTrue(block.properties.constant_nullspace)
        solution = poisson_solver.step(
            torch.zeros(1, cells, dtype=torch.float64), mode
        )
        self.assertTrue(solution.converged)
        self.assertTrue(
            torch.allclose(solution.state[0], mode / eigenvalue, atol=1e-12)
        )
        self.assertAlmostEqual(float(solution.state.mean()), 0.0, places=12)
        with self.assertRaises(thalassa.BoundaryCompatibilityError):
            poisson_solver.step(
                torch.zeros(1, cells, dtype=torch.float64),
                torch.ones(cells, dtype=torch.float64),
            )

        serialized = poisson_solver.program.to_dict()
        del serialized["coefficient_boundaries"]
        restored = thalassa.ProgramIR.from_dict(serialized)
        self.assertEqual(restored.coefficient_boundaries[0].kind, "neumann")

    def test_pointwise_coupled_and_newton(self):
        v = Function("v")(self.t, self.x)
        domain = thalassa.CartesianDomain(thalassa.Axis(self.x, 0, 1))
        coupled = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u, Eq(thalassa.new(self.u) + thalassa.new(v), 3)
                ),
                thalassa.PDEEquation(
                    v, Eq(thalassa.new(self.u) - thalassa.new(v), 1)
                ),
            ),
            self.t,
            domain,
            {self.u: thalassa.UniformDirichlet(), v: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        coupled_solver = thalassa.compile_solver(coupled, thalassa.UniformGrid((5,)))
        result = coupled_solver.step(torch.zeros(2, 5))
        self.assertTrue(torch.equal(result.state[0], torch.full((5,), 2.0)))
        self.assertTrue(torch.equal(result.state[1], torch.full((5,), 1.0)))

        nonlinear = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u,
                    Eq(D(self.u, self.t) + thalassa.new(self.u**3), 0),
                ),
            ),
            self.t,
            domain,
            {self.u: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(sp.Rational(1, 100), ()),
        )
        nonlinear_solver = thalassa.compile_solver(nonlinear, thalassa.UniformGrid((5,)))
        initial = torch.full((2, 1, 5), 0.5, dtype=torch.float64)
        output = nonlinear_solver.step(initial)
        self.assertTrue(torch.all(output.converged))
        self.assertLess(float(torch.max(torch.abs(output.state + 0.01 * output.state**3 - initial))), 1e-12)
        compiled_output = torch.compile(nonlinear_solver, backend="eager")(initial)
        self.assertTrue(torch.all(compiled_output.converged))
        self.assertTrue(torch.allclose(compiled_output.state, output.state))

    def test_export_roundtrip_and_torch_compile(self):
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    self.u,
                    Eq(D(self.u, self.t) + thalassa.old(D(self.u, self.x)), 0),
                ),
            ),
            self.t,
            thalassa.CartesianDomain(thalassa.Axis(self.x, 0, 1)),
            {self.u: thalassa.Periodic()},
            discretization=thalassa.TwoLevelDiscretization(
                sp.Rational(1, 100),
                (thalassa.centered(D(self.u, self.x)),),
            ),
        )
        solver = thalassa.compile_solver(problem, thalassa.UniformGrid((8,)))
        state = torch.arange(8, dtype=torch.float32)[None, :]
        expected = solver(state, steps=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exported.py"
            first = thalassa.export_python(solver, path)
            second = thalassa.export_python(solver)
            self.assertEqual(first, second)
            specification = importlib.util.spec_from_file_location("test_exported", path)
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            actual = module.solver(state, steps=2)
            self.assertEqual(module.solver.metadata, solver.metadata)
            self.assertTrue(torch.equal(actual.state, expected.state))
        compiled = torch.compile(solver, backend="eager")
        compiled_output = compiled(state, steps=2)
        self.assertTrue(torch.equal(compiled_output.state, expected.state))


if __name__ == "__main__":
    unittest.main()
