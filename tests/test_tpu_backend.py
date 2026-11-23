from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from sympy import Derivative as D
from sympy import Eq, Function, symbols

import thalassa
import thalassa.tpu as tpu
from thalassa.runtime import (
    CompiledSolver,
    bicgstab,
    conjugate_gradient,
    projected_conjugate_gradient,
)


def _heat_problem(*, nonlinear: bool = False):
    time, axis = symbols("t x")
    field = Function("u")(time, axis)
    right = thalassa.new(field**3) if nonlinear else thalassa.new(D(field, (axis, 2)))
    problem = thalassa.PDEProblem(
        thalassa.PDEEquation(field, Eq(D(field, time), right)),
        time,
        thalassa.CartesianDomain(thalassa.Axis(axis, 0, 1)),
        {field: thalassa.UniformDirichlet()},
        discretization=thalassa.TwoLevelDiscretization(
            1e-3,
            () if nonlinear else thalassa.centered(D(field, (axis, 2))),
        ),
    )
    return problem


class TPUBackendTests(unittest.TestCase):
    def tearDown(self):
        torch._dynamo.reset()

    def test_public_exports_and_lazy_dependency_error(self):
        self.assertIs(thalassa.compile_tpu_solver, tpu.compile_tpu_solver)
        self.assertIs(thalassa.TPUCompiledSolver, tpu.TPUCompiledSolver)
        with mock.patch.object(
            tpu.importlib,
            "import_module",
            side_effect=ImportError("torch_xla is absent"),
        ):
            with self.assertRaisesRegex(thalassa.CompilationError, r"thalassa\[tpu\]"):
                thalassa.compile_tpu_solver(
                    _heat_problem(), thalassa.UniformGrid(8)
                )

    def test_version_device_and_dtype_validation(self):
        mismatched = SimpleNamespace(__version__="0.0.1", device=lambda _: "xla:0")
        with mock.patch.object(tpu.importlib, "import_module", return_value=mismatched):
            with self.assertRaisesRegex(thalassa.CompilationError, "must match"):
                tpu._load_torch_xla()

        with self.assertRaisesRegex(TypeError, "float64"):
            thalassa.compile_tpu_solver(
                _heat_problem(), thalassa.UniformGrid(8), dtype=torch.float64
            )
        with mock.patch.object(
            tpu, "_load_torch_xla", return_value=mismatched
        ):
            with self.assertRaisesRegex(thalassa.CompilationError, "XLA device"):
                thalassa.compile_tpu_solver(
                    _heat_problem(), thalassa.UniformGrid(8), device="cpu"
                )

    def test_openxla_fullgraph_static_compiler_arguments(self):
        fake_xla = SimpleNamespace(__version__=torch.__version__, device=mock.Mock(return_value="xla:0"))
        compiled_callable = mock.Mock()
        with (
            mock.patch.object(tpu, "_load_torch_xla", return_value=fake_xla),
            mock.patch.object(
                CompiledSolver,
                "to",
                autospec=True,
                side_effect=lambda solver, **_: solver,
            ),
            mock.patch.object(
                tpu.torch, "compile", return_value=compiled_callable
            ) as compile_mock,
        ):
            solver = thalassa.compile_tpu_solver(
                _heat_problem(), thalassa.UniformGrid(8)
            )
        fake_xla.device.assert_called_once_with(0)
        compiled_module = compile_mock.call_args.args[0]
        self.assertIsInstance(compiled_module, CompiledSolver)
        self.assertEqual(
            compile_mock.call_args.kwargs,
            {"backend": "openxla", "fullgraph": True, "dynamic": False},
        )
        self.assertEqual(solver.dtype, torch.bfloat16)
        self.assertEqual(solver.device, torch.device("xla:0"))

    def test_unsupported_blocks_and_solve_api(self):
        nonlinear = thalassa.compile_program(
            _heat_problem(nonlinear=True), thalassa.UniformGrid(8)
        )
        with self.assertRaisesRegex(thalassa.CompilationError, "nonlinear"):
            tpu._compile_tpu_program(
                nonlinear, dtype=torch.bfloat16, device=torch.device("cpu")
            )

        time, axis = symbols("t x")
        fields = tuple(Function(f"v{index}")(time, axis) for index in range(5))
        coefficients = tuple(
            tuple(Function(f"a{row}_{column}")(axis) for column in range(5))
            for row in range(5)
        )
        variable_problem = thalassa.PDEProblem(
            tuple(
                thalassa.PDEEquation(
                    fields[row],
                    Eq(
                        sum(
                            coefficients[row][column]
                            * thalassa.new(fields[column])
                            for column in range(5)
                        ),
                        thalassa.old(fields[row]),
                    ),
                )
                for row in range(5)
            ),
            time,
            thalassa.CartesianDomain(thalassa.Axis(axis, 0, 1)),
            {field: thalassa.UniformDirichlet() for field in fields},
            coefficient_fields=tuple(
                thalassa.CoefficientField(coefficient)
                for row in coefficients
                for coefficient in row
            ),
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        variable_program = thalassa.compile_program(
            variable_problem, thalassa.UniformGrid(2)
        )
        with self.assertRaisesRegex(thalassa.CompilationError, "only through 4x4"):
            tpu._compile_tpu_program(
                variable_program,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )

        eager = thalassa.compile_solver(
            _heat_problem(), thalassa.UniformGrid(8)
        ).to(dtype=torch.float32)
        with mock.patch.object(tpu.torch, "compile", return_value=eager):
            solver = tpu.TPUCompiledSolver(
                eager, dtype=torch.float32, device=torch.device("cpu")
            )
        with self.assertRaisesRegex(thalassa.CompilationError, "host synchronization"):
            solver.solve(torch.zeros(1, 8))
        with self.assertRaisesRegex(TypeError, "dtype"):
            solver(torch.zeros(1, 8, dtype=torch.float64))

    def test_ir_default_scaling_and_scaled_buffer_preflight(self):
        time, axis = symbols("t x")
        field = Function("u")(time, axis)
        problem = thalassa.PDEProblem(
            thalassa.PDEEquation(
                field,
                Eq(131_008 * thalassa.new(field), thalassa.old(field)),
            ),
            time,
            thalassa.CartesianDomain(thalassa.Axis(axis, 0, 1)),
            {field: thalassa.UniformDirichlet()},
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        program = thalassa.compile_program(problem, thalassa.UniformGrid(4))
        block = program.schedule[0].blocks[0]
        self.assertEqual(block.residual_scale, "1/131072")
        serialized = program.to_dict()
        del serialized["schedule"][0]["blocks"][0]["residual_scale"]
        self.assertEqual(
            thalassa.ProgramIR.from_dict(serialized)
            .schedule[0]
            .blocks[0]
            .residual_scale,
            "1",
        )
        tpu._preflight_buffers(CompiledSolver(program), torch.float16)

    def test_reduced_precision_krylov_uses_fp32_recurrences(self):
        options = thalassa.CompileOptions(
            krylov=thalassa.KrylovOptions(
                rtol_float16=2e-3,
                rtol_bfloat16=3e-2,
            )
        )
        option_values = dict(
            thalassa.compile_program(
                _heat_problem(), thalassa.UniformGrid(8), options
            ).options
        )
        self.assertEqual(option_values["krylov_rtol_float16"], 2e-3)
        self.assertEqual(option_values["krylov_rtol_bfloat16"], 3e-2)

        for dtype, rtol in (
            (torch.float16, 1e-3),
            (torch.bfloat16, 1e-2),
        ):
            with self.subTest(dtype=dtype):
                spd = torch.tensor(
                    [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
                    dtype=dtype,
                )
                general = torch.tensor(
                    [[3.0, 2.0, 0.0], [0.0, 2.0, -1.0], [1.0, 0.0, 1.0]],
                    dtype=dtype,
                )
                rhs = torch.tensor(
                    [[1.0, -2.0, 3.0], [2.0, 1.0, -1.0]], dtype=dtype
                )
                matvec_dtypes: list[torch.dtype] = []

                def spd_operator(value):
                    matvec_dtypes.append(value.dtype)
                    return value @ spd.T

                cg = conjugate_gradient(
                    spd_operator,
                    rhs,
                    batch_ndim=1,
                    max_iterations=20,
                    rtol=rtol,
                )
                bi = bicgstab(
                    lambda value: value @ general.T,
                    rhs,
                    batch_ndim=1,
                    max_iterations=30,
                    rtol=rtol,
                )
                laplacian = torch.tensor(
                    [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]],
                    dtype=dtype,
                )
                projected = projected_conjugate_gradient(
                    lambda value: value @ laplacian.T,
                    torch.tensor(
                        [[1.0, -1.0, 0.0], [2.0, -1.0, -1.0]],
                        dtype=dtype,
                    ),
                    spatial_ndim=1,
                    batch_ndim=1,
                    max_iterations=10,
                    rtol=rtol,
                )
                self.assertTrue(torch.all(cg.converged))
                self.assertTrue(torch.all(bi.converged))
                self.assertTrue(torch.all(projected.converged))
                self.assertEqual(cg.solution.dtype, dtype)
                self.assertEqual(bi.solution.dtype, dtype)
                self.assertEqual(cg.residual_norm.dtype, torch.float32)
                self.assertEqual(set(matvec_dtypes), {dtype})

    def test_true_residual_cannot_retain_recursive_false_positive(self):
        matrix = torch.tensor(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        )
        rhs = torch.tensor([1.0, -2.0, 3.0])
        stopped_early = conjugate_gradient(
            lambda value: matrix @ value,
            rhs,
            batch_ndim=0,
            max_iterations=2,
            rtol=1e-7,
        )
        self.assertFalse(stopped_early.converged)
        completed = conjugate_gradient(
            lambda value: matrix @ value,
            rhs,
            batch_ndim=0,
            max_iterations=3,
            rtol=1e-7,
        )
        self.assertTrue(completed.converged)
        self.assertLessEqual(
            completed.residual_norm,
            1e-7 * torch.linalg.vector_norm(rhs),
        )


if __name__ == "__main__":
    unittest.main()
