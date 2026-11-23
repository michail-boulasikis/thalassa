from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import torch
from sympy import Eq, Function, symbols

import thalassa
from thalassa.ir import (
    CoefficientRef,
    GridRef,
    Literal,
    NaryOp,
    ScalarRef,
    TimeLevel,
)


def _domain(x):
    return thalassa.CartesianDomain(thalassa.Axis(x, 0, 1))


def _constant_solver(matrix: torch.Tensor, cells: int = 5):
    t, x = symbols("t x")
    size = matrix.shape[0]
    fields = tuple(Function(f"u{index}")(t, x) for index in range(size))
    equations = tuple(
        thalassa.PDEEquation(
            fields[row],
            Eq(
                sum(
                    float(matrix[row, column])
                    * thalassa.new(fields[column])
                    for column in range(size)
                ),
                thalassa.old(fields[row]),
            ),
        )
        for row in range(size)
    )
    problem = thalassa.PDEProblem(
        equations,
        t,
        _domain(x),
        {field: thalassa.UniformDirichlet() for field in fields},
        discretization=thalassa.TwoLevelDiscretization(1, ()),
    )
    return thalassa.compile_solver(problem, thalassa.UniformGrid((cells,)))


def _variable_solver(size: int, cells: int = 6):
    t, x = symbols("t x")
    fields = tuple(Function(f"u{index}")(t, x) for index in range(size))
    coefficient_expressions = tuple(
        tuple(
            Function(f"a{row}_{column}")(x) for column in range(size)
        )
        for row in range(size)
    )
    equations = tuple(
        thalassa.PDEEquation(
            fields[row],
            Eq(
                sum(
                    coefficient_expressions[row][column]
                    * thalassa.new(fields[column])
                    for column in range(size)
                ),
                thalassa.old(fields[row]),
            ),
        )
        for row in range(size)
    )
    coefficients = tuple(
        thalassa.CoefficientField(expression)
        for row in coefficient_expressions
        for expression in row
    )
    problem = thalassa.PDEProblem(
        equations,
        t,
        _domain(x),
        {field: thalassa.UniformDirichlet() for field in fields},
        coefficient_fields=coefficients,
        discretization=thalassa.TwoLevelDiscretization(1, ()),
    )
    solver = thalassa.compile_solver(problem, thalassa.UniformGrid((cells,)))
    return solver, coefficients


def _pivoting_matrices(size: int, cells: int, dtype: torch.dtype):
    generator = torch.Generator().manual_seed(700 + size)
    matrix = 0.1 * torch.randn(
        cells, size, size, generator=generator, dtype=dtype
    )
    matrix = matrix + (size + 1) * torch.eye(size, dtype=dtype)
    matrix[0, 0, 0] = 0
    matrix[0, 1, 0] = size + 2
    matrix[1, 0, 0] = size + 3
    matrix[1, 1:, 0] = 0.01
    return matrix


def _matrix_inputs(matrix: torch.Tensor):
    size = matrix.shape[-1]
    return tuple(
        matrix[:, row, column]
        for row in range(size)
        for column in range(size)
    )


def _reference_solution(matrix: torch.Tensor, right_hand_side: torch.Tensor):
    matrix_batch = matrix.unsqueeze(0).expand(
        right_hand_side.shape[0], *matrix.shape
    )
    rhs_last = right_hand_side.movedim(1, -1)
    return torch.linalg.solve(matrix_batch, rhs_last).movedim(-1, 1)


class PointwiseSystemTests(unittest.TestCase):
    def tearDown(self):
        torch._dynamo.reset()

    def test_compiler_extracts_and_serializes_exact_coefficient_matrices(self):
        t, x, alpha = symbols("t x alpha")
        u = Function("u")(t, x)
        v = Function("v")(t, x)
        coefficient = Function("c")(x)
        problem = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    u,
                    Eq(
                        2 * thalassa.new(u) + alpha * thalassa.new(v),
                        thalassa.old(u),
                    ),
                ),
                thalassa.PDEEquation(
                    v,
                    Eq(
                        coefficient * thalassa.new(u)
                        + (thalassa.old(u) + 3) * thalassa.new(v),
                        thalassa.old(v),
                    ),
                ),
            ),
            t,
            _domain(x),
            {u: thalassa.UniformDirichlet(), v: thalassa.UniformDirichlet()},
            parameters=(thalassa.ScalarParameter(alpha),),
            coefficient_fields=(thalassa.CoefficientField(coefficient),),
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        program = thalassa.compile_program(problem, thalassa.UniformGrid((4,)))
        block = program.schedule[0].blocks[0]
        self.assertEqual(
            block.pointwise_coefficients,
            (
                (Literal("2"), ScalarRef("alpha")),
                (
                    CoefficientRef("c", TimeLevel.OLD, (0,)),
                    NaryOp(
                        "add",
                        (
                            Literal("3"),
                            GridRef(0, TimeLevel.OLD, (0,)),
                        ),
                    ),
                ),
            ),
        )
        serialized = program.to_dict()
        self.assertIn(
            "pointwise_coefficients", serialized["schedule"][0]["blocks"][0]
        )
        self.assertEqual(thalassa.ProgramIR.from_dict(serialized), program)

        w = Function("w")(t, x)
        sparse = thalassa.PDEProblem(
            (
                thalassa.PDEEquation(
                    u, Eq(2 * thalassa.new(u) + alpha * thalassa.new(v), 1)
                ),
                thalassa.PDEEquation(
                    v, Eq(3 * thalassa.new(v) + alpha * thalassa.new(w), 1)
                ),
                thalassa.PDEEquation(
                    w, Eq(alpha * thalassa.new(u) + 4 * thalassa.new(w), 1)
                ),
            ),
            t,
            _domain(x),
            {
                u: thalassa.UniformDirichlet(),
                v: thalassa.UniformDirichlet(),
                w: thalassa.UniformDirichlet(),
            },
            parameters=(thalassa.ScalarParameter(alpha),),
            discretization=thalassa.TwoLevelDiscretization(1, ()),
        )
        sparse_block = thalassa.compile_program(
            sparse, thalassa.UniformGrid((4,))
        ).schedule[0].blocks[0]
        self.assertEqual(sparse_block.pointwise_coefficients[0][2], Literal("0"))
        self.assertEqual(sparse_block.pointwise_coefficients[1][0], Literal("0"))
        self.assertEqual(sparse_block.pointwise_coefficients[2][1], Literal("0"))

    def test_pointwise_ir_validation_and_legacy_loading(self):
        matrix = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        solver = _constant_solver(matrix)
        block = solver.program.schedule[0].blocks[0]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            replace(block, pointwise_coefficients=((Literal("1"),),))
        invalid = (
            (GridRef(block.targets[0], TimeLevel.NEW, (0,)), Literal("0")),
            (Literal("0"), Literal("1")),
        )
        with self.assertRaisesRegex(ValueError, "new-state unknowns"):
            replace(block, pointwise_coefficients=invalid)

        serialized = solver.program.to_dict()
        del serialized["schedule"][0]["blocks"][0]["pointwise_coefficients"]
        restored = thalassa.ProgramIR.from_dict(serialized)
        self.assertIsNone(
            restored.schedule[0].blocks[0].pointwise_coefficients
        )
        legacy = thalassa.CompiledSolver(restored)
        state = torch.randn(2, 2, 5, dtype=torch.float64)
        expected = _reference_solution(
            matrix.expand(5, 2, 2), state
        )
        original_eye = torch.eye
        with (
            mock.patch(
                "thalassa.runtime.torch.eye", wraps=original_eye
            ) as eye,
            mock.patch(
                "thalassa.runtime.torch.linalg.solve_ex",
                side_effect=AssertionError("legacy 2x2 used solve_ex"),
            ),
        ):
            actual = legacy.step(state)
        self.assertEqual(eye.call_count, 1)
        self.assertTrue(torch.allclose(actual.state, expected, atol=1e-12))

    def test_constant_lu_is_unrolled_for_all_sizes_and_fullgraph(self):
        matrices = (
            torch.tensor([[0.1, 2.0], [1.0, 3.0]], dtype=torch.float64),
            torch.tensor(
                [
                    [6.0, 1.0, 0.5, 0.0, -0.25],
                    [0.5, 7.0, 1.0, 0.25, 0.0],
                    [0.0, 0.5, 8.0, 1.0, 0.25],
                    [0.25, 0.0, 0.5, 9.0, 1.0],
                    [1.0, 0.25, 0.0, 0.5, 10.0],
                ],
                dtype=torch.float64,
            ),
        )
        for matrix in matrices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(size=matrix.shape[0], dtype=dtype):
                    solver = _constant_solver(matrix)
                    state = torch.randn(
                        3, matrix.shape[0], 5, dtype=dtype
                    )
                    expected = _reference_solution(
                        matrix.to(dtype).expand(5, *matrix.shape), state
                    )
                    with (
                        mock.patch(
                            "thalassa.runtime.torch.eye",
                            side_effect=AssertionError("constant solve built a basis"),
                        ),
                        mock.patch(
                            "thalassa.runtime.torch.linalg.solve_ex",
                            side_effect=AssertionError("constant solve used solve_ex"),
                        ),
                    ):
                        actual = solver.step(state)
                    tolerance = 2e-5 if dtype == torch.float32 else 1e-12
                    self.assertTrue(
                        torch.allclose(actual.state, expected, atol=tolerance)
                    )
                    self.assertTrue(torch.all(actual.converged))
                    if matrix.shape[0] == 2:
                        graphs = []

                        def capture(graph_module, _example_inputs):
                            graphs.append(graph_module)
                            return graph_module.forward

                        compiled = torch.compile(
                            solver, backend=capture, fullgraph=True
                        )(state)
                        self.assertTrue(
                            torch.allclose(
                                compiled.state, actual.state, atol=tolerance
                            )
                        )
                        graph_text = "\n".join(
                            str(graph.graph) for graph in graphs
                        )
                        self.assertNotIn("solve_ex", graph_text)
                        self.assertNotIn("linalg_solve", graph_text)
                        self.assertNotIn("aten.eye", graph_text)
                        roundtrip = thalassa.CompiledSolver(
                            thalassa.ProgramIR.from_dict(
                                solver.program.to_dict()
                            )
                        ).step(state)
                        self.assertTrue(
                            torch.allclose(
                                roundtrip.state, actual.state, atol=tolerance
                            )
                        )

    def test_variable_2x2_through_4x4_match_dense_reference(self):
        for size in range(2, 5):
            with self.subTest(size=size):
                cells = 6
                solver, _ = _variable_solver(size, cells)
                matrix = _pivoting_matrices(size, cells, torch.float64)
                inputs = _matrix_inputs(matrix)
                state = torch.randn(2, size, cells, dtype=torch.float64)
                expected = _reference_solution(matrix, state)
                with (
                    mock.patch(
                        "thalassa.runtime.torch.eye",
                        side_effect=AssertionError("variable solve built a basis"),
                    ),
                    mock.patch(
                        "thalassa.runtime.torch.linalg.solve_ex",
                        side_effect=AssertionError("small variable solve used solve_ex"),
                    ),
                ):
                    actual = solver.step(state, *inputs)
                self.assertTrue(
                    torch.allclose(actual.state, expected, atol=1e-11)
                )
                self.assertTrue(torch.all(actual.converged))
                if size == 4:
                    graphs = []

                    def capture(graph_module, _example_inputs):
                        graphs.append(graph_module)
                        return graph_module.forward

                    compiled = torch.compile(
                        solver, backend=capture, fullgraph=True
                    )(state, *inputs)
                    self.assertTrue(
                        torch.allclose(compiled.state, expected, atol=1e-11)
                    )
                    graph_text = "\n".join(
                        str(graph.graph) for graph in graphs
                    )
                    self.assertNotIn("solve_ex", graph_text)
                    self.assertNotIn("linalg_solve", graph_text)
                    self.assertNotIn("aten.eye", graph_text)

    def test_singular_and_nonfinite_matrices_break_down_safely(self):
        singular = torch.tensor(
            [[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64
        )
        constant = _constant_solver(singular)
        state = torch.randn(2, 2, 5, dtype=torch.float64)
        result = constant.solve(
            state, raise_on_nonconvergence=False
        )
        self.assertTrue(torch.all(torch.isfinite(result.state)))
        self.assertFalse(result.converged)
        self.assertTrue(torch.all(result.diagnostics[0].breakdown))

        block = constant.program.schedule[0].blocks[0]
        nonfinite_coefficients = (
            (Literal("nan"), block.pointwise_coefficients[0][1]),
            block.pointwise_coefficients[1],
        )
        nonfinite_program = replace(
            constant.program,
            schedule=(
                replace(
                    constant.program.schedule[0],
                    blocks=(
                        replace(
                            block,
                            pointwise_coefficients=nonfinite_coefficients,
                        ),
                    ),
                ),
            ),
        )
        nonfinite_constant = thalassa.CompiledSolver(nonfinite_program)
        nonfinite_result = nonfinite_constant.solve(
            state, raise_on_nonconvergence=False
        )
        self.assertTrue(torch.all(torch.isfinite(nonfinite_result.state)))
        self.assertFalse(nonfinite_result.converged)
        self.assertTrue(torch.all(nonfinite_result.diagnostics[0].breakdown))

        variable, coefficients = _variable_solver(2, 5)
        matrix = torch.eye(2, dtype=torch.float64).expand(5, 2, 2).clone()
        matrix[0] = singular
        matrix[1, 0, 0] = float("nan")
        inputs = _matrix_inputs(matrix)
        coefficient_mapping = {
            coefficient: value
            for coefficient, value in zip(coefficients, inputs, strict=True)
        }
        variable_result = variable.solve(
            state,
            coefficient_fields=coefficient_mapping,
            raise_on_nonconvergence=False,
        )
        self.assertTrue(torch.all(torch.isfinite(variable_result.state)))
        self.assertFalse(variable_result.converged)
        self.assertTrue(torch.all(variable_result.diagnostics[0].breakdown))

    def test_variable_5x5_retains_solve_ex_fallback(self):
        size = 5
        cells = 6
        solver, _ = _variable_solver(size, cells)
        matrix = _pivoting_matrices(size, cells, torch.float64)
        inputs = _matrix_inputs(matrix)
        state = torch.randn(2, size, cells, dtype=torch.float64)
        expected = _reference_solution(matrix, state)
        original_solve_ex = torch.linalg.solve_ex
        with mock.patch(
            "thalassa.runtime.torch.linalg.solve_ex",
            wraps=original_solve_ex,
        ) as solve_ex:
            actual = solver.step(state, *inputs)
        self.assertEqual(solve_ex.call_count, 1)
        self.assertTrue(torch.allclose(actual.state, expected, atol=1e-11))
        self.assertTrue(torch.all(actual.converged))


if __name__ == "__main__":
    unittest.main()
