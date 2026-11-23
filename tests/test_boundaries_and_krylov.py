import unittest

import torch

from thalassa.ir import BoundaryIR
from thalassa.runtime import (
    bicgstab,
    conjugate_gradient,
    extend_ghosts,
    ghost_shift,
    projected_conjugate_gradient,
)


class BoundaryAndKrylovTests(unittest.TestCase):
    def test_periodic_ghosts_and_shifts_wrap_exactly(self):
        values = torch.arange(4)
        self.assertTrue(
            torch.equal(
                extend_ghosts(values, (2,), kind="periodic"),
                torch.tensor([2, 3, 0, 1, 2, 3, 0, 1]),
            )
        )
        for offset in range(-4, 5):
            self.assertTrue(
                torch.equal(
                    ghost_shift(values, (offset,), "periodic"),
                    torch.roll(values, -offset),
                )
            )

        plane = torch.arange(12).reshape(3, 4)
        for offset in ((-1, -2), (1, 2), (3, -4)):
            expected = torch.roll(
                plane,
                shifts=tuple(-item for item in offset),
                dims=(0, 1),
            )
            self.assertTrue(
                torch.equal(ghost_shift(plane, offset, "periodic"), expected)
            )

    def test_dirichlet_and_neumann_radius_two(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.equal(ghost_shift(values, (-1,), "dirichlet", 5), torch.tensor([9.0, 1.0, 2.0])))
        self.assertTrue(torch.equal(ghost_shift(values, (1,), "dirichlet", 5), torch.tensor([2.0, 3.0, 7.0])))
        self.assertTrue(torch.equal(ghost_shift(values, (-2,), "neumann"), torch.tensor([2.0, 1.0, 1.0])))
        self.assertTrue(torch.equal(ghost_shift(values, (2,), "neumann"), torch.tensor([3.0, 3.0, 2.0])))

    def test_axis_order_corners_and_shape(self):
        values = torch.arange(2 * 3, dtype=torch.float64).reshape(2, 3)
        padded = extend_ghosts(values, (1, 2), kind="dirichlet", value=2.0)
        self.assertEqual(padded.shape, (4, 7))
        for offset in ((-1, 0), (0, 2), (1, -2), (1, 2)):
            shifted = ghost_shift(values, offset, BoundaryIR("dirichlet", 2.0))
            self.assertEqual(shifted.shape, values.shape)
        x_shift = ghost_shift(values, (1, 0), "neumann")
        y_shift = ghost_shift(values, (0, 1), "neumann")
        self.assertTrue(torch.equal(x_shift[-1], values[-1]))
        self.assertTrue(torch.equal(y_shift[:, -1], values[:, -1]))

    def test_batched_cg_matches_dense(self):
        matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        rhs = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
        result = conjugate_gradient(
            lambda value: value @ matrix.T,
            rhs,
            batch_ndim=1,
            rtol=1e-12,
            max_iterations=10,
        )
        expected = torch.linalg.solve(matrix, rhs.T).T
        self.assertTrue(torch.all(result.converged))
        self.assertTrue(torch.allclose(result.solution, expected, atol=1e-12, rtol=1e-12))

    def test_bicgstab_matches_nonsymmetric_dense(self):
        matrix = torch.tensor([[3.0, 2.0], [0.0, 1.0]], dtype=torch.float64)
        rhs = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
        result = bicgstab(
            lambda value: value @ matrix.T,
            rhs,
            batch_ndim=1,
            rtol=1e-12,
            max_iterations=10,
        )
        expected = torch.linalg.solve(matrix, rhs.T).T
        self.assertTrue(torch.all(result.converged))
        self.assertTrue(torch.allclose(result.solution, expected, atol=1e-12, rtol=1e-12))

    def test_projected_cg_and_zero_rhs(self):
        # Periodic graph Laplacian used only to exercise the mean-zero primitive.
        matrix = torch.tensor([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])
        rhs = torch.tensor([1.0, -1.0, 0.0])
        result = projected_conjugate_gradient(
            lambda value: matrix @ value,
            rhs,
            spatial_ndim=1,
            batch_ndim=0,
            max_iterations=10,
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(float(result.solution.mean()), 0.0, places=6)
        zero = conjugate_gradient(lambda value: value, torch.zeros(4), batch_ndim=0)
        self.assertTrue(zero.converged)
        self.assertEqual(int(zero.iterations), 0)


if __name__ == "__main__":
    unittest.main()
