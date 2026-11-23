import math
import unittest

import torch

from examples.implicit_heat import (
    analytical_solution,
    convergence_study,
    run_heat,
)


class ImplicitHeatTests(unittest.TestCase):
    def test_analytical_solution_matches_the_initial_condition(self):
        cells = 16
        centers = (torch.arange(cells, dtype=torch.float64) + 0.5) / cells
        x, y = torch.meshgrid(centers, centers, indexing="ij")
        expected = (
            1.0
            + torch.cos(math.pi * x)
            + 0.5 * torch.cos(2.0 * math.pi * x) * torch.cos(math.pi * y)
        )

        actual = analytical_solution(cells, 0.0, 0.1)

        self.assertEqual(actual.shape, (1, cells, cells))
        torch.testing.assert_close(actual[0], expected, rtol=1e-14, atol=1e-14)

    def test_implicit_step_converges_and_preserves_the_mean(self):
        initial = analytical_solution(16, 0.0, 0.1)
        result = run_heat(16, dt=1e-3, steps=1, diffusivity=0.1)

        self.assertTrue(bool(result.converged.item()))
        self.assertAlmostEqual(float(result.state.mean()), float(initial.mean()), places=12)
        self.assertLess(
            float(torch.linalg.vector_norm(result.state - analytical_solution(16, 1e-3, 0.1))),
            3e-4,
        )

    def test_convergence_study_is_second_order(self):
        samples = convergence_study((16, 32, 64))

        errors = tuple(sample.relative_error for sample in samples)
        self.assertTrue(
            all(fine < coarse for coarse, fine in zip(errors, errors[1:], strict=False))
        )
        for sample in samples[1:]:
            self.assertIsNotNone(sample.observed_order)
            self.assertGreater(sample.observed_order, 1.8)
            self.assertLess(sample.observed_order, 2.2)


if __name__ == "__main__":
    unittest.main()
