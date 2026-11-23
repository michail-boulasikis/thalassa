import unittest

import torch

from examples.barenblatt_pattle import (
    BarenblattParameters,
    barenblatt_state,
    benchmark_metrics,
    convergence_study,
    run_benchmark,
    solver_metrics,
)


class BarenblattPattleTests(unittest.TestCase):
    def test_default_parameters_have_safe_support_and_expected_evolution(self):
        parameters = BarenblattParameters()

        self.assertGreater(parameters.boundary_clearance(0.1), 0.5)
        self.assertGreater(parameters.support_radius(0.1), parameters.support_radius(0.0))
        self.assertLess(parameters.peak(0.1), parameters.peak(0.0))
        self.assertAlmostEqual(parameters.exact_mass, 0.2 * torch.pi, places=14)

    def test_exact_state_and_one_step_preserve_mass_and_positivity(self):
        parameters = BarenblattParameters(exponent=1.5)
        initial = barenblatt_state(24, 0.0, parameters)
        result = run_benchmark(
            24,
            dt=5e-4,
            steps=1,
            parameters=parameters,
        )
        spacing = 2.0 * parameters.half_width / 24
        initial_mass = float(spacing**2 * initial[0].sum())
        metrics = benchmark_metrics(
            result.state,
            time=5e-4,
            parameters=parameters,
            initial_mass=initial_mass,
        )

        self.assertTrue(result.converged)
        self.assertTrue(torch.all(torch.isfinite(result.state)))
        self.assertLess(metrics.relative_mass_drift, 1e-12)
        self.assertEqual(metrics.negative_mass_fraction, 0.0)
        self.assertFalse(solver_metrics(result).breakdown)

    def test_one_step_refinement_reduces_field_and_front_errors(self):
        samples = convergence_study((16, 32, 64), dt=5e-4)

        self.assertEqual(tuple(sample.cells for sample in samples), (16, 32, 64))
        l2_errors = tuple(sample.metrics.relative_l2_error for sample in samples)
        radius_errors = tuple(
            sample.metrics.relative_radius_error for sample in samples
        )
        self.assertTrue(all(fine < coarse for coarse, fine in zip(l2_errors, l2_errors[1:])))
        self.assertLess(radius_errors[-1], radius_errors[0])
        self.assertTrue(all(sample.solver.converged for sample in samples))

    def test_smooth_interior_recovers_second_order_before_time_error_floor(self):
        samples = convergence_study((32, 64, 128), dt=1e-4)

        interior_orders = tuple(
            sample.observed_interior_l2_order for sample in samples[1:]
        )
        self.assertTrue(all(order is not None for order in interior_orders))
        self.assertTrue(all(order > 1.7 for order in interior_orders if order is not None))
        self.assertLess(
            samples[-1].metrics.relative_interior_l2_error,
            samples[-1].metrics.relative_l2_error / 50.0,
        )


if __name__ == "__main__":
    unittest.main()
