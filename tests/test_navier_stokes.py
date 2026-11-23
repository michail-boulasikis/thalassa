import tempfile
import unittest
from pathlib import Path

import torch

from examples.navier_stokes import (
    convergence_study,
    periodic_streamfunction_from_vorticity,
    run_vortex_dipole,
    save_navier_output,
    vortex_dipole_state,
)


class NavierStokesTests(unittest.TestCase):
    def test_vortex_dipole_is_deterministic_zero_mean_and_seeded(self):
        first = vortex_dipole_state((32, 24), seed=42)
        repeat = vortex_dipole_state((32, 24), seed=42)
        changed = vortex_dipole_state((32, 24), seed=43)

        self.assertEqual(first.shape, (2, 32, 24))
        self.assertTrue(torch.equal(first, repeat))
        self.assertFalse(torch.equal(first, changed))
        self.assertTrue(torch.count_nonzero(first[1]).item() == 0)
        self.assertLess(abs(float(first[0].sum().item())), 1e-11)

    def test_vortex_dipole_cases_and_amplitude_scaling(self):
        easy = vortex_dipole_state(32, case="easy", seed=7)
        hard = vortex_dipole_state(32, case="hard", seed=7)
        scaled = vortex_dipole_state(32, case="easy", seed=7, amplitude_scale=2.5)

        self.assertGreater(float(hard[0].abs().max()), 2.0 * float(easy[0].abs().max()))
        torch.testing.assert_close(scaled[0], 2.5 * easy[0], rtol=1e-14, atol=1e-12)

    def test_vortex_streamfunction_solves_the_discrete_poisson_equation(self):
        omega = vortex_dipole_state(32, seed=7)[0]
        psi = periodic_streamfunction_from_vorticity(omega)
        laplacian = 32**2 * (
            torch.roll(psi, -1, 0)
            - 2.0 * psi
            + torch.roll(psi, 1, 0)
            + torch.roll(psi, -1, 1)
            - 2.0 * psi
            + torch.roll(psi, 1, 1)
        )
        torch.testing.assert_close(laplacian, -omega, rtol=1e-11, atol=1e-11)
        self.assertLess(abs(float(psi.mean())), 1e-14)

    def test_vortex_dipole_run_and_checkpoint(self):
        result = run_vortex_dipole(16, dt=1e-4, steps=1, viscosity=1e-2)
        self.assertTrue(bool(result.converged.item()))
        self.assertTrue(bool(torch.all(torch.isfinite(result.state)).item()))

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "navier.pt"
            save_navier_output(
                destination,
                result.state,
                dt=1e-4,
                steps=1,
                viscosity=1e-2,
                case="easy",
                seed=42,
                amplitude_scale=1.0,
                converged=True,
                residual_norm=float(result.residual_norm),
                iterations=int(result.iterations),
                relative_residual=float(result.relative_residual),
                tolerance_ratio=float(result.tolerance_ratio),
                precision_limited=bool(result.precision_limited),
                worst_residual_block="psi",
                worst_residual_step=0,
            )
            checkpoint = torch.load(destination, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["format"], "thalassa.navier-stokes.v1")
        self.assertEqual(checkpoint["axis_order"], ("field", "x", "y"))
        self.assertEqual(
            checkpoint["relative_residual"], float(result.relative_residual)
        )
        self.assertEqual(
            checkpoint["tolerance_ratio"], float(result.tolerance_ratio)
        )
        self.assertEqual(checkpoint["worst_residual_block"], "psi")
        self.assertEqual(checkpoint["worst_residual_step"], 0)
        torch.testing.assert_close(checkpoint["state"], result.state)

    def test_vortex_dipole_runs_with_reduced_precision_krylov(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                result = run_vortex_dipole(
                    16,
                    dt=1e-4,
                    steps=1,
                    viscosity=1e-2,
                    dtype=dtype,
                )
                self.assertEqual(result.state.dtype, dtype)
                self.assertEqual(result.residual_norm.dtype, torch.float32)
                self.assertEqual(result.relative_residual.dtype, torch.float32)
                self.assertEqual(result.tolerance_ratio.dtype, torch.float32)
                self.assertTrue(bool(result.converged.item()))
                self.assertGreaterEqual(float(result.relative_residual), 0.0)
                self.assertLessEqual(float(result.relative_residual), 1.0)
                self.assertLessEqual(float(result.tolerance_ratio), 1.0)
                self.assertTrue(bool(torch.all(torch.isfinite(result.state)).item()))

    def test_taylor_green_converges_at_second_order_through_1024_squared(self):
        samples = convergence_study()

        self.assertEqual(
            tuple(sample.cells for sample in samples),
            (32, 64, 128, 256, 512, 1024, 2048, 4096),
        )
        errors = tuple(sample.relative_error for sample in samples)
        self.assertTrue(
            all(
                fine < coarse
                for coarse, fine in zip(errors, errors[1:], strict=False)
            )
        )
        for sample in samples[1:]:
            self.assertIsNotNone(sample.observed_order)
            self.assertGreater(sample.observed_order, 1.6)
            self.assertLess(sample.observed_order, 2.1)
        self.assertLess(samples[-1].relative_error, 3.5e-6)


if __name__ == "__main__":
    unittest.main()
