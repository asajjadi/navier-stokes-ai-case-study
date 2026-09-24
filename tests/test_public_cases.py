"""Independent numerical checks for the two public cases."""
import unittest

import numpy as np

from scripts.check_ns0003_saved_fields import check
from src.physics.annular_baseline import (
    analytical_velocity, baseline_metrics, finite_difference_velocity,
    pressure_gradient_from_flow,
)
from src.physics.axisymmetric_fvm import FVMConfig, solve


class TestPublicCases(unittest.TestCase):
    def test_finite_difference_refinement(self):
        ri, ro, mu, q = 0.001, 0.002, 0.0035, 100e-6 / 60
        G = pressure_gradient_from_flow(q, ri, ro, mu)
        errors = []
        for n in (41, 81, 161):
            r, u = finite_difference_velocity(ri, ro, mu, G, n)
            exact = analytical_velocity(r, ri, ro, mu, G)
            errors.append(np.linalg.norm(u - exact) / np.linalg.norm(exact))
            self.assertAlmostEqual(u[0], 0.0, places=12)
            self.assertAlmostEqual(u[-1], 0.0, places=12)
        self.assertTrue(all(a > b for a, b in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 2e-6)

    def test_analytical_pressure_drop(self):
        self.assertAlmostEqual(baseline_metrics()["pressure_drop_Pa"], 736.92179, delta=0.001)

    def test_iterative_solver_against_analytical(self):
        result = solve(FVMConfig(nz=41, nr=81, max_iterations=5000, tolerance=1e-7))
        self.assertTrue(result.converged)
        self.assertLess(result.velocity_profile_relative_l2_error, 5e-4)
        self.assertLess(abs(result.pressure_drop_pa - 736.92179) / 736.92179, 1e-3)

    def test_ns0003_saved_field_metrics(self):
        observed = check()
        self.assertAlmostEqual(observed["pressure_drop_Pa"], 846.2615433795254, places=6)
        self.assertAlmostEqual(observed["peak_velocity_m_s"], 0.31400295952395707, places=7)


if __name__ == "__main__":
    unittest.main()
