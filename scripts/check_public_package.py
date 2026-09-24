"""Run the analytical/FD and archived-field checks without pytest."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.physics.annular_baseline import (
    analytical_velocity, baseline_metrics, finite_difference_velocity,
    pressure_gradient_from_flow,
)
from scripts.check_ns0003_saved_fields import check

ri, ro, mu, flow = 0.001, 0.002, 0.0035, 100e-6 / 60.0
gradient = pressure_gradient_from_flow(flow, ri, ro, mu)
r, numerical = finite_difference_velocity(ri, ro, mu, gradient, n=161)
exact = analytical_velocity(r, ri, ro, mu, gradient)
error = float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))
assert error < 2e-6
assert abs(numerical[0]) < 1e-12 and abs(numerical[-1]) < 1e-12
assert abs(baseline_metrics()["pressure_drop_Pa"] - 736.92179) < 0.001
archived = check()
print(f"NS-0001 finite-difference relative L2 error: {error:.3e}")
print(f"NS-0003 archived pressure drop: {archived['pressure_drop_Pa']:.6f} Pa")
print(f"NS-0003 archived peak speed: {archived['peak_velocity_m_s']:.6f} m/s")
print("PASS: benchmark and saved-field checks")
