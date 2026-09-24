"""Recompute selected NS-0003 outputs from the archived baseline field arrays.

This checks postprocessing only; the production CFD solver is not in this
public companion. Wall shear and residual gates come from the frozen record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "NS-0003"


def check(plot: bool = False) -> dict:
    reference = json.loads((EVIDENCE / "baseline_verification.json").read_text())
    with np.load(EVIDENCE / "field_data.npz") as field:
        zc = field["zc"]
        rc = field["rc"]
        pressure = field["pressure_Pa"]
        uz = field["uz_m_s"]
        ur = field["ur_m_s"]
        assert all(a.shape == (160, 40) for a in (zc, rc, pressure, uz, ur))
        assert all(np.isfinite(a).all() for a in (pressure, uz, ur))
        # Matches the frozen runner: arithmetic radial-cell mean at inlet/outlet.
        mean_pressure = pressure.mean(axis=1)
        pressure_drop = float(mean_pressure[0] - mean_pressure[-1])
        peak_speed = float(np.hypot(uz, ur).max())
        np.testing.assert_allclose(pressure_drop, reference["pressure_drop_Pa"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(peak_speed, reference["peak_velocity_m_s"], rtol=0, atol=1e-8)
        assert reference["acceptance"]["solver_gate_pass"] is True
        station = int(np.argmin(np.abs(zc[:, 0] - 0.05)))
        result = {
            "case": reference["case"],
            "pressure_drop_Pa": pressure_drop,
            "peak_velocity_m_s": peak_speed,
            "velocity_profile_station_mm": float(zc[station, 0] * 1000),
            "cell_shape": list(pressure.shape),
            "recorded_solver_gate_pass": reference["acceptance"]["solver_gate_pass"],
        }
        if plot:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(zc[:, 0] * 1000, mean_pressure, color="#1c5063")
            axes[0].set(xlabel="Axial position z (mm)", ylabel="Radial-cell mean pressure (Pa)")
            axes[1].plot(rc[station] * 1000, uz[station], color="#0ca58c")
            axes[1].set(xlabel="Radius r (mm)", ylabel="Axial velocity (m/s)")
            for axis in axes:
                axis.grid(alpha=0.2)
            fig.suptitle("NS-0003 baseline · saved CFD fields")
            fig.tight_layout()
            out = ROOT / "generated" / "NS-0003"
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / "profiles.png", dpi=160)
            plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Save pressure and velocity profiles under generated/NS-0003/")
    print(json.dumps(check(parser.parse_args().plot), indent=2))
