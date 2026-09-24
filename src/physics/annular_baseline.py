"""Analytical and finite-difference verification for concentric annular flow."""
from __future__ import annotations

import numpy as np


def pressure_gradient_from_flow(Q: float, Ri: float, Ro: float, mu: float) -> float:
    """Return G=-dp/dz [Pa/m] for steady fully developed annular flow."""
    B = Ro**4 - Ri**4 - (Ro**2 - Ri**2) ** 2 / np.log(Ro / Ri)
    return Q * 8.0 * mu / (np.pi * B)


def analytical_velocity(r, Ri: float, Ro: float, mu: float, G: float):
    """Exact axial velocity profile u(r)."""
    r = np.asarray(r, dtype=float)
    C = (Ro**2 - Ri**2) / np.log(Ro / Ri)
    return G / (4.0 * mu) * (Ro**2 - r**2 - C * np.log(Ro / r))


def analytical_shear(r, Ri: float, Ro: float, mu: float, G: float):
    """Exact shear stress tau_rz=mu du/dr."""
    r = np.asarray(r, dtype=float)
    C = (Ro**2 - Ri**2) / np.log(Ro / Ri)
    dudr = G / (4.0 * mu) * (-2.0 * r + C / r)
    return mu * dudr


def finite_difference_velocity(Ri: float, Ro: float, mu: float, G: float, n: int = 161):
    """Second-order radial FD solution of u''+(1/r)u'=-G/mu."""
    if n < 3:
        raise ValueError("n must be >= 3")
    r = np.linspace(Ri, Ro, n)
    dr = r[1] - r[0]
    A = np.zeros((n - 2, n - 2))
    b = np.full(n - 2, -G / mu)

    for j, i in enumerate(range(1, n - 1)):
        aw = 1.0 / dr**2 - 1.0 / (2.0 * r[i] * dr)
        ap = -2.0 / dr**2
        ae = 1.0 / dr**2 + 1.0 / (2.0 * r[i] * dr)
        if j > 0:
            A[j, j - 1] = aw
        A[j, j] = ap
        if j < n - 3:
            A[j, j + 1] = ae

    u = np.zeros(n)
    u[1:-1] = np.linalg.solve(A, b)
    return r, u


def baseline_metrics(Ri=0.001, Ro=0.002, length=0.1, rho=1060.0,
                     mu=0.0035, Q=100e-6 / 60.0):
    area = np.pi * (Ro**2 - Ri**2)
    mean_velocity = Q / area
    hydraulic_diameter = 2.0 * (Ro - Ri)
    reynolds = rho * mean_velocity * hydraulic_diameter / mu
    G = pressure_gradient_from_flow(Q, Ri, Ro, mu)
    r = np.linspace(Ri, Ro, 10001)
    u = analytical_velocity(r, Ri, Ro, mu, G)
    tau = analytical_shear(r, Ri, Ro, mu, G)
    imax = int(np.argmax(u))
    return {
        "mean_velocity_m_s": mean_velocity,
        "reynolds_number": reynolds,
        "pressure_gradient_Pa_m": G,
        "pressure_drop_Pa": G * length,
        "pressure_drop_mmHg": G * length / 133.322,
        "max_velocity_m_s": float(u[imax]),
        "max_velocity_radius_mm": float(r[imax] * 1000.0),
        "catheter_wall_shear_Pa": float(abs(tau[0])),
        "vessel_wall_shear_Pa": float(abs(tau[-1])),
    }


if __name__ == "__main__":
    for key, value in baseline_metrics().items():
        print(f"{key}: {value:.8g}")
