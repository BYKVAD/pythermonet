from __future__ import annotations

import math
import numpy as np


def reynolds_number(rho: float, mu: float, v, d):
    """Re = rho*v*d/mu. v og d kan være scalar eller numpy arrays."""
    return (rho * np.asarray(v) * np.asarray(d)) / mu


def darcy_friction_factor_beier(Re):
    """
    Beier correlation for Darcy friction factor.
    Understøtter scalar og numpy arrays.
    """
    Re = np.asarray(Re, dtype=float)

    # Undgå log/div med 0 (hvis Re=0 fra fx Q=0)
    Re_safe = np.maximum(Re, 1e-12)

    term = 1.0 / np.sqrt((8.0 / Re_safe) ** 10 + (Re_safe / 36500.0) ** 20) + (2.21 * np.log(Re_safe / 7.0)) ** 10
    fD = 8.0 * term ** (-1.0 / 5.0)

    # Hvis Re meget lav, kan udtrykket give nan -> sæt til 0 (ingen flow => intet dp)
    fD = np.where(np.isfinite(fD), fD, 0.0)
    return fD


def pressure_loss_per_length(rho: float, mu: float, Q: float, Di):
    """
    Pressure loss per length (Pa/m).
    Q: m3/s (scalar)
    Di: m (scalar eller array af diametre)
    """
    Di = np.asarray(Di, dtype=float)

    # Hvis Q=0 -> ingen tryktab
    if not np.isfinite(Q) or Q == 0.0:
        return np.zeros_like(Di)

    v = Q / (math.pi * (Di / 2.0) ** 2)     # m/s (array)
    K = rho / 2.0 * v**2                    # Pa (array)
    Re = reynolds_number(rho, mu, v, Di)    # (array)
    fD = darcy_friction_factor_beier(Re)    # (array)
    return fD * K / Di
