"""
heat_transfer.py
----------------
Single-integral Horizontal Finite Line Source (HFLS) for buried parallel pipes.

Physical setup
--------------
Two parallel horizontal pipes of equal length L are buried at the same depth D,
with a lateral (centre-to-centre) separation dy. Both pipes start at x = 0.
An isothermal ground surface (T = 0 at z = 0) is enforced by method of images:
a mirror source at depth -D with opposite sign heat flux.

The average temperature rise along the receiver pipe due to the source pipe is:

    ΔT = q' / (2π k_s) · h

Derivation of the single-integral form
---------------------------------------
The double integral over source (xs) and receiver (xr), both on [0, L], is:

    (1/L) · ∫₀ᴸ ∫₀ᴸ [erfc(r₁/√(4αt))/r₁ − erfc(r₂/√(4αt))/r₂] dxs dxr

Substituting u = xr − xs (range −L to L, triangular weight L − |u|):

    = (2/L) · ∫₀ᴸ (1 − u/L) · [erfc(r₁(u))/r₁(u) − erfc(r₂(u))/r₂(u)] du

Normalising to h = 2πk_s/q' · ΔT:

    h = ∫₀ᴸ (1 − u/L) · [erfc(r₁(u))/r₁(u) − erfc(r₂(u))/r₂(u)] du

where:
    r₁(u) = √(u² + d_perp²)              real source distance
    r₂(u) = √(u² + d_perp² + (2D)²)     mirror source distance

    For cross-pipe (dy > 0):   d_perp = dy
    For self        (dy = 0):  d_perp = r_pipe   (pipe outer radius)

This reduction is exact when source and receiver share the same start position
and length, which holds for all single-segment HHE traces.

References
----------
- Carslaw & Jaeger (1959): Conduction of Heat in Solids, 2nd ed., §10.2.
- Lamarche & Beauchamp (2007): Energy and Buildings, 39, 188–198.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc


def hfls_pipe_interaction(
    t: float,
    L: float,
    dy: float,
    depth: float,
    r_pipe: float,
    alpha: float,
) -> float:
    """
    Contribution of one source pipe (plus its mirror image) to the
    receiver pipe's average temperature, expressed as a dimensionless h:

        ΔT_avg = q' / (2π k_s) · h

    The source pipe runs from x = 0 to x = L at lateral position y = dy
    relative to the receiver pipe (dy = 0 for self-response).  The method
    of images mirror source at depth −D is included via the r₂ term.

    Derivation of the single-integral form
    ----------------------------------------
    The exact receiver average requires a double integral over source (xs)
    and receiver (xr) positions, both on [0, L].  Substituting u = xr − xs
    reduces this to a single integral with a triangular weight (1 − u/L),
    which accounts for the receiver averaging implicitly:

        h = ∫₀ᴸ (1 − u/L) · [erfc(r₁/√(4αt))/r₁ − erfc(r₂/√(4αt))/r₂] du

        r₁(u) = √(u² + d_perp²)            real source
        r₂(u) = √(u² + d_perp² + (2D)²)   mirror image (isothermal BC)

        d_perp = dy       for a neighbour pipe  (dy > 0)
        d_perp = r_pipe   for the self-response (dy = 0)

    Parameters
    ----------
    t : float
        Time (s).
    L : float
        Pipe length (m).
    dy : float
        Lateral centre-to-centre separation from receiver (m).
        Pass 0.0 for the self-response.
    depth : float
        Burial depth (m), positive downward.
    r_pipe : float
        Pipe outer radius (m), used as d_perp for the self-response.
    alpha : float
        Soil thermal diffusivity (m²/s).

    Returns
    -------
    float
        h such that ΔT_avg = q' / (2π k_s) · h.
    """
    sqrt_4at = np.sqrt(4.0 * alpha * t)
    is_self = (dy == 0.0)
    d_perp_sq = r_pipe**2 if is_self else dy**2
    d_image_sq = d_perp_sq + (2.0 * depth) ** 2

    def integrand(u: float) -> float:
        r1 = np.sqrt(u * u + d_perp_sq)
        r2 = np.sqrt(u * u + d_image_sq)
        return (1.0 - u / L) * (erfc(r1 / sqrt_4at) / r1 - erfc(r2 / sqrt_4at) / r2)

    result, _ = quad(
        integrand,
        0.0,
        L,
        limit=100,
        epsabs=1e-10,
        epsrel=1e-8,
        points=[0.0] if is_self else [],
    )
    return result


def thermal_propagation_distance(t: float, alpha: float, n_sigma: float = 3.0) -> float:
    """
    Distance beyond which thermal influence is negligible at time t.

    Uses erfc(r / sqrt(4*alpha*t)) < erfc(n_sigma) as the criterion.
    At n_sigma = 3: erfc(3) ~ 2e-5 (< 0.002 % of peak response).

    Parameters
    ----------
    t : float
        Time (s).
    alpha : float
        Soil thermal diffusivity (m²/s).
    n_sigma : float
        Multiplier. Default 3.

    Returns
    -------
    float
        Cutoff distance (m).
    """
    return n_sigma * np.sqrt(4.0 * alpha * t)
