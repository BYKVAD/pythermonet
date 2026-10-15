"""
heat_transfer.py
----------------
Single-integral Horizontal Finite Line Source (HFLS) for buried parallel pipes.

Physical setup
--------------
Two parallel horizontal pipes of equal length L are buried at the same depth D,
with a lateral (center-to-center) separation dy. Both pipes start at x = 0.
An isothermal ground surface (T = 0 at z = 0) is enforced by method of images:
each real pipe at depth D has a mirror source at depth -D (i.e. 2D above the
real pipe).

The temperature rise at the receiver pipe due to the source pipe is:

    ΔT = q' / (2π k_s) · h

Single-integral form
--------------------
The temperature at the near end (x = 0) of the receiver pipe due to the full
source pipe (xs in [0, L]) is evaluated as:

    h = ∫₀ᴸ [erfc(r₁(u)/√(4αt))/r₁(u) − erfc(r₂(u)/√(4αt))/r₂(u)] du

where u = xs (distance along the source from its start):

    r₁(u) = √(u² + d_perp²)          real source distance
    r₂(u) = √(u² + d_image²)         mirror source distance

    For cross-pipe (dy > 0):   d_perp = dy,   d_image = √(dy² + (2D)²)
    For self        (dy = 0):  d_perp = r_pipe (avoids singularity),
                               d_image = 2D   (image is never singular)

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
    Contribution of one source pipe plus its mirror image to the receiver pipe,
    expressed as a dimensionless h:

        ΔT = q' / (2π k_s) · h

    The source pipe runs from x = 0 to x = L at lateral position y = dy
    relative to the receiver.  The isothermal surface BC is enforced by
    a mirror source at depth −D (i.e. distance 2D above the real pipe).

    Integral form:
        h = ∫₀ᴸ [erfc(r₁(u)/√(4αt))/r₁(u) − erfc(r₂(u)/√(4αt))/r₂(u)] du

        r₁(u) = √(u² + d_perp²)           real source distance
        r₂(u) = √(u² + d_image²)          mirror source distance

        d_perp  = r_pipe  for self (dy = 0);  dy  for cross-pipe
        d_image = 2D      for self;           √(dy² + (2D)²)  for cross-pipe

    The image source is never singular (2D >> 0), so r_pipe is NOT added
    to the image distance.

    Parameters
    ----------
    t : float
        Time (s).
    L : float
        Pipe length (m).
    dy : float
        Lateral center-to-center separation from receiver (m).
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
    # Real source: use r_pipe as minimum distance for self (avoids singularity)
    d_perp_sq = r_pipe**2 if is_self else dy**2
    # Image source: mirror is at depth -D, so perpendicular distance to receiver
    # at depth +D is sqrt(dy^2 + (2D)^2). For self, dy=0 → distance is just 2D.
    d_image_sq = (0.0 if is_self else dy**2) + (2.0 * depth) ** 2

    def integrand(u: float) -> float:
        r1 = np.sqrt(u * u + d_perp_sq)
        r2 = np.sqrt(u * u + d_image_sq)
        return erfc(r1 / sqrt_4at) / r1 - erfc(r2 / sqrt_4at) / r2

    # The integrand peaks near u = 0 and decays on the scale of sqrt_4at.
    # Add interior breakpoints so that quad resolves the peak even when it is
    # narrow relative to the total pipe length (early times).
    inner_points = sorted({
        p for p in (sqrt_4at, 3.0 * sqrt_4at) if 0.0 < p < L
    })

    result, _ = quad(
        integrand,
        0.0,
        L,
        limit=200,
        epsabs=1e-10,
        epsrel=1e-8,
        points=inner_points,
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
