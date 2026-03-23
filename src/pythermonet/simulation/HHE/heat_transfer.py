"""
heat_transfer.py
----------------
Horizontal finite line source (HFLS) solution for buried pipes in a
semi-infinite medium with an isothermal ground surface boundary condition.

Physical setup
--------------
A pipe of length L is buried horizontally at depth D below an isothermal
ground surface (T = T_ground at z = 0). The pipe is aligned along the
x-axis. The method of images places a mirror source at z = -D with
opposite sign heat flux, enforcing the isothermal boundary condition.

The average temperature rise along a receiver pipe segment due to a source
pipe segment (both horizontal, parallel, at the same depth) is:

    DeltaT = q' / (4*pi*k_s) * I

where the double integral I is:

    I = (1/L_recv) * integral_{recv} integral_{src} [
            erfc(r1(xr,xs) / sqrt(4*alpha*t)) / r1(xr,xs)
          - erfc(r2(xr,xs) / sqrt(4*alpha*t)) / r2(xr,xs)
        ] dxs dxr

    r1 = sqrt((xr-xs)^2 + dy^2 + (D_recv-D_src)^2)   real source distance
    r2 = sqrt((xr-xs)^2 + dy^2 + (D_recv+D_src)^2)   image source distance
    dy = y_recv - y_src                                lateral separation

The g-function follows pygfunction's normalisation:
    g = 2*pi*k_s * DeltaT / q' = I / 2

Quadrature strategy
-------------------
The inner integral (over source) is smooth for cross-pipe interactions
(dy > 0) and can be evaluated with Gauss-Legendre quadrature.

For the self-response (same pipe, dy = 0), the integrand has a sharp peak
of width ~r_pipe and height ~1/r_pipe at xr = xs. Standard Gauss quadrature
on the full pipe length fails to resolve this peak. scipy.integrate.quad
with adaptive refinement is used instead for the inner integral in this case.

References
----------
- Carslaw & Jaeger (1959): Conduction of Heat in Solids, 2nd ed.
  Point source solution in an infinite medium, §10.2.
- Lamarche & Beauchamp (2007): A new contribution to the finite line-source
  model for geothermal boreholes. Energy and Buildings, 39, 188-198.
  Single-integral FLS form (adapted here for horizontal orientation).
- Cimmino & Bernier (2014): A semi-analytical method to generate g-functions
  for geothermal bore fields. Int. J. Heat Mass Transfer, 70, 641-650.
  Normalisation convention and double-integral average temperature.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc


# ---------------------------------------------------------------------------
# Low-level point source kernel
# ---------------------------------------------------------------------------

def _ps(r: np.ndarray, t: float, alpha: float) -> np.ndarray:
    """
    Point source kernel: erfc(r / sqrt(4*alpha*t)) / r.
    Returns 0 where r == 0 (no self-heating at zero distance).
    """
    out = np.zeros_like(r, dtype=float)
    mask = r > 0.0
    out[mask] = erfc(r[mask] / np.sqrt(4.0 * alpha * t)) / r[mask]
    return out


# ---------------------------------------------------------------------------
# Inner integral (source integration): smooth cross-pipe case
# ---------------------------------------------------------------------------

def _inner_gauss(
    x_recv: float,
    x_src_start: float,
    L_src: float,
    dy: float,
    dz_real: float,
    dz_image: float,
    t: float,
    alpha: float,
    xi_src: np.ndarray,   # pre-computed Gauss points on [-1,1]
    w_src: np.ndarray,    # pre-computed Gauss weights
) -> float:
    """
    Inner integral over source pipe using Gauss-Legendre quadrature.
    Used for cross-pipe interactions where the integrand is smooth.
    """
    half = L_src / 2.0
    mid = x_src_start + half
    xs = mid + half * xi_src

    dx = x_recv - xs
    r1 = np.sqrt(dx**2 + dy**2 + dz_real**2)
    r2 = np.sqrt(dx**2 + dy**2 + dz_image**2)

    integrand = _ps(r1, t, alpha) - _ps(r2, t, alpha)
    return half * float(np.dot(w_src, integrand))


def _inner_quad(
    x_recv: float,
    x_src_start: float,
    L_src: float,
    r_pipe: float,
    depth: float,
    t: float,
    alpha: float,
    n_sigma: float = 6.0,
) -> float:
    """
    Inner integral over source pipe using adaptive quadrature (scipy.quad).
    Used for self-response where dy = 0 and the integrand has a sharp peak
    at xs = x_recv of half-width ~ sqrt(4*alpha*t).

    At short times sqrt(4*alpha*t) can be much smaller than the pipe length,
    so integrating over the full pipe length causes scipy.quad to miss the
    spike entirely. We therefore restrict the integration to a local window
    of half-width max(n_sigma * sqrt(4*alpha*t), 10*r_pipe) around x_recv,
    clipped to the pipe extent. The contribution outside this window is
    negligible by construction (erfc argument >> 1).
    """
    sqrt_4at = np.sqrt(4.0 * alpha * t)
    dz_image = 2.0 * depth

    # Local window: wide enough to capture the full spike
    half_window = max(n_sigma * sqrt_4at, 10.0 * r_pipe)
    lo = max(x_src_start, x_recv - half_window)
    hi = min(x_src_start + L_src, x_recv + half_window)

    if lo >= hi:
        return 0.0

    def integrand(xs: float) -> float:
        dx = x_recv - xs
        r1 = np.sqrt(dx**2 + r_pipe**2)
        r2 = np.sqrt(dx**2 + r_pipe**2 + dz_image**2)
        return erfc(r1 / sqrt_4at) / r1 - erfc(r2 / sqrt_4at) / r2

    val, _ = quad(
        integrand,
        lo,
        hi,
        limit=200,
        epsabs=1e-10,
        epsrel=1e-8,
        points=[x_recv] if lo < x_recv < hi else [],
    )
    return val


# ---------------------------------------------------------------------------
# Segment-to-segment interaction (the core building block)
# ---------------------------------------------------------------------------

def hfls_segment_interaction(
    t: float,
    x_src: float,
    L_src: float,
    y_src: float,
    x_recv: float,
    L_recv: float,
    y_recv: float,
    depth: float,
    r_pipe: float,
    alpha: float,
    n_gauss: int = 21,
    _gauss_cache: dict = {},
) -> float:
    """
    Dimensionless interaction factor h between a source segment and a
    receiver segment. Both segments are horizontal, parallel, at depth D.

    The average temperature rise along the receiver is:
        DeltaT = q' / (2*pi*k_s) * h

    Parameters
    ----------
    x_src, L_src, y_src : float
        Start x-position, length, and lateral y-position of the source (m).
    x_recv, L_recv, y_recv : float
        Start x-position, length, and lateral y-position of the receiver (m).
    depth : float
        Burial depth of both segments (m), positive downward.
    r_pipe : float
        Pipe outer radius (m). Used as the evaluation radius for self-response.
    alpha : float
        Soil thermal diffusivity (m2/s).
    n_gauss : int
        Gauss-Legendre points for the outer (receiver) integral and for
        inner integrals in cross-pipe interactions.

    Returns
    -------
    float
        Dimensionless h such that DeltaT = q'/(2*pi*k_s) * h.
    """
    # Cache Gauss points (keyed by n_gauss)
    if n_gauss not in _gauss_cache:
        _gauss_cache[n_gauss] = np.polynomial.legendre.leggauss(n_gauss)
    xi_g, w_g = _gauss_cache[n_gauss]

    dy = y_recv - y_src
    dz_real = 0.0          # same depth: D_recv - D_src = 0
    dz_image = 2.0 * depth  # D_recv + D_src = 2D

    is_self = (dy == 0.0 and x_src == x_recv and L_src == L_recv)

    # Outer integral: Gauss points along receiver
    half_r = L_recv / 2.0
    mid_r = x_recv + half_r
    xr_pts = mid_r + half_r * xi_g

    total = 0.0
    for i, xr in enumerate(xr_pts):
        if is_self:
            inner = _inner_quad(xr, x_src, L_src, r_pipe, depth, t, alpha)
        else:
            inner = _inner_gauss(
                xr, x_src, L_src, dy, dz_real, dz_image,
                t, alpha, xi_g, w_g,
            )
        total += w_g[i] * inner

    # Outer Gauss integral over receiver length, then normalise to h:
    # DeltaT_avg = q'/(4*pi*k_s) * (1/L_recv) * half_r * total
    # h = 2*pi*k_s/q' * DeltaT_avg = half_r * total / (2 * L_recv)
    return half_r * total / (2.0 * L_recv)


# ---------------------------------------------------------------------------
# Thermal propagation cutoff
# ---------------------------------------------------------------------------

def thermal_propagation_distance(t: float, alpha: float, n_sigma: float = 3.0) -> float:
    """
    Distance beyond which thermal influence is negligible at time t.

    Uses erfc(r / sqrt(4*alpha*t)) < erfc(n_sigma) as the criterion.
    At n_sigma=3: erfc(3) ~ 2e-5 (< 0.002% of peak response).

    Parameters
    ----------
    t : float
        Time (s).
    alpha : float
        Soil thermal diffusivity (m2/s).
    n_sigma : float
        Multiplier. Default 3.

    Returns
    -------
    float
        Cutoff distance (m).
    """
    return n_sigma * np.sqrt(4.0 * alpha * t)
