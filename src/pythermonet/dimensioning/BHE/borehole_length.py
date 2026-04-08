from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from pythermonet.components.vhe_field import VHEField
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.physics.bhe_resistance import compute_rb_for_vhe_field
from pythermonet.physics.sources import ils


def apply_annual_balance(
    P_heating_W: np.ndarray,
    P_cooling_W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply thermal ground balancing to the annual pulse.

    Cooling rejection during the year regenerates the ground for heating,
    and vice versa.  Only the annual pulse (index 0) is corrected — the
    winter and peak pulses represent single-mode dominant conditions.

    Parameters
    ----------
    P_heating_W : array, shape (3,)  [P_annual, P_winter, P_peak]
    P_cooling_W : array, shape (3,)  [P_annual, P_winter, P_peak]

    Returns
    -------
    P_heating_net, P_cooling_net : arrays with balanced annual pulse.
    P_heating_net[0] can be negative if cooling dominates annually.
    """
    P_h = np.array(P_heating_W, dtype=float)
    P_c = np.array(P_cooling_W, dtype=float)
    P_h[0] -= P_c[0]
    P_c[0] -= P_heating_W[0]   # use original heating annual, not already-modified P_h[0]
    return P_h, P_c




# ---------------------------------------------------------------------------
# Rb helpers
# ---------------------------------------------------------------------------

def _rb_simple(
    vhe_field: VHEField,
    brine: HeatCarrier,
    soil: Soil,
    m_dot_per_borehole_kg_s: float,
) -> float:
    """Rb without flow/length correction — used for the ILS initial guess."""
    return compute_rb_for_vhe_field(
        vhe_field=vhe_field,
        brine=brine,
        soil=soil,
        L_bhe_m=1.0,          # not used without correction - so its just a dummy value
        m_dot_kg_s=m_dot_per_borehole_kg_s,
        use_flow_length_correction=False,
    ).Rb_K_m_W


def _rb_corrected(
    vhe_field: VHEField,
    brine: HeatCarrier,
    soil: Soil,
    H: float,
    m_dot_per_borehole_kg_s: float,
) -> float:
    """Rb with flow/length correction at the current H — used during bisection."""
    return compute_rb_for_vhe_field(
        vhe_field=vhe_field,
        brine=brine,
        soil=soil,
        L_bhe_m=H,
        m_dot_kg_s=m_dot_per_borehole_kg_s,
        use_flow_length_correction=True,
    ).Rb_K_m_W


# ---------------------------------------------------------------------------
# ILS field g-function (used for initial guess — no pygfunction call)
# ---------------------------------------------------------------------------

def _g_ils_field(times_s: np.ndarray, alpha: float, vhe_field: VHEField) -> np.ndarray:
    """
    ILS g-function averaged over the borehole field.

    Includes self-response at r_b and cross-borehole superposition at the
    borehole separations. Does not depend on H, so it is evaluated once and
    used analytically to estimate the required borehole length.

    Parameters
    ----------
    times_s : array, shape (3,)
        [t_peak, t_winter, t_annual] [s].

    Returns
    -------
    g : array, shape (3,)
        g-function values at each time (same convention as pygfunction).
    """
    r_b = float(vhe_field.r_b_m)
    xy = np.array(vhe_field.xy_m, dtype=float)  # (N, 2)
    N = vhe_field.n_boreholes
    TWO_PI = 2.0 * math.pi

    g = np.zeros(len(times_s))
    for k, t in enumerate(times_s):
        g_self = TWO_PI * ils(alpha, float(t), r_b)

        g_cross = 0.0
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
                    g_cross += TWO_PI * ils(alpha, float(t), d)
        g_cross /= N

        g[k] = g_self + g_cross

    return g


# ---------------------------------------------------------------------------
# Analytical initial guess from the quadratic sizing equation
# ---------------------------------------------------------------------------

def _K_factor(
    P_bhe_W: np.ndarray,
    g: np.ndarray,
    vhe_field: VHEField,
    soil: Soil,
    Rb: float,
) -> float:
    """
    Combined thermal load coefficient K [K·m].

    The sizing equation is:  T_undisturbed(H) ± K/H = T_fluid_limit
    (− for heating, + for cooling).

        K = [P_a·g[2] + (P_w−P_a)·g[1] + (P_p−P_w)·g[0]] / (2π k_s N)
          + P_p·Rb / N

    g : [g(t_peak), g(t_winter), g(t_annual)]
    P_bhe_W : [P_annual, P_winter, P_peak]
    """
    N = vhe_field.n_boreholes
    k_s = float(soil.thermalCond)
    P_a, P_w, P_p = float(P_bhe_W[0]), float(P_bhe_W[1]), float(P_bhe_W[2])

    ground_term = (
        P_a * g[2] + (P_w - P_a) * g[1] + (P_p - P_w) * g[0]
    ) / (2.0 * math.pi * k_s * N)

    return ground_term + P_p * float(Rb) / N


def _initial_guess_heating(
    P_bhe_W: np.ndarray,
    g_ils: np.ndarray,
    vhe_field: VHEField,
    soil: Soil,
    Rb_simple: float,
    T_fluid_min: float,
) -> float:
    """
    Solve  T_surface + a·H − K/H = T_fluid_min  for H  (quadratic).

    Uses Rb without flow/length correction (H is not yet known).
    Returns the positive root via a numerically stable formula.
    """
    K = _K_factor(P_bhe_W, g_ils, vhe_field, soil, Rb_simple)
    a = float(soil.Qgeo) / (2.0 * float(soil.thermalCond))
    dT0 = float(soil.surfaceTemp) - T_fluid_min

    disc = dT0 ** 2 + 4.0 * a * K
    return 2.0 * K / (dT0 + math.sqrt(disc))


def _initial_guess_cooling(
    P_bhe_W: np.ndarray,
    g_ils: np.ndarray,
    vhe_field: VHEField,
    soil: Soil,
    Rb_simple: float,
    T_fluid_max: float,
) -> float | None:
    """
    Solve  T_surface + a·H + K/H = T_fluid_max  for the smaller positive root.

    Uses Rb without flow/length correction (H is not yet known).
    Returns None if infeasible with ILS (caller falls back to H_max).
    """
    K = _K_factor(P_bhe_W, g_ils, vhe_field, soil, Rb_simple)
    a = float(soil.Qgeo) / (2.0 * float(soil.thermalCond))
    margin = T_fluid_max - float(soil.surfaceTemp)

    disc = margin ** 2 - 4.0 * a * K
    if disc < 0.0:
        return None

    return 2.0 * K / (margin + math.sqrt(disc))


# ---------------------------------------------------------------------------
# Temperature evaluation (pygfunction + corrected Rb — called during bisection)
# ---------------------------------------------------------------------------

def _T_ground_mean(H: float, soil: Soil) -> float:
    """Mean undisturbed ground temperature along borehole [°C]."""
    return float(soil.surfaceTemp) + float(soil.Qgeo) * H / (2.0 * float(soil.thermalCond))


def _delta_T_ground(
    H: float,
    P_bhe_W: np.ndarray,
    g: np.ndarray,
    vhe_field: VHEField,
    soil: Soil,
) -> float:
    """Three-pulse temperature penalty [K]. g: [g(t_peak), g(t_winter), g(t_annual)]."""
    N = vhe_field.n_boreholes
    q_a = float(P_bhe_W[0]) / (N * H)
    q_w = float(P_bhe_W[1]) / (N * H)
    q_p = float(P_bhe_W[2]) / (N * H)
    return (
        q_a * g[2] + (q_w - q_a) * g[1] + (q_p - q_w) * g[0]
    ) / (2.0 * math.pi * float(soil.thermalCond))


def _T_fluid_heat_at_H(
    H: float,
    P_bhe_W: np.ndarray,
    times_s: np.ndarray,
    vhe_field: VHEField,
    brine: HeatCarrier,
    soil: Soil,
    m_dot_per_borehole_kg_s: float,
    alpha_m2_s: float,
) -> float:
    """
    Mean fluid temperature [°C] at peak for heating (extraction) mode.

    Rb is recomputed with flow/length correction at the current H.
    """
    field_H = replace(vhe_field, H_m=H)
    g = field_H.compute_pygfunctions(times_s=times_s, alpha_m2_s=alpha_m2_s)
    Rb = _rb_corrected(vhe_field, brine, soil, H, m_dot_per_borehole_kg_s)
    q_p = float(P_bhe_W[2]) / (vhe_field.n_boreholes * H)
    return _T_ground_mean(H, soil) - _delta_T_ground(H, P_bhe_W, g, vhe_field, soil) - q_p * Rb


def _T_fluid_cool_at_H(
    H: float,
    P_bhe_W: np.ndarray,
    times_s: np.ndarray,
    vhe_field: VHEField,
    brine: HeatCarrier,
    soil: Soil,
    m_dot_per_borehole_kg_s: float,
    alpha_m2_s: float,
) -> float:
    """
    Mean fluid temperature [°C] at peak for cooling (rejection) mode.

    Rb is recomputed with flow/length correction at the current H.
    """
    field_H = replace(vhe_field, H_m=H)
    g = field_H.compute_pygfunctions(times_s=times_s, alpha_m2_s=alpha_m2_s)
    Rb = _rb_corrected(vhe_field, brine, soil, H, m_dot_per_borehole_kg_s)
    q_p = float(P_bhe_W[2]) / (vhe_field.n_boreholes * H)
    return _T_ground_mean(H, soil) + _delta_T_ground(H, P_bhe_W, g, vhe_field, soil) + q_p * Rb


# ---------------------------------------------------------------------------
# Public sizing functions
# ---------------------------------------------------------------------------

def size_borehole_length(
    *,
    T_fluid_min: float,
    P_bhe_W: np.ndarray,
    times_s: np.ndarray,
    vhe_field: VHEField,
    brine: HeatCarrier,
    soil: Soil,
    m_dot_per_borehole_kg_s: float,
    H_min: float = 50.0,
    H_max: float = 500.0,
    tol_m: float = 0.01,
) -> float:
    """
    Find the minimum borehole length satisfying the heating constraint
    T_fluid ≥ T_fluid_min.

    Rb handling
    -----------
    - Initial guess (ILS quadratic): Rb without flow/length correction.
    - Bisection:  Rb recomputed with flow/length correction at each H.

    Parameters
    ----------
    T_fluid_min : float
        Minimum allowable **mean** BHE fluid temperature [°C].
        If your limit is the HP evaporator inlet (BHE outlet), pass
        ``T_hp_inlet - 0.5 * deltaT_sys`` here.
    P_bhe_W : array, shape (3,)
        Total BHE ground loads [W]: [P_annual, P_winter, P_peak].
    times_s : array, shape (3,)
        Pulse durations [s]: [t_peak, t_winter, t_annual] (ascending).
    vhe_field : VHEField
        Borehole field geometry.
    brine : HeatCarrier
        Brine fluid properties.
    soil : Soil
        Ground properties.
    m_dot_per_borehole_kg_s : float
        Mass flow rate per borehole [kg/s].
    H_min, H_max : float
        Search bracket [m].
    tol_m : float
        Bisection convergence tolerance [m].  Default 0.01 m gives a
        temperature convergence of ~0.001 K, comparable to the 1e-4 K
        tolerance used in earlier Halley-method implementations.
    """
    alpha = float(soil.thermalCond) / (float(soil.rho) * float(soil.c))
    times_s = np.asarray(times_s, dtype=float)
    P_bhe_W = np.asarray(P_bhe_W, dtype=float)

    def f(H: float) -> float:
        return (
            _T_fluid_heat_at_H(H, P_bhe_W, times_s, vhe_field, brine, soil, m_dot_per_borehole_kg_s, alpha)
            - T_fluid_min
        )

    if f(H_min) >= 0.0:
        return H_min

    # ILS initial guess as upper bracket
    g_ils = _g_ils_field(times_s, alpha, vhe_field)
    Rb_simple = _rb_simple(vhe_field, brine, soil, m_dot_per_borehole_kg_s)
    H_guess = _initial_guess_heating(P_bhe_W, g_ils, vhe_field, soil, Rb_simple, T_fluid_min)
    H_guess = float(np.clip(H_guess, H_min, H_max))

    hi = H_guess if f(H_guess) >= 0.0 else H_max

    if f(hi) < 0.0:
        raise ValueError(
            f"Heating constraint not met at H={hi:.1f} m: "
            f"T_fluid={f(hi) + T_fluid_min:.2f} °C < T_min={T_fluid_min} °C. "
            "Increase H_max or check inputs."
        )

    lo = H_min
    while (hi - lo) > tol_m:
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def size_borehole_length_heating_cooling(
    *,
    T_fluid_min: float,
    P_heating_W: np.ndarray,
    times_heat_s: np.ndarray,
    T_fluid_max: float = float("inf"),
    P_cooling_W: np.ndarray | None = None,
    times_cool_s: np.ndarray | None = None,
    vhe_field: VHEField,
    brine: HeatCarrier,
    soil: Soil,
    m_dot_per_borehole_heat_kg_s: float,
    m_dot_per_borehole_cool_kg_s: float | None = None,
    H_min: float = 50.0,
    H_max: float = 500.0,
    tol_m: float = 0.01,
    pre_balanced: bool = False,
) -> FieldSizingResult:
    """
    Find the minimum borehole length satisfying both heating and cooling
    temperature constraints.

    Rb handling
    -----------
    - Initial guesses: Rb without flow/length correction (H unknown).
    - All bisection evaluations: Rb recomputed with flow/length correction
      at the current H, using the mode-specific flow rate.

    Strategy
    --------
    1. Size for heating (ILS-guided bisection) → H_heat.
    2. Check cooling at H_heat. If satisfied, heating governs — done.
       This avoids a spurious cooling optimum from the geothermal gradient.
    3. If cooling is violated, use ILS cooling guess as upper bracket and
       bisect between H_heat and that bracket.

    Returns
    -------
    FieldSizingResult
        L_m (borehole depth), governing mode, and Rb for heating and
        cooling at the final depth.
    """
    has_cooling = P_cooling_W is not None

    alpha = float(soil.thermalCond) / (float(soil.rho) * float(soil.c))
    times_heat_s = np.asarray(times_heat_s, dtype=float)
    P_heating_W = np.asarray(P_heating_W, dtype=float)

    if has_cooling:
        times_cool_s = np.asarray(times_cool_s, dtype=float)
        if not pre_balanced:
            P_heating_W, P_cooling_W = apply_annual_balance(P_heating_W, np.asarray(P_cooling_W, dtype=float))

    def _rb_final(H: float, m_dot: float) -> float:
        return compute_rb_for_vhe_field(
            vhe_field=vhe_field, brine=brine, soil=soil,
            L_bhe_m=H, m_dot_kg_s=m_dot,
            use_flow_length_correction=True,
        ).Rb_K_m_W

    # --- Step 1: size for heating ---
    H_heat = size_borehole_length(
        T_fluid_min=T_fluid_min,
        P_bhe_W=P_heating_W,
        times_s=times_heat_s,
        vhe_field=vhe_field,
        brine=brine,
        soil=soil,
        m_dot_per_borehole_kg_s=m_dot_per_borehole_heat_kg_s,
        H_min=H_min,
        H_max=H_max,
        tol_m=tol_m,
    )

    # --- Heating-only: return immediately ---
    if not has_cooling:
        return FieldSizingResult(
            L_m=H_heat,
            governing="heating",
            R_heating_K_m_W=_rb_final(H_heat, m_dot_per_borehole_heat_kg_s),
            R_cooling_K_m_W=None,
        )

    # --- Step 2: quick check — is cooling already satisfied at H_heat? ---
    T_cool_at_H_heat = _T_fluid_cool_at_H(
        H_heat, P_cooling_W, times_cool_s, vhe_field, brine, soil, m_dot_per_borehole_cool_kg_s, alpha
    )
    if T_cool_at_H_heat <= T_fluid_max:
        return FieldSizingResult(
            L_m=H_heat,
            governing="heating",
            R_heating_K_m_W=_rb_final(H_heat, m_dot_per_borehole_heat_kg_s),
            R_cooling_K_m_W=_rb_final(H_heat, m_dot_per_borehole_cool_kg_s),
        )

    # --- Step 3: cooling is binding — ILS bracket then bisect ---
    g_ils_cool = _g_ils_field(times_cool_s, alpha, vhe_field)
    Rb_simple_cool = _rb_simple(vhe_field, brine, soil, m_dot_per_borehole_cool_kg_s)
    H_guess_cool = _initial_guess_cooling(
        P_cooling_W, g_ils_cool, vhe_field, soil, Rb_simple_cool, T_fluid_max
    )

    if H_guess_cool is None:
        hi_cool = H_max
    else:
        hi_cool = float(np.clip(H_guess_cool, H_heat, H_max))

    def f_cool(H: float) -> float:
        return (
            _T_fluid_cool_at_H(H, P_cooling_W, times_cool_s, vhe_field, brine, soil, m_dot_per_borehole_cool_kg_s, alpha)
            - T_fluid_max
        )

    if f_cool(hi_cool) > 0.0:
        hi_cool = H_max

    if f_cool(hi_cool) > 0.0:
        raise ValueError(
            f"Cooling constraint not met at H_max={H_max} m: "
            f"T_fluid_cool={f_cool(hi_cool) + T_fluid_max:.2f} °C > T_max={T_fluid_max} °C. "
            "Increase H_max or check inputs."
        )

    lo, hi = H_heat, hi_cool
    while (hi - lo) > tol_m:
        mid = 0.5 * (lo + hi)
        if f_cool(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    H_cool = 0.5 * (lo + hi)
    return FieldSizingResult(
        L_m=H_cool,
        governing="cooling",
        R_heating_K_m_W=_rb_final(H_cool, m_dot_per_borehole_heat_kg_s),
        R_cooling_K_m_W=_rb_final(H_cool, m_dot_per_borehole_cool_kg_s),
    )


# ===========================================================================
# Generic field sizing — works with BHEGroundField and HHEGroundField
# ===========================================================================

from pythermonet.dimensioning.ground_field import GroundField  # noqa: E402


@dataclass(frozen=True)
class FieldSizingResult:
    """Result from the generic ground-field sizing solver."""
    L_m: float                   # Required element length [m]
    governing: str               # "heating" or "cooling"
    R_heating_K_m_W: float       # Effective thermal resistance at L_m, heating flow
    R_cooling_K_m_W: float | None


# ---------------------------------------------------------------------------
# Generic internal helpers (mirror the BHE-specific ones above)
# ---------------------------------------------------------------------------

def _K_factor_field(
    P_W: np.ndarray,
    g: np.ndarray,
    field: GroundField,
    soil: Soil,
    R: float,
) -> float:
    N = field.n_thermal_elements
    k_s = field.k_s_eff(soil)
    P_a, P_w, P_p = float(P_W[0]), float(P_W[1]), float(P_W[2])
    ground_term = (
        P_a * g[2] + (P_w - P_a) * g[1] + (P_p - P_w) * g[0]
    ) / (2.0 * math.pi * k_s * N)
    return ground_term + P_p * R / N


def _delta_T_ground_field(
    L: float,
    P_W: np.ndarray,
    g: np.ndarray,
    field: GroundField,
    soil: Soil,
) -> float:
    N = field.n_thermal_elements
    q_a = float(P_W[0]) / (N * L)
    q_w = float(P_W[1]) / (N * L)
    q_p = float(P_W[2]) / (N * L)
    return (
        q_a * g[2] + (q_w - q_a) * g[1] + (q_p - q_w) * g[0]
    ) / (2.0 * math.pi * field.k_s_eff(soil))


def _initial_guess_heating_field(
    P_W: np.ndarray,
    g_ils: np.ndarray,
    field: GroundField,
    soil: Soil,
    R_simple: float,
    T_fluid_min: float,
) -> float:
    """
    Solve  T_ref + a·L - K/L = T_fluid_min  for L.

    For BHE  a = Qgeo/(2·k_s) > 0  →  standard quadratic.
    For HHE  a = 0               →  degenerates to L = K/dT0.
    Both cases are handled by the same numerically stable formula.
    """
    K = _K_factor_field(P_W, g_ils, field, soil, R_simple)
    a = field.T_gradient(soil)
    T_ref = field.T_undisturbed(0.0, soil) - field.seasonal_amplitude(soil)
    dT0 = T_ref - T_fluid_min
    disc = dT0 ** 2 + 4.0 * a * K
    return 2.0 * K / (dT0 + math.sqrt(disc))


def _initial_guess_cooling_field(
    P_W: np.ndarray,
    g_ils: np.ndarray,
    field: GroundField,
    soil: Soil,
    R_simple: float,
    T_fluid_max: float,
) -> float | None:
    K = _K_factor_field(P_W, g_ils, field, soil, R_simple)
    a = field.T_gradient(soil)
    T_ref = field.T_undisturbed(0.0, soil) + field.seasonal_amplitude(soil)
    margin = T_fluid_max - T_ref
    disc = margin ** 2 - 4.0 * a * K
    if disc < 0.0:
        return None
    return 2.0 * K / (margin + math.sqrt(disc))


def _T_fluid_heat_at_L(
    L: float,
    P_W: np.ndarray,
    times_s: np.ndarray,
    field: GroundField,
    brine: HeatCarrier,
    soil: Soil,
    m_dot: float,
    alpha: float,
) -> float:
    g = field.compute_gfunction(L, times_s, alpha)
    R = field.R_at_L(L, m_dot, brine, soil)
    q_p = float(P_W[2]) / (field.n_thermal_elements * L)
    return (
        field.T_undisturbed(L, soil)
        - field.seasonal_amplitude(soil)
        - _delta_T_ground_field(L, P_W, g, field, soil)
        - q_p * R
    )


def _T_fluid_cool_at_L(
    L: float,
    P_W: np.ndarray,
    times_s: np.ndarray,
    field: GroundField,
    brine: HeatCarrier,
    soil: Soil,
    m_dot: float,
    alpha: float,
) -> float:
    g = field.compute_gfunction(L, times_s, alpha)
    R = field.R_at_L(L, m_dot, brine, soil)
    q_p = float(P_W[2]) / (field.n_thermal_elements * L)
    return (
        field.T_undisturbed(L, soil)
        + field.seasonal_amplitude(soil)
        + _delta_T_ground_field(L, P_W, g, field, soil)
        + q_p * R
    )


# ---------------------------------------------------------------------------
# Public generic sizing API
# ---------------------------------------------------------------------------

def size_ground_field_length(
    *,
    T_fluid_min: float,
    P_W: np.ndarray,
    times_s: np.ndarray,
    field: GroundField,
    brine: HeatCarrier,
    soil: Soil,
    m_dot_per_element: float,
    L_min: float = 20.0,
    L_max: float = 500.0,
    tol_m: float = 0.01,
) -> float:
    """
    Find the minimum element length satisfying the heating constraint
    T_fluid ≥ T_fluid_min.

    Works with any GroundField implementation (BHEGroundField or HHEGroundField).
    """
    alpha = field.k_s_eff_heating(soil) / (float(soil.rho) * float(soil.c))
    times_s = np.asarray(times_s, dtype=float)
    P_W = np.asarray(P_W, dtype=float)

    def f(L: float) -> float:
        return _T_fluid_heat_at_L(L, P_W, times_s, field, brine, soil, m_dot_per_element, alpha) - T_fluid_min

    if f(L_min) >= 0.0:
        return L_min

    g_ils = field.ils_gfunction(times_s, alpha)
    R_sim = field.R_simple(m_dot_per_element, brine, soil)
    L_guess = _initial_guess_heating_field(P_W, g_ils, field, soil, R_sim, T_fluid_min)
    L_guess = float(np.clip(L_guess, L_min, L_max))

    hi = L_guess if f(L_guess) >= 0.0 else L_max

    if f(hi) < 0.0:
        raise ValueError(
            f"Heating constraint not met at L={hi:.1f} m: "
            f"T_fluid={f(hi) + T_fluid_min:.2f} °C < T_min={T_fluid_min} °C. "
            "Increase L_max or check inputs."
        )

    lo = L_min
    while (hi - lo) > tol_m:
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def size_ground_field_length_heating_cooling(
    *,
    T_fluid_min: float,
    P_heating_W: np.ndarray,
    times_heat_s: np.ndarray,
    T_fluid_max: float = float("inf"),
    P_cooling_W: np.ndarray | None = None,
    times_cool_s: np.ndarray | None = None,
    field: GroundField,
    brine: HeatCarrier,
    soil: Soil,
    m_dot_per_element_heat: float,
    m_dot_per_element_cool: float | None = None,
    L_min: float = 20.0,
    L_max: float = 500.0,
    tol_m: float = 0.01,
    pre_balanced: bool = False,
) -> FieldSizingResult:
    """
    Find the minimum element length satisfying both heating and cooling
    temperature constraints.

    Works with any GroundField implementation (BHEGroundField or HHEGroundField).
    Strategy mirrors size_borehole_length_heating_cooling:
    1. Size for heating.
    2. If cooling is satisfied at H_heat, heating governs.
    3. Otherwise bisect for cooling between H_heat and the ILS upper bracket.
    """
    has_cooling = P_cooling_W is not None

    # Use mode-specific effective conductivity for alpha so that shallow HHE fields
    # use the correct shallow diffusivity rather than the deep-soil value.
    alpha_heat = field.k_s_eff_heating(soil) / (float(soil.rho) * float(soil.c))
    alpha_cool = field.k_s_eff_cooling(soil) / (float(soil.rho) * float(soil.c))

    times_heat_s = np.asarray(times_heat_s, dtype=float)
    P_heating_W = np.asarray(P_heating_W, dtype=float)

    if has_cooling:
        times_cool_s = np.asarray(times_cool_s, dtype=float)
        if not pre_balanced:
            P_heating_W, P_cooling_W = apply_annual_balance(
                P_heating_W, np.asarray(P_cooling_W, dtype=float)
            )

    # Step 1: heating (size_ground_field_length already uses k_s_eff_heating internally)
    L_heat = size_ground_field_length(
        T_fluid_min=T_fluid_min,
        P_W=P_heating_W,
        times_s=times_heat_s,
        field=field,
        brine=brine,
        soil=soil,
        m_dot_per_element=m_dot_per_element_heat,
        L_min=L_min,
        L_max=L_max,
        tol_m=tol_m,
    )

    if not has_cooling:
        return FieldSizingResult(
            L_m=L_heat,
            governing="heating",
            R_heating_K_m_W=field.R_at_L(L_heat, m_dot_per_element_heat, brine, soil),
            R_cooling_K_m_W=None,
        )

    # Step 2: check cooling at L_heat
    T_cool_at_L_heat = _T_fluid_cool_at_L(
        L_heat, P_cooling_W, times_cool_s, field, brine, soil, m_dot_per_element_cool, alpha_cool
    )
    if T_cool_at_L_heat <= T_fluid_max:
        return FieldSizingResult(
            L_m=L_heat,
            governing="heating",
            R_heating_K_m_W=field.R_at_L(L_heat, m_dot_per_element_heat, brine, soil),
            R_cooling_K_m_W=field.R_at_L(L_heat, m_dot_per_element_cool, brine, soil),
        )

    # Step 3: cooling governs — bisect
    g_ils_cool = field.ils_gfunction(times_cool_s, alpha_cool)
    R_sim_cool = field.R_simple(m_dot_per_element_cool, brine, soil)
    L_guess_cool = _initial_guess_cooling_field(
        P_cooling_W, g_ils_cool, field, soil, R_sim_cool, T_fluid_max
    )

    hi_cool = (
        float(np.clip(L_guess_cool, L_heat, L_max))
        if L_guess_cool is not None
        else L_max
    )

    def f_cool(L: float) -> float:
        return (
            _T_fluid_cool_at_L(L, P_cooling_W, times_cool_s, field, brine, soil, m_dot_per_element_cool, alpha_cool)
            - T_fluid_max
        )

    if f_cool(hi_cool) > 0.0:
        hi_cool = L_max

    if f_cool(hi_cool) > 0.0:
        raise ValueError(
            f"Cooling constraint not met at L_max={L_max} m: "
            f"T_fluid_cool={f_cool(hi_cool) + T_fluid_max:.2f} °C > T_max={T_fluid_max} °C. "
            "Increase L_max or check inputs."
        )

    lo, hi = L_heat, hi_cool
    while (hi - lo) > tol_m:
        mid = 0.5 * (lo + hi)
        if f_cool(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    L_cool = 0.5 * (lo + hi)
    return FieldSizingResult(
        L_m=L_cool,
        governing="cooling",
        R_heating_K_m_W=field.R_at_L(L_cool, m_dot_per_element_heat, brine, soil),
        R_cooling_K_m_W=field.R_at_L(L_cool, m_dot_per_element_cool, brine, soil),
    )
