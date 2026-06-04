from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from pythermonet.components.vhe_field import VHEField
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.hydraulic_result import HydraulicResult
from pythermonet.dimensioning.BHE.borehole_length import FieldSizingResult, apply_annual_balance
from pythermonet.simulation.distribution_pipe_thermal_model import ModeResult as DistModeResult


@dataclass(frozen=True)
class SystemBrineTemperatureResult:
    """
    Volume-weighted mean brine temperature for the complete ground-side circuit
    (BHEs + distribution pipes) at each of the three load pulses.

    Convention
    ----------
    Each temperature is the mean BHE fluid temperature at that pulse,
    computed via the ASHRAE step-load superposition (T_g − ΔT_ground − q·Rb
    for heating, + for cooling).  Because the distribution pipes carry both
    supply and return legs their mean equals the BHE mean, so the
    volume-weighted system mean equals the BHE mean directly.

    Fields
    ------
    T_avg_heat_{annual,winter,peak}_C : float
        Mean brine temperature at each heating pulse [°C].
    T_avg_cool_{annual,winter,peak}_C : float
        Mean brine temperature at each cooling pulse [°C].
    V_bhe_m3 : float
        Brine volume in BHEs (both U-pipe legs) [m³].
    V_dist_m3 : float
        Brine volume in distribution pipes (supply + return) [m³].
    V_total_m3 : float
        Total brine volume [m³].
    bhe_fraction : float
        V_bhe / V_total.
    """

    T_avg_heat_annual_C: float
    T_avg_heat_winter_C: float
    T_avg_heat_peak_C: float

    T_avg_cool_annual_C: float | None
    T_avg_cool_winter_C: float | None
    T_avg_cool_peak_C: float | None

    T_bhe_heat_annual_C: float
    T_bhe_heat_winter_C: float
    T_bhe_heat_peak_C: float

    T_bhe_cool_annual_C: float | None
    T_bhe_cool_winter_C: float | None
    T_bhe_cool_peak_C: float | None

    T_dist_heat_annual_C: float | None
    T_dist_heat_winter_C: float | None
    T_dist_heat_peak_C: float | None

    T_dist_cool_annual_C: float | None
    T_dist_cool_winter_C: float | None
    T_dist_cool_peak_C: float | None

    V_bhe_m3: float
    V_dist_m3: float
    V_total_m3: float
    bhe_fraction: float


def _bhe_mean_temperatures_heating(
    H_m: float,
    P_bhe_W: np.ndarray,
    g: np.ndarray,
    vhe_field: VHEField,
    soil: Soil,
    Rb: float,
) -> tuple[float, float, float]:
    """
    BHE mean fluid temperatures for the three heating pulses at H_m.

    Uses step-load superposition.  g = [g(t_peak), g(t_winter), g(t_annual)].
    P_bhe_W = [P_annual, P_winter, P_peak].

    Returns (T_annual, T_winter, T_peak).
    """
    N = vhe_field.n_boreholes
    k_s = float(soil.thermal_conductivity)
    T_g = float(soil.surface_temperature) + float(soil.geothermal_heat_flux) * H_m / (2.0 * k_s)
    two_pi_ks = 2.0 * math.pi * k_s

    q_a = float(P_bhe_W[0]) / (N * H_m)
    q_w = float(P_bhe_W[1]) / (N * H_m)
    q_p = float(P_bhe_W[2]) / (N * H_m)

    # Annual: only the annual step load has run
    T_annual = T_g - q_a * g[2] / two_pi_ks - q_a * Rb

    # Winter: annual + winter increment
    T_winter = T_g - (q_a * g[2] + (q_w - q_a) * g[1]) / two_pi_ks - q_w * Rb

    # Peak: all three steps
    T_peak = T_g - (q_a * g[2] + (q_w - q_a) * g[1] + (q_p - q_w) * g[0]) / two_pi_ks - q_p * Rb

    return T_annual, T_winter, T_peak


def _bhe_mean_temperatures_cooling(
    H_m: float,
    P_bhe_W: np.ndarray,
    g: np.ndarray,
    vhe_field: VHEField,
    soil: Soil,
    Rb: float,
) -> tuple[float, float, float]:
    """
    BHE mean fluid temperatures for the three cooling pulses at H_m.

    Returns (T_annual, T_winter, T_peak).
    """
    N = vhe_field.n_boreholes
    k_s = float(soil.thermal_conductivity)
    T_g = float(soil.surface_temperature) + float(soil.geothermal_heat_flux) * H_m / (2.0 * k_s)
    two_pi_ks = 2.0 * math.pi * k_s

    q_a = float(P_bhe_W[0]) / (N * H_m)
    q_w = float(P_bhe_W[1]) / (N * H_m)
    q_p = float(P_bhe_W[2]) / (N * H_m)

    T_annual = T_g + q_a * g[2] / two_pi_ks + q_a * Rb
    T_winter = T_g + (q_a * g[2] + (q_w - q_a) * g[1]) / two_pi_ks + q_w * Rb
    T_peak   = T_g + (q_a * g[2] + (q_w - q_a) * g[1] + (q_p - q_w) * g[0]) / two_pi_ks + q_p * Rb

    return T_annual, T_winter, T_peak


def compute_system_brine_temperatures(
    *,
    sizing: FieldSizingResult,
    vhe_field: VHEField,
    hydraulic: HydraulicResult,
    P_heating_W: np.ndarray,
    times_heat_s: np.ndarray,
    P_cooling_W: np.ndarray | None = None,
    times_cool_s: np.ndarray | None = None,
    soil: Soil,
    dist_thermal_heat: DistModeResult | None = None,
    dist_thermal_cool: DistModeResult | None = None,
    pre_balanced: bool = False,
) -> SystemBrineTemperatureResult:
    """
    Compute volume-weighted mean brine temperatures for all three load pulses.

    Parameters
    ----------
    sizing : BoreholeSizingResult
        Contains the final H_m and mode-specific Rb values.
    vhe_field : VHEField
        Borehole field geometry.
    hydraulic : HydraulicResult
        Hydraulic dimensioning result.
    heat_pumps : HeatPumps
        Provides deltaT_sys_heat and deltaT_sys_cool.
    P_heating_W : array, shape (3,)
        BHE heating loads [W]: [P_annual, P_winter, P_peak].
    times_heat_s : array, shape (3,)
        Heating pulse durations [s]: [t_peak, t_winter, t_annual].
    P_cooling_W : array, shape (3,)
        BHE cooling loads [W]: [P_annual, P_winter, P_peak].
    times_cool_s : array, shape (3,)
        Cooling pulse durations [s]: [t_peak, t_winter, t_annual].
    soil : Soil
        Ground properties.
    dist_thermal_heat : DistModeResult, optional
        Result from the distribution pipe thermal model for heating.
        ``T_dimv_C`` must be ordered [annual, winter, peak] (i.e. the
        distribution model was run with times in descending order).
        When provided the reported system temperatures are the
        volume-weighted average of distribution pipe and BHE temperatures,
        matching the convention of the reference implementation.
    dist_thermal_cool : DistModeResult, optional
        Same as dist_thermal_heat but for cooling mode.
    """
    has_cooling = P_cooling_W is not None

    H_m = sizing.L_m
    alpha = float(soil.thermal_conductivity) / (float(soil.density) * float(soil.specific_heat))
    P_heating_W = np.asarray(P_heating_W, dtype=float)

    if has_cooling and not pre_balanced:
        P_heating_W, P_cooling_W = apply_annual_balance(
            P_heating_W, np.asarray(P_cooling_W, dtype=float)
        )

    field_H = replace(vhe_field, H_m=H_m)

    # ------------------------------------------------------------------
    # G-functions at the final H_m
    # ------------------------------------------------------------------
    g_heat = field_H.compute_pygfunctions(
        times_s=times_heat_s, alpha_m2_s=alpha
    )

    # ------------------------------------------------------------------
    # Fluid volumes (needed for weighted averaging below)
    # ------------------------------------------------------------------
    Di_bhe = float(vhe_field.pipe.outer_diameter) * (1.0 - 2.0 / float(vhe_field.pipe.sdr))
    V_bhe = vhe_field.n_boreholes * 2.0 * (math.pi / 4.0) * Di_bhe ** 2 * H_m

    network = hydraulic.network
    n_parallel = int(network.infrastructure.NParallelPipes)
    L_traces = np.asarray(network.L_traces, dtype=float)
    N_traces = np.asarray(network.N_traces, dtype=int)
    V_dist = float(
        np.sum(N_traces * n_parallel * L_traces * (math.pi / 4.0) * hydraulic.inner_diameter ** 2)
    )

    V_total = V_bhe + V_dist

    # ------------------------------------------------------------------
    # BHE mean temperatures at each heating pulse
    # ------------------------------------------------------------------
    T_h_ann, T_h_win, T_h_peak = _bhe_mean_temperatures_heating(
        H_m, P_heating_W, g_heat, vhe_field, soil, sizing.R_heating_K_m_W
    )

    # ------------------------------------------------------------------
    # System mean temperatures
    #
    # When distribution pipe thermal results are provided the system
    # temperature is the volume-weighted average of the distribution pipe
    # mean temperature and the BHE mean temperature, matching the
    # reference implementation.  Distribution pipe T_dimv_C must be
    # ordered [annual, winter, peak].
    # When not provided (backward-compatible) only the BHE mean is used.
    # ------------------------------------------------------------------
    T_dist_heat_ann: float | None = None
    T_dist_heat_win: float | None = None
    T_dist_heat_peak_val: float | None = None

    if dist_thermal_heat is not None:
        T_d = np.asarray(dist_thermal_heat.T_dimv_C, dtype=float)  # [annual, winter, peak]
        T_dist_heat_ann  = float(T_d[0])
        T_dist_heat_win  = float(T_d[1])
        T_dist_heat_peak_val = float(T_d[2])
        T_avg_heat_annual = (V_bhe * T_h_ann  + V_dist * T_d[0]) / V_total
        T_avg_heat_winter = (V_bhe * T_h_win  + V_dist * T_d[1]) / V_total
        T_avg_heat_peak   = (V_bhe * T_h_peak + V_dist * T_d[2]) / V_total
    else:
        T_avg_heat_annual = T_h_ann
        T_avg_heat_winter = T_h_win
        T_avg_heat_peak   = T_h_peak

    T_avg_cool_annual = T_avg_cool_winter = T_avg_cool_peak = None
    T_c_ann = T_c_win = T_c_peak = None
    T_dist_cool_ann: float | None = None
    T_dist_cool_win: float | None = None
    T_dist_cool_peak_val: float | None = None

    if has_cooling:
        g_cool = field_H.compute_pygfunctions(
            times_s=times_cool_s, alpha_m2_s=alpha
        )
        T_c_ann, T_c_win, T_c_peak = _bhe_mean_temperatures_cooling(
            H_m, P_cooling_W, g_cool, vhe_field, soil, sizing.R_cooling_K_m_W
        )
        if dist_thermal_cool is not None:
            T_dc = np.asarray(dist_thermal_cool.T_dimv_C, dtype=float)
            T_dist_cool_ann  = float(T_dc[0])
            T_dist_cool_win  = float(T_dc[1])
            T_dist_cool_peak_val = float(T_dc[2])
            T_avg_cool_annual = (V_bhe * T_c_ann  + V_dist * T_dc[0]) / V_total
            T_avg_cool_winter = (V_bhe * T_c_win  + V_dist * T_dc[1]) / V_total
            T_avg_cool_peak   = (V_bhe * T_c_peak + V_dist * T_dc[2]) / V_total
        else:
            T_avg_cool_annual = T_c_ann
            T_avg_cool_winter = T_c_win
            T_avg_cool_peak   = T_c_peak

    return SystemBrineTemperatureResult(
        T_avg_heat_annual_C=T_avg_heat_annual,
        T_avg_heat_winter_C=T_avg_heat_winter,
        T_avg_heat_peak_C=T_avg_heat_peak,
        T_avg_cool_annual_C=T_avg_cool_annual,
        T_avg_cool_winter_C=T_avg_cool_winter,
        T_avg_cool_peak_C=T_avg_cool_peak,
        T_bhe_heat_annual_C=T_h_ann,
        T_bhe_heat_winter_C=T_h_win,
        T_bhe_heat_peak_C=T_h_peak,
        T_bhe_cool_annual_C=T_c_ann,
        T_bhe_cool_winter_C=T_c_win,
        T_bhe_cool_peak_C=T_c_peak,
        T_dist_heat_annual_C=T_dist_heat_ann,
        T_dist_heat_winter_C=T_dist_heat_win,
        T_dist_heat_peak_C=T_dist_heat_peak_val,
        T_dist_cool_annual_C=T_dist_cool_ann,
        T_dist_cool_winter_C=T_dist_cool_win,
        T_dist_cool_peak_C=T_dist_cool_peak_val,
        V_bhe_m3=V_bhe,
        V_dist_m3=V_dist,
        V_total_m3=V_total,
        bhe_fraction=V_bhe / V_total,
    )
