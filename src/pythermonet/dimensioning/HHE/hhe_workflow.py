from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.BHE.borehole_length import (
    FieldSizingResult,
    apply_annual_balance,
    size_ground_field_length_heating_cooling,
)
from pythermonet.dimensioning.ground_field import HHEGroundField
from pythermonet.dimensioning.hydraulic_result import HydraulicResult
from pythermonet.dimensioning.sizing_parameters import SizingParameters
from pythermonet.physics.hydraulics import pressure_loss_per_length as _dp_per_m
from pythermonet.simulation.distribution_pipe_thermal_model import ModeResult as DistModeResult
from pythermonet.simulation.run_distribution_pipe_thermal_model import (
    compute_distribution_pipe_thermal_capacity,
)


@dataclass(frozen=True)
class HHEWorkflowResult:
    sizing: FieldSizingResult
    T_hhe_heat_annual_C: float
    T_hhe_heat_winter_C: float
    T_hhe_heat_peak_C: float
    T_hhe_cool_annual_C: float | None
    T_hhe_cool_winter_C: float | None
    T_hhe_cool_peak_C: float | None
    T_avg_heat_annual_C: float
    T_avg_heat_winter_C: float
    T_avg_heat_peak_C: float
    T_avg_cool_annual_C: float | None
    T_avg_cool_winter_C: float | None
    T_avg_cool_peak_C: float | None
    P_hhe_heating_W: np.ndarray
    P_hhe_cooling_W: np.ndarray | None
    P_full_heating_W: np.ndarray
    P_full_cooling_W: np.ndarray | None
    dist_thermal_heat: DistModeResult
    dist_thermal_cool: DistModeResult | None
    hydraulic: HydraulicResult
    brine: HeatCarrier
    hhe_dp_heat_Pa: float
    hhe_dp_cool_Pa: float | None


def _hhe_mean_temperatures(
    L: float,
    P_W: np.ndarray,
    g: np.ndarray,
    field: HHEGroundField,
    soil: Soil,
    R: float,
    sign: float,  # -1 for heating, +1 for cooling
) -> tuple[float, float, float]:
    """
    Three-pulse mean fluid temperatures via step-load superposition.

    g must be ordered [g(t_peak), g(t_winter), g(t_annual)] (shortest first).
    P_W must be ordered [P_annual, P_winter, P_peak].

    sign = -1 for heating (extraction lowers T), +1 for cooling (injection raises T).
    """
    N = field.n_thermal_elements
    k_s = field.k_s_eff(soil)
    two_pi_ks = 2.0 * math.pi * k_s
    seasonal = field.seasonal_amplitude(soil)

    q_a = float(P_W[0]) / (N * L)
    q_w = float(P_W[1]) / (N * L)
    q_p = float(P_W[2]) / (N * L)

    T_base = field.T_undisturbed(L, soil) + sign * seasonal

    T_annual = T_base + sign * (q_a * g[2] / two_pi_ks + q_a * R)
    T_winter = T_base + sign * ((q_a * g[2] + (q_w - q_a) * g[1]) / two_pi_ks + q_w * R)
    T_peak   = T_base + sign * ((q_a * g[2] + (q_w - q_a) * g[1] + (q_p - q_w) * g[0]) / two_pi_ks + q_p * R)

    return T_annual, T_winter, T_peak


def run_hhe_sizing_workflow(
    *,
    heat_pumps,
    hhe_field: HHEGroundField,
    pipe_infrastructure: PipeInfrastructure,
    hydraulic: HydraulicResult,
    brine: HeatCarrier,
    soil: Soil,
    sizing: SizingParameters,
    T_brine_min_heat: float,
    T_brine_max_cool: float,
) -> HHEWorkflowResult:
    """
    Run the complete HHE sizing workflow.

    Steps: one-sided annual balance (heating only) → distribution pipe
    thermal model → HHE load subtraction → pipe length sizing → system
    brine temperatures.

    Parameters
    ----------
    hhe_field : HHEGroundField
        Ground-field adapter (wraps pipe_infrastructure).
    pipe_infrastructure : PipeInfrastructure
        Raw HHE definition — used for volume calculation.
    """
    P_heat_full = np.asarray(heat_pumps.heating_ground_load_W, dtype=float)
    P_cool_full = (
        np.asarray(heat_pumps.cooling_ground_load_W, dtype=float)
        if heat_pumps.has_cooling else None
    )

    # Two-sided annual balance: for shallow HHE the isothermal surface
    # boundary damps any annual surplus or deficit.  Cooling rejection
    # during the year regenerates the ground for heating, and heating
    # extraction during the year cools the ground for summer cooling.
    # Both annual loads are therefore reduced symmetrically (P_h[0] can
    # go negative if cooling dominates annually, allowing the sizing
    # solver to find a shorter pipe length — the user can elect a deeper
    # burial to exploit the stronger surface-reset effect).
    if P_cool_full is not None:
        P_heat_full, P_cool_full = apply_annual_balance(P_heat_full, P_cool_full)

    times_heat_s = sizing.times_s_peak_heating(heat_pumps.peak_heating_h)
    times_cool_s = (
        sizing.times_s_peak_cooling(heat_pumps.peak_cooling_h)
        if heat_pumps.has_cooling else None
    )

    # Distribution pipe thermal model
    dist_thermal = compute_distribution_pipe_thermal_capacity(
        hydraulic=hydraulic,
        brine=brine,
        soil=soil,
        heat_pumps=heat_pumps,
        times_heat_s=np.flip(times_heat_s),
        times_cool_s=np.flip(times_cool_s) if times_cool_s is not None else None,
        T_brine_min_heat=T_brine_min_heat,
        T_brine_max_cool=T_brine_max_cool if heat_pumps.has_cooling else None,
    )

    # HHE loads = full balanced loads × (1 − distribution fraction)
    P_heating = (1 - dist_thermal["heating"].F_total) * P_heat_full
    P_cooling = (
        (1 - dist_thermal["cooling"].F_total) * P_cool_full
        if heat_pumps.has_cooling else None
    )

    n_pipes = pipe_infrastructure.NParallelPipes
    n_loops = n_pipes // 2  # each loop = one outgoing + one return pipe

    # Size pipe length
    sizing = size_ground_field_length_heating_cooling(
        T_fluid_min=T_brine_min_heat - 0.5 * heat_pumps.deltaT_sys_heat,
        P_heating_W=P_heating,
        times_heat_s=times_heat_s,
        T_fluid_max=T_brine_max_cool + (0.5 * heat_pumps.deltaT_sys_cool if heat_pumps.has_cooling else 0.0),
        P_cooling_W=P_cooling,
        times_cool_s=times_cool_s,
        field=hhe_field,
        brine=brine,
        soil=soil,
        m_dot_per_element_heat=heat_pumps.aggregated_mdot_peak_heat_kg_s / n_loops,
        m_dot_per_element_cool=(
            heat_pumps.aggregated_mdot_peak_cool_kg_s / n_loops
            if heat_pumps.has_cooling else None
        ),
        pre_balanced=True,  # no annual balance for HHE (HFLS already models surface reset)
    )

    L = sizing.L_m
    alpha = float(soil.thermalCond) / (float(soil.rho) * float(soil.c))
    m_dot_heat = heat_pumps.aggregated_mdot_peak_heat_kg_s / n_loops

    # HHE temperatures at each pulse (heating)
    g_heat = hhe_field.compute_gfunction(L, np.asarray(times_heat_s, dtype=float), alpha)
    R_heat = hhe_field.R_at_L(L, m_dot_heat, brine, soil)
    T_h_ann, T_h_win, T_h_peak = _hhe_mean_temperatures(
        L, P_heating, g_heat, hhe_field, soil, R_heat, sign=-1.0
    )

    # Cooling temperatures
    T_c_ann = T_c_win = T_c_peak = None
    if heat_pumps.has_cooling:
        m_dot_cool = heat_pumps.aggregated_mdot_peak_cool_kg_s / n_loops
        g_cool = hhe_field.compute_gfunction(L, np.asarray(times_cool_s, dtype=float), alpha)
        R_cool = hhe_field.R_at_L(L, m_dot_cool, brine, soil)
        T_c_ann, T_c_win, T_c_peak = _hhe_mean_temperatures(
            L, P_cooling, g_cool, hhe_field, soil, R_cool, sign=+1.0
        )

    # Volume-weighted system temperatures
    seg = pipe_infrastructure.traceSegments[0]
    Di_hhe = float(seg.outerDiameter) * (1.0 - 2.0 / float(seg.SDR))
    V_hhe = n_pipes * (math.pi / 4.0) * Di_hhe ** 2 * L

    network = hydraulic.network
    n_parallel = int(network.infrastructure.NParallelPipes)
    L_traces = np.asarray(network.L_traces, dtype=float)
    N_traces = np.asarray(network.N_traces, dtype=int)
    V_dist = float(
        np.sum(N_traces * n_parallel * L_traces * (math.pi / 4.0) * hydraulic.inner_diameter ** 2)
    )
    V_total = V_hhe + V_dist

    def _weighted(T_hhe: float, T_dist: float) -> float:
        return (V_hhe * T_hhe + V_dist * T_dist) / V_total

    T_d = np.asarray(dist_thermal["heating"].T_dimv_C, dtype=float)
    T_avg_heat_annual = _weighted(T_h_ann,  T_d[0])
    T_avg_heat_winter = _weighted(T_h_win,  T_d[1])
    T_avg_heat_peak   = _weighted(T_h_peak, T_d[2])

    T_avg_cool_annual = T_avg_cool_winter = T_avg_cool_peak = None
    if heat_pumps.has_cooling:
        T_dc = np.asarray(dist_thermal["cooling"].T_dimv_C, dtype=float)
        T_avg_cool_annual = _weighted(T_c_ann,  T_dc[0])
        T_avg_cool_winter = _weighted(T_c_win,  T_dc[1])
        T_avg_cool_peak   = _weighted(T_c_peak, T_dc[2])

    # HHE pressure drop (full loop = out + return = 2 × L)
    Q_per_loop_heat = heat_pumps.aggregated_mdot_peak_heat_kg_s / (n_loops * brine.rho)
    hhe_dp_heat_Pa = float(_dp_per_m(brine.rho, brine.dynamicViscosity, Q_per_loop_heat, Di_hhe)) * 2.0 * L

    hhe_dp_cool_Pa: float | None = None
    if heat_pumps.has_cooling:
        Q_per_loop_cool = heat_pumps.aggregated_mdot_peak_cool_kg_s / (n_loops * brine.rho)
        hhe_dp_cool_Pa = float(_dp_per_m(brine.rho, brine.dynamicViscosity, Q_per_loop_cool, Di_hhe)) * 2.0 * L

    return HHEWorkflowResult(
        sizing=sizing,
        T_hhe_heat_annual_C=T_h_ann,
        T_hhe_heat_winter_C=T_h_win,
        T_hhe_heat_peak_C=T_h_peak,
        T_hhe_cool_annual_C=T_c_ann,
        T_hhe_cool_winter_C=T_c_win,
        T_hhe_cool_peak_C=T_c_peak,
        T_avg_heat_annual_C=T_avg_heat_annual,
        T_avg_heat_winter_C=T_avg_heat_winter,
        T_avg_heat_peak_C=T_avg_heat_peak,
        T_avg_cool_annual_C=T_avg_cool_annual,
        T_avg_cool_winter_C=T_avg_cool_winter,
        T_avg_cool_peak_C=T_avg_cool_peak,
        P_hhe_heating_W=P_heating,
        P_hhe_cooling_W=P_cooling,
        P_full_heating_W=P_heat_full,
        P_full_cooling_W=P_cool_full,
        dist_thermal_heat=dist_thermal["heating"],
        dist_thermal_cool=dist_thermal["cooling"] if heat_pumps.has_cooling else None,
        hydraulic=hydraulic,
        brine=brine,
        hhe_dp_heat_Pa=hhe_dp_heat_Pa,
        hhe_dp_cool_Pa=hhe_dp_cool_Pa,
    )


def print_hhe_results(result: HHEWorkflowResult, pipe_infrastructure: PipeInfrastructure) -> None:
    """Print formatted dimensioning results to terminal."""
    import math as _math
    from pythermonet.physics.hydraulics import pressure_loss_per_length as _dp_per_m2

    hydraulic    = result.hydraulic
    brine        = result.brine
    sizing       = result.sizing
    has_cooling  = result.P_hhe_cooling_W is not None
    network      = hydraulic.network
    n_traces     = len(hydraulic.outer_diameter)

    # ── velocities and dp/m per trace ──────────────────────────────────────
    A = _math.pi * hydraulic.inner_diameter ** 2 / 4.0
    v_heat = hydraulic.m3_s_heating / A
    dp_m_heat = np.array([
        float(_dp_per_m2(brine.rho, brine.dynamicViscosity,
                         float(hydraulic.m3_s_heating[i]),
                         float(hydraulic.inner_diameter[i])))
        for i in range(n_traces)
    ])

    has_cool_hydro = has_cooling and hydraulic.m3_s_cooling is not None
    if has_cool_hydro:
        v_cool = hydraulic.m3_s_cooling / A
        dp_m_cool = np.array([
            float(_dp_per_m2(brine.rho, brine.dynamicViscosity,
                             float(hydraulic.m3_s_cooling[i]),
                             float(hydraulic.inner_diameter[i])))
            for i in range(n_traces)
        ])
    else:
        v_cool = dp_m_cool = None

    # ── distribution network table ─────────────────────────────────────────
    TC, DC, VC, RC, PC = 14, 8, 10, 9, 12

    base_cols = [TC, DC, DC, VC, RC, PC]
    cols = base_cols + ([VC, RC, PC] if has_cool_hydro else [])

    def _border(left, mid, right):
        return left + mid.join('─' * w for w in cols) + right

    def _cell(val: str, w: int, align: str = 'r') -> str:
        inner = w - 2
        return f' {val:<{inner}} ' if align == 'l' else f' {val:>{inner}} '

    def _hrow() -> str:
        cells = [
            _cell('Trace', TC, 'l'),
            _cell('Do[mm]', DC),
            _cell('Di[mm]', DC),
            _cell('v_H[m/s]', VC),
            _cell('Re_H', RC),
            _cell('dp_H[Pa/m]', PC),
        ]
        if has_cool_hydro:
            cells += [_cell('v_C[m/s]', VC), _cell('Re_C', RC), _cell('dp_C[Pa/m]', PC)]
        return '│' + '│'.join(cells) + '│'

    def _drow(i: int) -> str:
        name = network.trace_names[i]
        if len(name) > TC - 2:
            name = name[:TC - 5] + '...'
        cells = [
            _cell(name, TC, 'l'),
            _cell(f'{hydraulic.outer_diameter[i] * 1000:.1f}', DC),
            _cell(f'{hydraulic.inner_diameter[i] * 1000:.1f}', DC),
            _cell(f'{v_heat[i]:.3f}', VC),
            _cell(f'{hydraulic.Re_heating[i]:,.0f}', RC),
            _cell(f'{dp_m_heat[i]:.1f}', PC),
        ]
        if has_cool_hydro:
            cells += [
                _cell(f'{v_cool[i]:.3f}', VC),
                _cell(f'{hydraulic.Re_cooling[i]:,.0f}', RC),
                _cell(f'{dp_m_cool[i]:.1f}', PC),
            ]
        return '│' + '│'.join(cells) + '│'

    print('\n  Distribution network')
    print(_border('┌', '┬', '┐'))
    print(_hrow())
    print(_border('├', '┼', '┤'))
    for i in range(n_traces):
        print(_drow(i))
    print(_border('└', '┴', '┘'))

    # ── HHE sizing + pressure drop ─────────────────────────────────────────
    seg = pipe_infrastructure.traceSegments[0]
    Di_hhe = float(seg.outerDiameter) * (1.0 - 2.0 / float(seg.SDR))
    print()
    print(f'  HHE loop length  :  {2.0 * sizing.L_m:.2f} m  (governed by {sizing.governing})')
    print(f'  Dist. fraction   :  {result.dist_thermal_heat.F_total * 100:.1f} %  (heating, distribution grid)')
    if result.dist_thermal_cool is not None:
        print(f'  Dist. fraction   :  {result.dist_thermal_cool.F_total * 100:.1f} %  (cooling, distribution grid)')
    print(f'  HHE Do / Di      :  {float(seg.outerDiameter)*1000:.1f} mm / {Di_hhe*1000:.1f} mm  '
          f'(SDR {float(seg.SDR):.0f})')
    print(f'  N parallel pipes :  {pipe_infrastructure.NParallelPipes}')
    print(f'  Burial depth     :  {float(pipe_infrastructure.burialDepth):.2f} m')
    hhe_dp_m_heat = result.hhe_dp_heat_Pa / (2.0 * sizing.L_m)
    print(f'  HHE ΔP (heating) :  {result.hhe_dp_heat_Pa:,.0f} Pa  |  {hhe_dp_m_heat:.1f} Pa/m')
    if result.hhe_dp_cool_Pa is not None:
        hhe_dp_m_cool = result.hhe_dp_cool_Pa / (2.0 * sizing.L_m)
        print(f'  HHE ΔP (cooling) :  {result.hhe_dp_cool_Pa:,.0f} Pa  |  {hhe_dp_m_cool:.1f} Pa/m')

    # ── brine temperatures table ───────────────────────────────────────────
    LW, VW = 36, 14

    top_t = f"┌{'─'*LW}┬{'─'*VW}┬{'─'*VW}┬{'─'*VW}┐"
    div_t = f"├{'─'*LW}┼{'─'*VW}┼{'─'*VW}┼{'─'*VW}┤"
    bot_t = f"└{'─'*LW}┴{'─'*VW}┴{'─'*VW}┴{'─'*VW}┘"

    def hdr() -> str:
        return f"│{'':{LW}}│{'Annual':^{VW}}│{'Winter':^{VW}}│{'Peak':^{VW}}│"

    def trow(label: str, a: float, w: float, p: float) -> str:
        fa, fw, fp = f'{a:.2f}', f'{w:.2f}', f'{p:.2f}'
        return f"│ {label:<{LW-2}} │{fa:>{VW-1}} │{fw:>{VW-1}} │{fp:>{VW-1}} │"

    print()
    lines = [top_t, hdr(), div_t]
    lines.append(trow(
        'System mean temp (heating)   [°C]',
        result.T_avg_heat_annual_C,
        result.T_avg_heat_winter_C,
        result.T_avg_heat_peak_C,
    ))
    lines.append(bot_t)
    print('\n'.join(lines))
