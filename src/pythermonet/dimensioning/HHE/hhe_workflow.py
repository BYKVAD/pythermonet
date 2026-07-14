from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pythermonet.components.ground_loads import GroundLoads
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
from pythermonet.simulation.distribution_pipe_thermal_model import ThermonetPerformance
from pythermonet.simulation.run_distribution_pipe_thermal_model import (
    compute_distribution_pipe_thermal_capacity,
)


@dataclass(frozen=True)
class HHEWorkflowResult:
    sizing: FieldSizingResult
    temperature_hhe_annual_heating: float              # [°C]
    temperature_hhe_winter_heating: float              # [°C]
    temperature_hhe_peak_heating: float                # [°C]
    temperature_hhe_annual_cooling: float | None       # [°C]
    temperature_hhe_summer_cooling: float | None       # [°C]
    temperature_hhe_peak_cooling: float | None         # [°C]
    temperature_system_annual_heating: float           # [°C]
    temperature_system_winter_heating: float           # [°C]
    temperature_system_peak_heating: float             # [°C]
    temperature_system_annual_cooling: float | None    # [°C]
    temperature_system_summer_cooling: float | None    # [°C]
    temperature_system_peak_cooling: float | None      # [°C]
    loads_hhe_heating: np.ndarray                      # [W]
    loads_hhe_cooling: np.ndarray | None               # [W]
    loads_total_heating: np.ndarray                    # [W]
    loads_total_cooling: np.ndarray | None             # [W]
    performance_thermonet_heating: ThermonetPerformance
    performance_thermonet_cooling: ThermonetPerformance | None
    hydraulic: HydraulicResult
    brine: HeatCarrier
    pressure_loss_hhe_heating: float                   # [Pa]
    pressure_loss_hhe_cooling: float | None            # [Pa]


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
    ground_loads: GroundLoads,
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
    P_heat_full = np.asarray(ground_loads.loads_ground_heating, dtype=float)
    P_cool_full = (
        np.asarray(ground_loads.loads_ground_cooling, dtype=float)
        if ground_loads.has_cooling else None
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

    times_heat_s = sizing.times_s_peak_heating(ground_loads.hours_peak_heating)
    times_cool_s = (
        sizing.times_s_peak_cooling(ground_loads.hours_peak_cooling)
        if ground_loads.has_cooling else None
    )

    # Distribution pipe thermal model
    dist_thermal = compute_distribution_pipe_thermal_capacity(
        hydraulic=hydraulic,
        brine=brine,
        soil=soil,
        ground_loads=ground_loads,
        times_heat_s=np.flip(times_heat_s),
        times_cool_s=np.flip(times_cool_s) if times_cool_s is not None else None,
        T_brine_min_heat=T_brine_min_heat,
        T_brine_max_cool=T_brine_max_cool if ground_loads.has_cooling else None,
    )

    # HHE loads = full balanced loads × (1 − distribution fraction)
    P_heating = (1 - dist_thermal["heating"].load_supply_fraction) * P_heat_full
    P_cooling = (
        (1 - dist_thermal["cooling"].load_supply_fraction) * P_cool_full
        if ground_loads.has_cooling else None
    )

    n_pipes = pipe_infrastructure.n_pipes_parallel
    n_loops = n_pipes // 2  # each loop = one outgoing + one return pipe

    # Size pipe length
    sizing = size_ground_field_length_heating_cooling(
        T_fluid_min=T_brine_min_heat - 0.5 * ground_loads.temperature_delta_brine_flow_weighted_heating,
        P_heating_W=P_heating,
        times_heat_s=times_heat_s,
        T_fluid_max=T_brine_max_cool + (0.5 * ground_loads.temperature_delta_brine_flow_weighted_cooling if ground_loads.has_cooling else 0.0),
        P_cooling_W=P_cooling,
        times_cool_s=times_cool_s,
        field=hhe_field,
        brine=brine,
        soil=soil,
        m_dot_per_element_heat=ground_loads.mass_flow_peak_summed_heating / n_loops,
        m_dot_per_element_cool=(
            ground_loads.mass_flow_peak_summed_cooling / n_loops
            if ground_loads.has_cooling else None
        ),
        pre_balanced=True,  # no annual balance for HHE (HFLS already models surface reset)
    )

    L = sizing.length_element
    alpha_heat = hhe_field.k_s_eff_heating(soil) / (float(soil.density) * float(soil.specific_heat))
    m_dot_heat = ground_loads.mass_flow_peak_summed_heating / n_loops

    # HHE temperatures at each pulse (heating)
    g_heat = hhe_field.compute_gfunction(L, np.asarray(times_heat_s, dtype=float), alpha_heat)
    R_heat = hhe_field.R_at_L(L, m_dot_heat, brine, soil)
    T_h_ann, T_h_win, T_h_peak = _hhe_mean_temperatures(
        L, P_heating, g_heat, hhe_field, soil, R_heat, sign=-1.0
    )

    # Cooling temperatures
    T_c_ann = T_c_win = T_c_peak = None
    if ground_loads.has_cooling:
        alpha_cool = hhe_field.k_s_eff_cooling(soil) / (float(soil.density) * float(soil.specific_heat))
        m_dot_cool = ground_loads.mass_flow_peak_summed_cooling / n_loops
        g_cool = hhe_field.compute_gfunction(L, np.asarray(times_cool_s, dtype=float), alpha_cool)
        R_cool = hhe_field.R_at_L(L, m_dot_cool, brine, soil)
        T_c_ann, T_c_win, T_c_peak = _hhe_mean_temperatures(
            L, P_cooling, g_cool, hhe_field, soil, R_cool, sign=+1.0
        )

    # Volume-weighted system temperatures
    seg = pipe_infrastructure.segments_trace[0]
    Di_hhe = float(seg.diameter_outer) * (1.0 - 2.0 / float(seg.sdr))
    V_hhe = n_pipes * (math.pi / 4.0) * Di_hhe ** 2 * L

    network = hydraulic.network
    n_parallel = int(network.pipe_infrastructure.n_pipes_parallel)
    trace_lengths = np.asarray(network.lengths_trace, dtype=float)
    trace_counts = np.asarray(network.counts_trace, dtype=int)
    V_dist = float(
        np.sum(trace_counts * n_parallel * trace_lengths * (math.pi / 4.0) * hydraulic.diameters_inner ** 2)
    )
    V_total = V_hhe + V_dist

    def _weighted(T_hhe: float, T_dist: float) -> float:
        return (V_hhe * T_hhe + V_dist * T_dist) / V_total

    T_d = np.asarray(dist_thermal["heating"].temperatures_mean, dtype=float)
    T_avg_heat_annual = _weighted(T_h_ann,  T_d[0])
    T_avg_heat_winter = _weighted(T_h_win,  T_d[1])
    T_avg_heat_peak   = _weighted(T_h_peak, T_d[2])

    T_avg_cool_annual = T_avg_cool_summer = T_avg_cool_peak = None
    if ground_loads.has_cooling:
        T_dc = np.asarray(dist_thermal["cooling"].temperatures_mean, dtype=float)
        T_avg_cool_annual = _weighted(T_c_ann,  T_dc[0])
        T_avg_cool_summer = _weighted(T_c_win,  T_dc[1])
        T_avg_cool_peak   = _weighted(T_c_peak, T_dc[2])

    # HHE pressure drop (full loop = out + return = 2 × L)
    Q_per_loop_heat = ground_loads.mass_flow_peak_summed_heating / (n_loops * brine.density)
    pressure_loss_hhe_heating = float(_dp_per_m(brine.density, brine.dynamic_viscosity, Q_per_loop_heat, Di_hhe)) * 2.0 * L

    pressure_loss_hhe_cooling: float | None = None
    if ground_loads.has_cooling:
        Q_per_loop_cool = ground_loads.mass_flow_peak_summed_cooling / (n_loops * brine.density)
        pressure_loss_hhe_cooling = float(_dp_per_m(brine.density, brine.dynamic_viscosity, Q_per_loop_cool, Di_hhe)) * 2.0 * L

    return HHEWorkflowResult(
        sizing=sizing,
        temperature_hhe_annual_heating=T_h_ann,
        temperature_hhe_winter_heating=T_h_win,
        temperature_hhe_peak_heating=T_h_peak,
        temperature_hhe_annual_cooling=T_c_ann,
        temperature_hhe_summer_cooling=T_c_win,
        temperature_hhe_peak_cooling=T_c_peak,
        temperature_system_annual_heating=T_avg_heat_annual,
        temperature_system_winter_heating=T_avg_heat_winter,
        temperature_system_peak_heating=T_avg_heat_peak,
        temperature_system_annual_cooling=T_avg_cool_annual,
        temperature_system_summer_cooling=T_avg_cool_summer,
        temperature_system_peak_cooling=T_avg_cool_peak,
        loads_hhe_heating=P_heating,
        loads_hhe_cooling=P_cooling,
        loads_total_heating=P_heat_full,
        loads_total_cooling=P_cool_full,
        performance_thermonet_heating=dist_thermal["heating"],
        performance_thermonet_cooling=dist_thermal["cooling"] if ground_loads.has_cooling else None,
        hydraulic=hydraulic,
        brine=brine,
        pressure_loss_hhe_heating=pressure_loss_hhe_heating,
        pressure_loss_hhe_cooling=pressure_loss_hhe_cooling,
    )


def print_hhe_results(result: HHEWorkflowResult, pipe_infrastructure: PipeInfrastructure) -> None:
    """Print formatted dimensioning results to terminal."""
    import math as _math
    from pythermonet.physics.hydraulics import pressure_loss_per_length as _dp_per_m2

    hydraulic    = result.hydraulic
    brine        = result.brine
    sizing       = result.sizing
    has_cooling  = result.loads_hhe_cooling is not None
    network      = hydraulic.network
    n_traces     = len(hydraulic.diameters_outer)

    # ── velocities and dp/m per trace ──────────────────────────────────────
    A = _math.pi * hydraulic.diameters_inner ** 2 / 4.0
    v_heat = hydraulic.volume_flow_rates_peak_heating / A
    dp_m_heat = np.array([
        float(_dp_per_m2(brine.density, brine.dynamic_viscosity,
                         float(hydraulic.volume_flow_rates_peak_heating[i]),
                         float(hydraulic.diameters_inner[i])))
        for i in range(n_traces)
    ])

    has_cool_hydro = has_cooling and hydraulic.volume_flow_rates_peak_cooling is not None
    if has_cool_hydro:
        v_cool = hydraulic.volume_flow_rates_peak_cooling / A
        dp_m_cool = np.array([
            float(_dp_per_m2(brine.density, brine.dynamic_viscosity,
                             float(hydraulic.volume_flow_rates_peak_cooling[i]),
                             float(hydraulic.diameters_inner[i])))
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
        name = network.names_trace[i]
        if len(name) > TC - 2:
            name = name[:TC - 5] + '...'
        cells = [
            _cell(name, TC, 'l'),
            _cell(f'{hydraulic.diameters_outer[i] * 1000:.1f}', DC),
            _cell(f'{hydraulic.diameters_inner[i] * 1000:.1f}', DC),
            _cell(f'{v_heat[i]:.3f}', VC),
            _cell(f'{hydraulic.reynolds_numbers_heating[i]:,.0f}', RC),
            _cell(f'{dp_m_heat[i]:.1f}', PC),
        ]
        if has_cool_hydro:
            cells += [
                _cell(f'{v_cool[i]:.3f}', VC),
                _cell(f'{hydraulic.reynolds_numbers_cooling[i]:,.0f}', RC),
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
    seg = pipe_infrastructure.segments_trace[0]
    Di_hhe = float(seg.diameter_outer) * (1.0 - 2.0 / float(seg.sdr))
    print()
    print(f'  HHE loop length  :  {2.0 * sizing.length_element:.2f} m  (governed by {sizing.governing_mode})')
    print(f'  Dist. fraction   :  {result.performance_thermonet_heating.load_supply_fraction * 100:.1f} %  (heating, distribution grid)')
    if result.performance_thermonet_cooling is not None:
        print(f'  Dist. fraction   :  {result.performance_thermonet_cooling.load_supply_fraction * 100:.1f} %  (cooling, distribution grid)')
    print(f'  HHE Do / Di      :  {float(seg.diameter_outer)*1000:.1f} mm / {Di_hhe*1000:.1f} mm  '
          f'(sdr {float(seg.sdr):.0f})')
    print(f'  N parallel pipes :  {pipe_infrastructure.n_pipes_parallel}')
    print(f'  Burial depth     :  {float(pipe_infrastructure.burial_depth):.2f} m')
    hhe_dp_m_heat = result.pressure_loss_hhe_heating / (2.0 * sizing.length_element)
    print(f'  HHE ΔP (heating) :  {result.pressure_loss_hhe_heating:,.0f} Pa  |  {hhe_dp_m_heat:.1f} Pa/m')
    if result.pressure_loss_hhe_cooling is not None:
        hhe_dp_m_cool = result.pressure_loss_hhe_cooling / (2.0 * sizing.length_element)
        print(f'  HHE ΔP (cooling) :  {result.pressure_loss_hhe_cooling:,.0f} Pa  |  {hhe_dp_m_cool:.1f} Pa/m')

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
        result.temperature_system_annual_heating,
        result.temperature_system_winter_heating,
        result.temperature_system_peak_heating,
    ))
    lines.append(bot_t)
    print('\n'.join(lines))
