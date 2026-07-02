from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pythermonet.components.ground_loads import GroundLoads
from pythermonet.components.vhe_field import VHEField
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.BHE.borehole_length import (
    FieldSizingResult,
    apply_annual_balance,
    size_borehole_length_heating_cooling,
)
from pythermonet.dimensioning.hydraulic_result import HydraulicResult
from pythermonet.dimensioning.sizing_parameters import SizingParameters
from pythermonet.dimensioning.system_temperatures import (
    SystemBrineTemperatureResult,
    compute_system_brine_temperatures,
)
from pythermonet.physics.hydraulics import pressure_loss_per_length as _dp_per_m
from pythermonet.simulation.distribution_pipe_thermal_model import ModeResult as DistModeResult
from pythermonet.simulation.run_distribution_pipe_thermal_model import (
    compute_distribution_pipe_thermal_capacity,
)


@dataclass(frozen=True)
class BHEWorkflowResult:
    sizing: FieldSizingResult
    sys_temps: SystemBrineTemperatureResult
    P_bhe_heating_W: np.ndarray
    P_bhe_cooling_W: np.ndarray | None
    P_full_heating_W: np.ndarray
    P_full_cooling_W: np.ndarray | None
    dist_thermal_heat: DistModeResult
    dist_thermal_cool: DistModeResult | None
    hydraulic: HydraulicResult
    brine: HeatCarrier
    bhe_dp_heat_Pa: float
    bhe_dp_cool_Pa: float | None


def run_bhe_sizing_workflow(
    *,
    ground_loads: GroundLoads,
    vhe_field: VHEField,
    hydraulic: HydraulicResult,
    brine: HeatCarrier,
    soil: Soil,
    sizing: SizingParameters,
    T_brine_min_heat: float,
    T_brine_max_cool: float,
) -> BHEWorkflowResult:
    """
    Run the complete BHE sizing workflow from ground loads to system temperatures.

    Covers: annual balance, distribution pipe thermal model, BHE load
    subtraction, borehole length sizing, and system brine temperature
    computation.
    """
    # Annual balance
    P_heat_full = np.asarray(ground_loads.ground_load_heating, dtype=float)
    P_cool_full = (
        np.asarray(ground_loads.ground_load_cooling, dtype=float)
        if ground_loads.has_cooling else None
    )
    if P_cool_full is not None:
        P_heat_full, P_cool_full = apply_annual_balance(P_heat_full.copy(), P_cool_full.copy())

    times_heat_s = sizing.times_s_peak_heating(ground_loads.peak_hours_heating)
    times_cool_s = (
        sizing.times_s_peak_cooling(ground_loads.peak_hours_cooling)
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

    # BHE loads = full balanced loads × (1 − distribution fraction)
    P_heating = (1 - dist_thermal["heating"].F_total) * P_heat_full
    P_cooling = (
        (1 - dist_thermal["cooling"].F_total) * P_cool_full
        if ground_loads.has_cooling else None
    )

    n_boreholes = vhe_field.n_boreholes

    # Size borehole length
    field_sizing = size_borehole_length_heating_cooling(
        T_fluid_min=T_brine_min_heat - 0.5 * ground_loads.flow_weighted_brine_delta_temperature_heating,
        P_heating_W=P_heating,
        times_heat_s=times_heat_s,
        T_fluid_max=T_brine_max_cool + (
            0.5 * ground_loads.flow_weighted_brine_delta_temperature_cooling if ground_loads.has_cooling else 0.0
        ),
        P_cooling_W=P_cooling,
        times_cool_s=times_cool_s,
        vhe_field=vhe_field,
        brine=brine,
        soil=soil,
        m_dot_per_borehole_heat_kg_s=ground_loads.summed_peak_mass_flow_heating / n_boreholes,
        m_dot_per_borehole_cool_kg_s=(
            ground_loads.summed_peak_mass_flow_cooling / n_boreholes
            if ground_loads.has_cooling else None
        ),
        pre_balanced=True,
    )

    # System brine temperatures
    sys_temps = compute_system_brine_temperatures(
        sizing=field_sizing,
        vhe_field=vhe_field,
        hydraulic=hydraulic,
        P_heating_W=P_heating,
        times_heat_s=times_heat_s,
        P_cooling_W=P_cooling,
        times_cool_s=times_cool_s,
        soil=soil,
        dist_thermal_heat=dist_thermal["heating"],
        dist_thermal_cool=dist_thermal["cooling"] if ground_loads.has_cooling else None,
        pre_balanced=True,
    )

    # BHE pressure drop at peak flow (2 legs of the U-pipe)
    Di_bhe = float(vhe_field.pipe.outer_diameter) * (1.0 - 2.0 / float(vhe_field.pipe.sdr))
    Q_per_bh_heat = ground_loads.summed_peak_mass_flow_heating / (n_boreholes * brine.density)
    bhe_dp_heat_Pa = (
        float(_dp_per_m(brine.density, brine.dynamic_viscosity, Q_per_bh_heat, Di_bhe))
        * 2.0 * field_sizing.L_m
    )

    bhe_dp_cool_Pa: float | None = None
    if ground_loads.has_cooling:
        Q_per_bh_cool = ground_loads.summed_peak_mass_flow_cooling / (n_boreholes * brine.density)
        bhe_dp_cool_Pa = (
            float(_dp_per_m(brine.density, brine.dynamic_viscosity, Q_per_bh_cool, Di_bhe))
            * 2.0 * field_sizing.L_m
        )

    return BHEWorkflowResult(
        sizing=field_sizing,
        sys_temps=sys_temps,
        P_bhe_heating_W=P_heating,
        P_bhe_cooling_W=P_cooling,
        P_full_heating_W=P_heat_full,
        P_full_cooling_W=P_cool_full,
        dist_thermal_heat=dist_thermal["heating"],
        dist_thermal_cool=dist_thermal["cooling"] if ground_loads.has_cooling else None,
        hydraulic=hydraulic,
        brine=brine,
        bhe_dp_heat_Pa=bhe_dp_heat_Pa,
        bhe_dp_cool_Pa=bhe_dp_cool_Pa,
    )


def print_bhe_results(result: BHEWorkflowResult) -> None:
    """Print formatted dimensioning results to terminal."""
    hydraulic   = result.hydraulic
    brine       = result.brine
    sizing      = result.sizing
    t           = result.sys_temps
    has_cooling = result.P_bhe_cooling_W is not None
    network     = hydraulic.network
    n_traces    = len(hydraulic.pipe_outer_diameters)

    # ── velocities and dp/m per trace ──────────────────────────────────────
    A = math.pi * hydraulic.pipe_inner_diameters**2 / 4.0
    v_heat = hydraulic.peak_volume_flow_rate_heating / A
    dp_m_heat = np.array([
        float(_dp_per_m(brine.density, brine.dynamic_viscosity,
                        float(hydraulic.peak_volume_flow_rate_heating[i]),
                        float(hydraulic.pipe_inner_diameters[i])))
        for i in range(n_traces)
    ])

    has_cool_hydro = has_cooling and hydraulic.peak_volume_flow_rate_cooling is not None
    if has_cool_hydro:
        v_cool = hydraulic.peak_volume_flow_rate_cooling / A
        dp_m_cool = np.array([
            float(_dp_per_m(brine.density, brine.dynamic_viscosity,
                            float(hydraulic.peak_volume_flow_rate_cooling[i]),
                            float(hydraulic.pipe_inner_diameters[i])))
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
            _cell(f'{hydraulic.pipe_outer_diameters[i] * 1000:.1f}', DC),
            _cell(f'{hydraulic.pipe_inner_diameters[i] * 1000:.1f}', DC),
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

    # ── BHE sizing + pressure drop ─────────────────────────────────────────
    print()
    print(f'  Borehole length  :  {sizing.L_m:.2f} m  (governed by {sizing.governing})')
    print(f'  Borehole resist. :  {sizing.R_heating_K_m_W:.4f} K·m/W  (heating)')
    if sizing.R_cooling_K_m_W is not None:
        print(f'  Borehole resist. :  {sizing.R_cooling_K_m_W:.4f} K·m/W  (cooling)')
    print(f'  Dist. fraction   :  {result.dist_thermal_heat.F_total * 100:.1f} %  (heating, distribution grid)')
    if result.dist_thermal_cool is not None:
        print(f'  Dist. fraction   :  {result.dist_thermal_cool.F_total * 100:.1f} %  (cooling, distribution grid)')
    bhe_dp_m_heat = result.bhe_dp_heat_Pa / (2.0 * sizing.L_m)
    print(f'  BHE ΔP (heating) :  {result.bhe_dp_heat_Pa:,.0f} Pa  |  {bhe_dp_m_heat:.1f} Pa/m  (peak flow, 2 × {sizing.L_m:.2f} m U-pipe)')
    if result.bhe_dp_cool_Pa is not None:
        bhe_dp_m_cool = result.bhe_dp_cool_Pa / (2.0 * sizing.L_m)
        print(f'  BHE ΔP (cooling) :  {result.bhe_dp_cool_Pa:,.0f} Pa  |  {bhe_dp_m_cool:.1f} Pa/m  (peak flow, 2 × {sizing.L_m:.2f} m U-pipe)')

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
        t.system_mean_annual_temperature_heating, t.system_mean_winter_temperature_heating, t.system_mean_peak_temperature_heating,
    ))
    lines.append(bot_t)
    print('\n'.join(lines))
