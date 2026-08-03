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
from pythermonet.simulation.distribution_pipe_thermal_model import ThermonetPerformance
from pythermonet.simulation.run_distribution_pipe_thermal_model import (
    compute_distribution_pipe_thermal_capacity,
)


@dataclass(frozen=True)
class BHEWorkflowResult:
    """
    Output of the BHE sizing workflow.

    Load arrays are ordered [annual, winter, peak] in watts.
    ``loads_bhe_heating`` is the BHE portion after subtracting the
    thermonet distribution pipe contribution from ``loads_total_heating``.
    """

    sizing: FieldSizingResult
    temperatures_system: SystemBrineTemperatureResult
    loads_bhe_heating: np.ndarray                       # [W]
    loads_bhe_cooling: np.ndarray | None                # [W]
    loads_total_heating: np.ndarray                     # [W]
    loads_total_cooling: np.ndarray | None              # [W]
    performance_thermonet_heating: ThermonetPerformance
    performance_thermonet_cooling: ThermonetPerformance | None
    hydraulic: HydraulicResult
    brine: HeatCarrier
    pressure_loss_bhe_heating: float                    # [Pa]
    pressure_loss_bhe_cooling: float | None             # [Pa]


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
    P_heat_full = np.asarray(ground_loads.loads_ground_heating, dtype=float)
    P_cool_full = (
        np.asarray(ground_loads.loads_ground_cooling, dtype=float)
        if ground_loads.has_cooling else None
    )
    if P_cool_full is not None:
        P_heat_full, P_cool_full = apply_annual_balance(P_heat_full.copy(), P_cool_full.copy())

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

    # BHE loads = full balanced loads × (1 − distribution fraction)
    P_heating = (1 - dist_thermal["heating"].load_supply_fraction) * P_heat_full
    P_cooling = (
        (1 - dist_thermal["cooling"].load_supply_fraction) * P_cool_full
        if ground_loads.has_cooling else None
    )

    n_boreholes = vhe_field.n_boreholes

    # Size borehole length
    field_sizing = size_borehole_length_heating_cooling(
        T_fluid_min=T_brine_min_heat - 0.5 * ground_loads.temperature_delta_brine_flow_weighted_heating,
        P_heating_W=P_heating,
        times_heat_s=times_heat_s,
        T_fluid_max=T_brine_max_cool + (
            0.5 * ground_loads.temperature_delta_brine_flow_weighted_cooling if ground_loads.has_cooling else 0.0
        ),
        P_cooling_W=P_cooling,
        times_cool_s=times_cool_s,
        vhe_field=vhe_field,
        brine=brine,
        soil=soil,
        m_dot_per_borehole_heat_kg_s=ground_loads.mass_flow_peak_summed_heating / n_boreholes,
        m_dot_per_borehole_cool_kg_s=(
            ground_loads.mass_flow_peak_summed_cooling / n_boreholes
            if ground_loads.has_cooling else None
        ),
        pre_balanced=True,
    )

    # System brine temperatures
    temperatures_system = compute_system_brine_temperatures(
        sizing=field_sizing,
        vhe_field=vhe_field,
        hydraulic=hydraulic,
        P_heating_W=P_heating,
        times_heat_s=times_heat_s,
        P_cooling_W=P_cooling,
        times_cool_s=times_cool_s,
        soil=soil,
        performance_thermonet_heating=dist_thermal["heating"],
        performance_thermonet_cooling=dist_thermal["cooling"] if ground_loads.has_cooling else None,
        pre_balanced=True,
    )

    # BHE pressure drop at peak flow (2 legs of the U-pipe)
    Di_bhe = float(vhe_field.pipe.diameter_outer) * (1.0 - 2.0 / float(vhe_field.pipe.sdr))
    Q_per_bh_heat = ground_loads.mass_flow_peak_summed_heating / (n_boreholes * brine.density)
    pressure_loss_bhe_heating = (
        float(_dp_per_m(brine.density, brine.dynamic_viscosity, Q_per_bh_heat, Di_bhe))
        * 2.0 * field_sizing.length_element
    )

    pressure_loss_bhe_cooling: float | None = None
    if ground_loads.has_cooling:
        Q_per_bh_cool = ground_loads.mass_flow_peak_summed_cooling / (n_boreholes * brine.density)
        pressure_loss_bhe_cooling = (
            float(_dp_per_m(brine.density, brine.dynamic_viscosity, Q_per_bh_cool, Di_bhe))
            * 2.0 * field_sizing.length_element
        )

    return BHEWorkflowResult(
        sizing=field_sizing,
        temperatures_system=temperatures_system,
        loads_bhe_heating=P_heating,
        loads_bhe_cooling=P_cooling,
        loads_total_heating=P_heat_full,
        loads_total_cooling=P_cool_full,
        performance_thermonet_heating=dist_thermal["heating"],
        performance_thermonet_cooling=dist_thermal["cooling"] if ground_loads.has_cooling else None,
        hydraulic=hydraulic,
        brine=brine,
        pressure_loss_bhe_heating=pressure_loss_bhe_heating,
        pressure_loss_bhe_cooling=pressure_loss_bhe_cooling,
    )


def print_bhe_results(result: BHEWorkflowResult) -> None:
    """Print formatted dimensioning results to terminal."""
    hydraulic   = result.hydraulic
    brine       = result.brine
    sizing      = result.sizing
    t           = result.temperatures_system
    has_cooling = result.loads_bhe_cooling is not None
    network     = hydraulic.network
    n_traces    = len(hydraulic.diameters_outer)

    # ── velocities and dp/m per trace ──────────────────────────────────────
    A = math.pi * hydraulic.diameters_inner**2 / 4.0
    v_heat = hydraulic.volume_flow_rates_peak_heating / A
    dp_m_heat = np.array([
        float(_dp_per_m(brine.density, brine.dynamic_viscosity,
                        float(hydraulic.volume_flow_rates_peak_heating[i]),
                        float(hydraulic.diameters_inner[i])))
        for i in range(n_traces)
    ])

    has_cool_hydro = has_cooling and hydraulic.volume_flow_rates_peak_cooling is not None
    if has_cool_hydro:
        v_cool = hydraulic.volume_flow_rates_peak_cooling / A
        dp_m_cool = np.array([
            float(_dp_per_m(brine.density, brine.dynamic_viscosity,
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

    # ── BHE sizing + pressure drop ─────────────────────────────────────────
    print()
    print(f'  Borehole length  :  {sizing.length_element:.2f} m  (governed by {sizing.governing_mode})')
    print(f'  Borehole resist. :  {sizing.thermal_resistance_heating:.4f} K·m/W  (heating)')
    if sizing.thermal_resistance_cooling is not None:
        print(f'  Borehole resist. :  {sizing.thermal_resistance_cooling:.4f} K·m/W  (cooling)')
    print(f'  Dist. fraction   :  {result.performance_thermonet_heating.load_supply_fraction * 100:.1f} %  (heating, distribution grid)')
    if result.performance_thermonet_cooling is not None:
        print(f'  Dist. fraction   :  {result.performance_thermonet_cooling.load_supply_fraction * 100:.1f} %  (cooling, distribution grid)')
    bhe_dp_m_heat = result.pressure_loss_bhe_heating / (2.0 * sizing.length_element)
    print(f'  BHE ΔP (heating) :  {result.pressure_loss_bhe_heating:,.0f} Pa  |  {bhe_dp_m_heat:.1f} Pa/m  (peak flow, 2 × {sizing.length_element:.2f} m U-pipe)')
    if result.pressure_loss_bhe_cooling is not None:
        bhe_dp_m_cool = result.pressure_loss_bhe_cooling / (2.0 * sizing.length_element)
        print(f'  BHE ΔP (cooling) :  {result.pressure_loss_bhe_cooling:,.0f} Pa  |  {bhe_dp_m_cool:.1f} Pa/m  (peak flow, 2 × {sizing.length_element:.2f} m U-pipe)')

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
        t.temperature_system_mean_annual_heating, t.temperature_system_mean_winter_heating, t.temperature_system_mean_peak_heating,
    ))
    lines.append(bot_t)
    print('\n'.join(lines))
