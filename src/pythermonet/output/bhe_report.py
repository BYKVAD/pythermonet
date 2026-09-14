"""Printing formatted BHE sizing results to the terminal."""

from __future__ import annotations

import math

import numpy as np

from pythermonet.dimensioning.BHE.bhe_workflow import BHEWorkflowResult
from pythermonet.physics.hydraulics import pressure_loss_per_length as _dp_per_m


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
