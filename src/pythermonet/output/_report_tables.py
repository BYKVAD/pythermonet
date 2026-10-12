"""Report sections shared by the BHE and HHE results reports."""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import version

from pythermonet.dimensioning.hydraulic_result import HydraulicResult


def _format_header(mode: str, timestamp_run: datetime) -> str:
    """Identify the report: mode, pythermonet version and run time with offset."""
    if timestamp_run.tzinfo is None:
        raise ValueError("timestamp_run must be timezone-aware to print its offset.")
    offset = timestamp_run.strftime('%z')
    computed = f'{timestamp_run:%Y-%m-%d %H:%M:%S} {offset[:3]}:{offset[3:]}'
    lines = [
        f'  Mode             :  {mode}',
        f'  pythermonet      :  {version("pythermonet")}',
        f'  Computed         :  {computed}',
    ]
    return '\n'.join(lines)


def _format_trace_table(hydraulic: HydraulicResult, has_cooling: bool) -> str:
    """Distribution network table: diameters, velocity, Re and dp/m per trace."""
    network = hydraulic.network
    n_traces = len(hydraulic.diameters_outer)
    has_cool_hydro = has_cooling and hydraulic.velocities_cooling is not None

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
            _cell(f'{hydraulic.velocities_heating[i]:.3f}', VC),
            _cell(f'{hydraulic.reynolds_numbers_heating[i]:,.0f}', RC),
            _cell(f'{hydraulic.pressure_losses_per_length_heating[i]:.1f}', PC),
        ]
        if has_cool_hydro:
            cells += [
                _cell(f'{hydraulic.velocities_cooling[i]:.3f}', VC),
                _cell(f'{hydraulic.reynolds_numbers_cooling[i]:,.0f}', RC),
                _cell(f'{hydraulic.pressure_losses_per_length_cooling[i]:.1f}', PC),
            ]
        return '│' + '│'.join(cells) + '│'

    lines = [
        '  Distribution network',
        _border('┌', '┬', '┐'),
        _hrow(),
        _border('├', '┼', '┤'),
    ]
    lines += [_drow(i) for i in range(n_traces)]
    lines.append(_border('└', '┴', '┘'))
    return '\n'.join(lines)


def _format_temperature_table(
    temperature_annual: float,
    temperature_winter: float,
    temperature_peak: float,
) -> str:
    """System mean brine temperature table (heating) [°C]."""
    LW, VW = 36, 14

    top_t = f"┌{'─'*LW}┬{'─'*VW}┬{'─'*VW}┬{'─'*VW}┐"
    div_t = f"├{'─'*LW}┼{'─'*VW}┼{'─'*VW}┼{'─'*VW}┤"
    bot_t = f"└{'─'*LW}┴{'─'*VW}┴{'─'*VW}┴{'─'*VW}┘"

    def hdr() -> str:
        return f"│{'':{LW}}│{'Annual':^{VW}}│{'Winter':^{VW}}│{'Peak':^{VW}}│"

    def trow(label: str, a: float, w: float, p: float) -> str:
        fa, fw, fp = f'{a:.2f}', f'{w:.2f}', f'{p:.2f}'
        return f"│ {label:<{LW-2}} │{fa:>{VW-1}} │{fw:>{VW-1}} │{fp:>{VW-1}} │"

    lines = [
        top_t,
        hdr(),
        div_t,
        trow(
            'System mean temp (heating)   [°C]',
            temperature_annual, temperature_winter, temperature_peak,
        ),
        bot_t,
    ]
    return '\n'.join(lines)
