"""Formatting BHE sizing results as text for the terminal and .txt files."""

from __future__ import annotations

import os
from pathlib import Path

from pythermonet.dimensioning.BHE.bhe_workflow import BHEWorkflowResult
from pythermonet.output._report_tables import (
    _format_header,
    _format_temperature_table,
    _format_trace_table,
)


def format_bhe_results(result: BHEWorkflowResult) -> str:
    """
    Format BHE dimensioning results as a text report.

    The report contains a header (mode, pythermonet version, run time), the
    distribution network table, the BHE sizing summary and the system mean
    brine temperatures. The same result always gives the same text.

    Parameters
    ----------
    result : BHEWorkflowResult
        Output of ``run_bhe_sizing_workflow``.

    Returns
    -------
    str
        Report text with LF line endings, ending with a single newline.
    """
    sizing = result.sizing
    t = result.temperatures_system
    has_cooling = result.loads_bhe_cooling is not None

    lines = [
        f'  Borehole length  :  {sizing.length_element:.2f} m  (governed by {sizing.governing_mode})',
        f'  Borehole resist. :  {sizing.thermal_resistance_heating:.4f} K·m/W  (heating)',
    ]
    if sizing.thermal_resistance_cooling is not None:
        lines.append(f'  Borehole resist. :  {sizing.thermal_resistance_cooling:.4f} K·m/W  (cooling)')
    lines.append(f'  Dist. fraction   :  {result.performance_thermonet_heating.load_supply_fraction * 100:.1f} %  (heating, distribution grid)')
    if result.performance_thermonet_cooling is not None:
        lines.append(f'  Dist. fraction   :  {result.performance_thermonet_cooling.load_supply_fraction * 100:.1f} %  (cooling, distribution grid)')
    lines.append(f'  BHE ΔP (heating) :  {result.pressure_loss_bhe_heating:,.0f} Pa  |  {result.pressure_loss_per_length_bhe_heating:.1f} Pa/m  (peak flow, 2 × {sizing.length_element:.2f} m U-pipe)')
    if result.pressure_loss_bhe_cooling is not None:
        lines.append(f'  BHE ΔP (cooling) :  {result.pressure_loss_bhe_cooling:,.0f} Pa  |  {result.pressure_loss_per_length_bhe_cooling:.1f} Pa/m  (peak flow, 2 × {sizing.length_element:.2f} m U-pipe)')

    sections = [
        _format_header('BHE', result.timestamp_run),
        _format_trace_table(result.hydraulic, has_cooling),
        '\n'.join(lines),
        _format_temperature_table(
            t.temperature_system_mean_annual_heating,
            t.temperature_system_mean_winter_heating,
            t.temperature_system_mean_peak_heating,
        ),
    ]
    return '\n\n'.join(sections) + '\n'


def print_bhe_results(result: BHEWorkflowResult) -> None:
    """
    Print BHE dimensioning results to the terminal.

    Parameters
    ----------
    result : BHEWorkflowResult
        Output of ``run_bhe_sizing_workflow``.
    """
    print('\n' + format_bhe_results(result), end='')


def save_bhe_results_txt(result: BHEWorkflowResult, path: str | os.PathLike) -> None:
    """
    Write the BHE results report to a UTF-8 text file with LF line endings.

    The file holds exactly the text returned by ``format_bhe_results``. An
    existing file is overwritten; missing parent folders are not created.

    Parameters
    ----------
    result : BHEWorkflowResult
        Output of ``run_bhe_sizing_workflow``.
    path : str or os.PathLike
        Destination file path.

    Raises
    ------
    FileNotFoundError
        If the parent folder of ``path`` does not exist.
    """
    Path(path).write_bytes(format_bhe_results(result).encode('utf-8'))
