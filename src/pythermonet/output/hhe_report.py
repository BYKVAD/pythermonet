"""Formatting HHE sizing results as text for the terminal and .txt files."""

from __future__ import annotations

import os
from pathlib import Path

from pythermonet.dimensioning.HHE.hhe_workflow import HHEWorkflowResult
from pythermonet.output._report_tables import (
    _format_header,
    _format_temperature_table,
    _format_trace_table,
)


def format_hhe_results(result: HHEWorkflowResult) -> str:
    """
    Format HHE dimensioning results as a text report.

    The report contains a header (mode, pythermonet version, run time), the
    distribution network table, the HHE sizing summary and the system mean
    brine temperatures. The same result always gives the same text.

    Parameters
    ----------
    result : HHEWorkflowResult
        Output of ``run_hhe_sizing_workflow``.

    Returns
    -------
    str
        Report text with LF line endings, ending with a single newline.
    """
    sizing = result.sizing
    has_cooling = result.loads_hhe_cooling is not None

    lines = [f'  HHE loop length  :  {2.0 * sizing.length_element:.2f} m  (governed by {sizing.governing_mode})']
    lines.append(f'  Dist. fraction   :  {result.performance_thermonet_heating.load_supply_fraction * 100:.1f} %  (heating, distribution grid)')
    if result.performance_thermonet_cooling is not None:
        lines.append(f'  Dist. fraction   :  {result.performance_thermonet_cooling.load_supply_fraction * 100:.1f} %  (cooling, distribution grid)')
    lines.append(f'  HHE Do / Di      :  {result.diameter_outer_hhe*1000:.1f} mm / {result.diameter_inner_hhe*1000:.1f} mm  '
                 f'(sdr {result.sdr_hhe:.0f})')
    lines.append(f'  N parallel pipes :  {result.n_pipes_parallel_hhe}')
    lines.append(f'  Burial depth     :  {result.burial_depth_hhe:.2f} m')
    lines.append(f'  HHE ΔP (heating) :  {result.pressure_loss_hhe_heating:,.0f} Pa  |  {result.pressure_loss_per_length_hhe_heating:.1f} Pa/m')
    if result.pressure_loss_hhe_cooling is not None:
        lines.append(f'  HHE ΔP (cooling) :  {result.pressure_loss_hhe_cooling:,.0f} Pa  |  {result.pressure_loss_per_length_hhe_cooling:.1f} Pa/m')

    sections = [
        _format_header('HHE', result.timestamp_run),
        _format_trace_table(result.hydraulic, has_cooling),
        '\n'.join(lines),
        _format_temperature_table(
            result.temperature_system_annual_heating,
            result.temperature_system_winter_heating,
            result.temperature_system_peak_heating,
        ),
    ]
    return '\n\n'.join(sections) + '\n'


def print_hhe_results(result: HHEWorkflowResult) -> None:
    """
    Print HHE dimensioning results to the terminal.

    Parameters
    ----------
    result : HHEWorkflowResult
        Output of ``run_hhe_sizing_workflow``.
    """
    print('\n' + format_hhe_results(result), end='')


def write_hhe_results_txt(result: HHEWorkflowResult, path: str | os.PathLike) -> None:
    """
    Write the HHE results report to a UTF-8 text file with LF line endings.

    The file holds exactly the text returned by ``format_hhe_results``. An
    existing file is overwritten; missing parent folders are not created.

    Parameters
    ----------
    result : HHEWorkflowResult
        Output of ``run_hhe_sizing_workflow``.
    path : str or os.PathLike
        Destination file path.

    Raises
    ------
    FileNotFoundError
        If the parent folder of ``path`` does not exist.
    """
    Path(path).write_bytes(format_hhe_results(result).encode('utf-8'))
