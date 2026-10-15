"""HHE sizing results as text (terminal, .txt) and as JSON (.json)."""

from __future__ import annotations

import os
from pathlib import Path

from pythermonet.dimensioning.BHE.borehole_length import FieldSizingResult
from pythermonet.dimensioning.HHE.hhe_workflow import HHEWorkflowResult
from pythermonet.output._report_tables import (
    _format_header,
    _format_temperature_table,
    _format_trace_table,
)
from pythermonet.output._results_json import (
    _PULSES_TEMPERATURE,
    _check_all_placed,
    _collect_quantities,
    _distribution_network_section,
    _metadata_section,
    _save_results_json,
    _take,
)

# HHEWorkflowResult fields that are not exported quantities: nested results
# (exported through their own fields), the run time (metadata section), and
# input echoes (brine; HHE pipe values stored only for the .txt report).
_HHE_RESULT_FIELDS_NOT_QUANTITIES = frozenset({
    "sizing", "performance_thermonet_heating", "performance_thermonet_cooling",
    "hydraulic", "brine", "timestamp_run",
    "diameter_outer_hhe", "diameter_inner_hhe", "sdr_hhe",
    "n_pipes_parallel_hhe", "burial_depth_hhe",
})


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


def save_hhe_results_txt(result: HHEWorkflowResult, path: str | os.PathLike) -> None:
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


def hhe_results_to_dict(result: HHEWorkflowResult) -> dict[str, object]:
    """
    Build the structured HHE results: every computed quantity with unit and description.

    The dict has the sections ``metadata``, ``ground_field``,
    ``distribution_network`` (``performance``, ``traces``) and ``system``.
    Each quantity is ``{"value": ..., "unit": ..., "description": ...}``, keyed
    by its pythermonet attribute name; arrays (per trace, or [annual,
    winter/summer, peak]) are lists. Values that don't exist in the run (e.g.
    cooling without cooling) are None. Inputs (brine, soil, settings, HHE
    pipe geometry) are not included.

    Parameters
    ----------
    result : HHEWorkflowResult
        Output of ``run_hhe_sizing_workflow``.

    Returns
    -------
    dict
        JSON-serialisable results, as written by ``save_hhe_results_json``.

    Raises
    ------
    ValueError
        If a result field has no unit/description metadata or is not placed in
        any section.
    """
    own = _collect_quantities(
        HHEWorkflowResult, result, excluded=_HHE_RESULT_FIELDS_NOT_QUANTITIES
    )
    sizing = _collect_quantities(FieldSizingResult, result.sizing)

    results = {
        "metadata": _metadata_section("HHE", result.timestamp_run),
        "ground_field": {
            **_take(sizing, list(sizing)),
            **_take(own, [
                "loads_hhe_heating", "loads_hhe_cooling",
                "pressure_loss_hhe_heating", "pressure_loss_hhe_cooling",
                "pressure_loss_per_length_hhe_heating",
                "pressure_loss_per_length_hhe_cooling",
                "mass_flow_rate_peak_hhe_heating", "mass_flow_rate_peak_hhe_cooling",
            ]),
            **_take(own, [f"temperature_hhe_{p}" for p in _PULSES_TEMPERATURE]),
        },
        "distribution_network": _distribution_network_section(
            result.hydraulic,
            result.performance_thermonet_heating,
            result.performance_thermonet_cooling,
            extra_performance={},
        ),
        "system": {
            **_take(own, ["loads_total_heating", "loads_total_cooling"]),
            **_take(own, [f"temperature_system_{p}" for p in _PULSES_TEMPERATURE]),
        },
    }
    _check_all_placed(own=own, sizing=sizing)
    return results


def save_hhe_results_json(
    result: HHEWorkflowResult, path: str | os.PathLike
) -> None:
    """
    Write the HHE results to a UTF-8 JSON file with LF line endings.

    The file holds exactly ``hhe_results_to_dict(result)``, with each
    quantity's value, unit and description on one line. An existing file is
    overwritten; missing parent folders are not created.

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
    ValueError
        If any value is NaN (not valid JSON), or as raised by
        ``hhe_results_to_dict``.
    """
    _save_results_json(hhe_results_to_dict(result), path)
