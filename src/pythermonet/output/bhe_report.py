"""BHE sizing results as text (terminal, .txt) and as JSON (.json)."""

from __future__ import annotations

import os
from pathlib import Path

from pythermonet.dimensioning.BHE.bhe_workflow import BHEWorkflowResult
from pythermonet.dimensioning.BHE.borehole_length import FieldSizingResult
from pythermonet.dimensioning.system_temperatures import SystemBrineTemperatureResult
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

# BHEWorkflowResult fields that are not quantities of their own: nested results
# (exported through their own fields), an input echo (brine) and the run time
# (written to the metadata section).
_BHE_RESULT_FIELDS_NOT_QUANTITIES = frozenset({
    "sizing", "temperatures_system", "performance_thermonet_heating",
    "performance_thermonet_cooling", "hydraulic", "brine", "timestamp_run",
})

# Exact copies of ThermonetPerformance.temperatures_mean, which is exported instead.
_TEMPERATURE_THERMONET_DUPLICATES = frozenset(
    f"temperature_thermonet_{pulse}" for pulse in _PULSES_TEMPERATURE
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


def bhe_results_to_dict(result: BHEWorkflowResult) -> dict[str, object]:
    """
    Build the structured BHE results: every computed quantity with unit and description.

    The dict has the sections ``metadata``, ``ground_field``,
    ``distribution_network`` (``performance``, ``traces``) and ``system``.
    Each quantity is ``{"value": ..., "unit": ..., "description": ...}``, keyed
    by its pythermonet attribute name; arrays (per trace, or [annual,
    winter/summer, peak]) are lists. Values that don't exist in the run (e.g.
    cooling without cooling) are None. Inputs (brine, soil, settings) are not
    included.

    Parameters
    ----------
    result : BHEWorkflowResult
        Output of ``run_bhe_sizing_workflow``.

    Returns
    -------
    dict
        JSON-serialisable results, as written by ``save_bhe_results_json``.

    Raises
    ------
    ValueError
        If a result field has no unit/description metadata or is not placed in
        any section.
    """
    own = _collect_quantities(
        BHEWorkflowResult, result, excluded=_BHE_RESULT_FIELDS_NOT_QUANTITIES
    )
    sizing = _collect_quantities(FieldSizingResult, result.sizing)
    temperatures = _collect_quantities(
        SystemBrineTemperatureResult,
        result.temperatures_system,
        excluded=_TEMPERATURE_THERMONET_DUPLICATES,
    )

    results = {
        "metadata": _metadata_section("BHE", result.timestamp_run),
        "ground_field": {
            **_take(sizing, list(sizing)),
            **_take(own, [
                "loads_bhe_heating", "loads_bhe_cooling",
                "pressure_loss_bhe_heating", "pressure_loss_bhe_cooling",
                "pressure_loss_per_length_bhe_heating",
                "pressure_loss_per_length_bhe_cooling",
                "mass_flow_rate_peak_bhe_heating", "mass_flow_rate_peak_bhe_cooling",
            ]),
            **_take(temperatures, [
                f"temperature_bhe_{p}" for p in _PULSES_TEMPERATURE
            ]),
            **_take(temperatures, ["volume_brine_bhe"]),
        },
        "distribution_network": _distribution_network_section(
            result.hydraulic,
            result.performance_thermonet_heating,
            result.performance_thermonet_cooling,
            extra_performance=_take(temperatures, ["volume_brine_thermonet"]),
        ),
        "system": {
            **_take(own, ["loads_total_heating", "loads_total_cooling"]),
            **_take(temperatures, [
                f"temperature_system_mean_{p}" for p in _PULSES_TEMPERATURE
            ]),
            **_take(temperatures, ["volume_brine_total", "volume_fraction_bhe"]),
        },
    }
    _check_all_placed(own=own, sizing=sizing, temperatures=temperatures)
    return results


def save_bhe_results_json(
    result: BHEWorkflowResult, path: str | os.PathLike
) -> None:
    """
    Write the BHE results to a UTF-8 JSON file with LF line endings.

    The file holds exactly ``bhe_results_to_dict(result)``, with each
    quantity's value, unit and description on one line. An existing file is
    overwritten; missing parent folders are not created.

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
    ValueError
        If any value is NaN (not valid JSON), or as raised by
        ``bhe_results_to_dict``.
    """
    _save_results_json(bhe_results_to_dict(result), path)
