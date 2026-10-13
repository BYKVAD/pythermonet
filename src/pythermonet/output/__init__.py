"""Writing settings files and dimensioning results (text and JSON)."""

from __future__ import annotations

from pythermonet.output.bhe_report import (
    bhe_results_to_dict,
    format_bhe_results,
    print_bhe_results,
    save_bhe_results_json,
    save_bhe_results_txt,
)
from pythermonet.output.hhe_report import (
    format_hhe_results,
    hhe_results_to_dict,
    print_hhe_results,
    save_hhe_results_json,
    save_hhe_results_txt,
)
from pythermonet.output.hydraulic_report import print_pipe_dimensioning_table
from pythermonet.output.pipe_thermal_report import print_pipe_thermal_table
from pythermonet.output.save_settings import save_settings

__all__ = [
    "bhe_results_to_dict",
    "format_bhe_results",
    "format_hhe_results",
    "hhe_results_to_dict",
    "print_bhe_results",
    "print_hhe_results",
    "print_pipe_dimensioning_table",
    "print_pipe_thermal_table",
    "save_bhe_results_json",
    "save_bhe_results_txt",
    "save_hhe_results_json",
    "save_hhe_results_txt",
    "save_settings",
]
