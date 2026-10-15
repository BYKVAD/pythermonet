"""
test_distribution_pipe_thermal_model.py
---------------------------------------
Physics tests for the distribution-pipe thermal model.

Unlike the Silkeborg regression tests, these check properties the model must
satisfy for any input: at the peak pulse the brine reaches the design mean
temperature Tm, and the load moves the brine temperature towards Tm (falling in
heating, rising in cooling).

Run with:
    python -m pytest tests/test_distribution_pipe_thermal_model.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from pythermonet.core import HeatCarrier, Soil
from pythermonet.simulation.distribution_pipe_thermal_model import (
    DistributionPipeModel,
    ModeInput,
    PipeGroup,
    ThermonetPerformance,
    compute_distribution_pipe_thermal_response,
)

_PULSE_TIMESCALES = np.array([30 * 365.25 * 86400.0, 91.5 * 86400.0, 4 * 3600.0])
_TEMPERATURE_MEAN_HEATING = -4.5
_TEMPERATURE_MEAN_COOLING = 22.0


@pytest.fixture(scope="module")
def performance() -> dict[str, ThermonetPerformance]:
    """Return heating and cooling results for a two-group pipe network."""
    brine = HeatCarrier(
        density=965, specific_heat=4450,
        thermal_conductivity=0.45, dynamic_viscosity=5e-3,
    )
    soil = Soil(
        density=2500,
        specific_heat=1000,
        thermal_conductivity=2.36,
        thermal_conductivity_shallow_heating=1.25,
        thermal_conductivity_shallow_cooling=1.25,
        geothermal_heat_flux=0.0185,
        temperature_surface_mean=9.03,
        temperature_surface_amplitude=7.9,
    )
    pipe_groups = [
        PipeGroup(
            id_=1, length=120.0, diameter_inner=0.0900, diameter_outer=0.110,
            reynolds_number=8000.0, thermal_conductivity_pipe=0.4,
            burial_depth=1.2, n_parallel_pipes=2, n_traces=1, pipe_spacing=0.3,
        ),
        PipeGroup(
            id_=2, length=60.0, diameter_inner=0.0514, diameter_outer=0.063,
            reynolds_number=5000.0, thermal_conductivity_pipe=0.4,
            burial_depth=1.2, n_parallel_pipes=2, n_traces=3, pipe_spacing=0.3,
        ),
    ]
    model = DistributionPipeModel(
        brine=brine,
        soil=soil,
        temperature_ground_undisturbed=9.03,
        temperature_surface_amplitude=7.9,
        pipe_groups=pipe_groups,
        heating=ModeInput(
            pulse_timescales=_PULSE_TIMESCALES,
            pulse_loads=np.array([2000.0, 5000.0, 9000.0]),
            temperature_heat_pump_inlet=-3.0,
            temperature_heat_pump_outlet=-6.0,
        ),
        cooling=ModeInput(
            pulse_timescales=_PULSE_TIMESCALES,
            pulse_loads=np.array([500.0, 2500.0, 6000.0]),
            temperature_heat_pump_inlet=20.0,
            temperature_heat_pump_outlet=24.0,
        ),
    )
    return compute_distribution_pipe_thermal_response(model)


@pytest.mark.parametrize(
    "mode, temperature_mean_design",
    [("heating", _TEMPERATURE_MEAN_HEATING), ("cooling", _TEMPERATURE_MEAN_COOLING)],
)
def test_peak_temperature_equals_design_mean(
    performance, mode, temperature_mean_design
):
    temperature_peak = performance[mode].temperatures_mean[-1]
    assert temperature_peak == pytest.approx(temperature_mean_design)


def test_heating_temperatures_fall_towards_peak(performance):
    temperatures = performance["heating"].temperatures_mean
    assert np.all(np.diff(temperatures) < 0), temperatures


def test_cooling_temperatures_rise_towards_peak(performance):
    temperatures = performance["cooling"].temperatures_mean
    assert np.all(np.diff(temperatures) > 0), temperatures
