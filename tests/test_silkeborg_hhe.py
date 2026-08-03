"""
test_silkeborg_hhe.py
---------------------
Regression tests for the Silkeborg HHE full-dimensioning example
(heat only, pre-dimensioned topology, aggregated load).

Reference values were captured from:
  examples/silkeborg_hhe_full_dimensioning_heat/main.py

Run with:
    python -m pytest tests/test_silkeborg_hhe.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path

from pythermonet.components.ground_loads import ground_loads_from_district
from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.ground_field import HHEGroundField
from pythermonet.dimensioning.HHE.hhe_workflow import run_hhe_sizing_workflow
from pythermonet.dimensioning.sizing_parameters import SizingParameters
from pythermonet.input.read_aggregated_load import read_aggregated_load_tsv
from pythermonet.input.read_dimensioned_topology import (
    read_dimensioned_topology_tsv_to_hydraulic,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "silkeborg_hhe_full_dimensioning_heat"
)

# ---------------------------------------------------------------------------
# Module-scoped fixture — runs the full dimensioning once for all tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hhe_dimensioning():
    """Return result from the Silkeborg HHE full-dimensioning run."""
    pipe_material_dist = Material(
        density=975, specific_heat=1900, thermal_conductivity=0.4
    )

    brine = HeatCarrier(
        density=965, specific_heat=4450,
        thermal_conductivity=0.45, dynamic_viscosity=5e-3,
    )

    soil = Soil(
        density=2500,
        specific_heat=1000,
        thermal_conductivity=1.25,
        thermal_conductivity_shallow_heating=1.25,
        thermal_conductivity_shallow_cooling=1.25,
        geothermal_heat_flux=0.0185,
        temperature_surface_mean=9.03,
        temperature_surface_amplitude=7.9,
    )

    _, hydraulic = read_dimensioned_topology_tsv_to_hydraulic(
        _EXAMPLE_DIR / "data" / "silkeborg_hhe_topology_dimensioned.dat",
        pipe_material=pipe_material_dist,
        brine=brine,
        burial_depth=1.2,
        pipe_distance=0.3,
        n_parallel_pipes=2,
    )

    hhe_segment = PipeSegment(
        diameter_outer=0.040, sdr=17.0, material=pipe_material_dist,
        roughness=1e-6, id_=0, length=100.0,
    )
    pipe_infrastructure = PipeInfrastructure(
        n_pipes_parallel=20,
        segments_trace=[hhe_segment],
        pipe_spacing=1.5,
        burial_depth=1.2,
    )
    hhe_field = HHEGroundField(
        pipe_infrastructure=pipe_infrastructure,
        soil_thermal_conductivity_heating=float(
            soil.thermal_conductivity_shallow_heating
        ),
        soil_thermal_conductivity_cooling=float(
            soil.thermal_conductivity_shallow_cooling
        ),
    )

    agg_load_input = read_aggregated_load_tsv(
        _EXAMPLE_DIR / "data" / "silkeborg_hhe_aggregated_load_heat.dat"
    )
    loads = ground_loads_from_district(
        agg_load_input,
        brine,
        f_peak_heating=1.0,
        f_peak_cooling=1.0,
        peak_hours_heating=4.0,
        peak_hours_cooling=4.0,
    )

    sizing = SizingParameters(thermal_dimensioning_lifetime=30.0)

    result = run_hhe_sizing_workflow(
        ground_loads=loads,
        hhe_field=hhe_field,
        pipe_infrastructure=pipe_infrastructure,
        hydraulic=hydraulic,
        brine=brine,
        soil=soil,
        sizing=sizing,
        T_brine_min_heat=-3.0,
        T_brine_max_cool=20.0,
    )

    return result


# ===========================================================================
# HHE thermal dimensioning tests
# ===========================================================================

class TestHHEThermalDimensioning:

    def test_loop_length(self, hhe_dimensioning):
        # sizing.L_m is the one-way pipe segment length;
        # the printed "loop length" is 2 × L_m = 220.98 m.
        result = hhe_dimensioning
        L = result.sizing.length_element
        assert abs(L - 110.49) < 0.5, (
            f"HHE segment length {L:.2f} m deviates >0.5 m from 110.49 m"
        )

    def test_governing_mode_heating(self, hhe_dimensioning):
        result = hhe_dimensioning
        assert result.sizing.governing_mode == "heating", (
            f"Expected governing='heating', got '{result.sizing.governing_mode}'"
        )

    def test_distribution_fraction_heating(self, hhe_dimensioning):
        result = hhe_dimensioning
        f = result.performance_thermonet_heating.load_supply_fraction
        assert abs(f * 100 - 35.2) < 0.5, (
            f"Dist. fraction (heating) = {f*100:.1f}%"
            " deviates >0.5% from 35.2%"
        )

    def test_hhe_pressure_drop_heating(self, hhe_dimensioning):
        result = hhe_dimensioning
        dp = result.pressure_loss_hhe_heating
        assert abs(dp - 9_490) < 200, (
            f"HHE ΔP (heating) = {dp:.0f} Pa deviates >200 Pa from 9,490 Pa"
        )

    def test_system_temperature_heating_annual(self, hhe_dimensioning):
        result = hhe_dimensioning
        T = result.temperature_system_annual_heating
        assert abs(T - 0.22) < 0.05, (
            f"T_avg_heat_annual = {T:.2f}°C deviates >0.05°C from 0.22°C"
        )

    def test_system_temperature_heating_winter(self, hhe_dimensioning):
        result = hhe_dimensioning
        T = result.temperature_system_winter_heating
        assert abs(T - (-2.14)) < 0.05, (
            f"T_avg_heat_winter = {T:.2f}°C deviates >0.05°C from -2.14°C"
        )

    def test_system_temperature_heating_peak(self, hhe_dimensioning):
        result = hhe_dimensioning
        T = result.temperature_system_peak_heating
        assert abs(T - (-4.50)) < 0.05, (
            f"T_avg_heat_peak = {T:.2f}°C deviates >0.05°C from -4.50°C"
        )

    def test_no_cooling_results(self, hhe_dimensioning):
        """Heat-only case should have no cooling sizing results."""
        result = hhe_dimensioning
        assert result.sizing.thermal_resistance_cooling is None
        assert result.pressure_loss_hhe_cooling is None
        assert result.temperature_system_annual_cooling is None
