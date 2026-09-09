"""
test_silkeborg_hhe.py
----------------------
Regression tests for the Silkeborg HHE full-dimensioning example
(heat + cooling, undimensioned topology).

Reference values were captured from:
  examples/silkeborg_hhe_full_dimensioning_heat/main.py

Run with:
    python -m pytest tests/test_silkeborg_hhe.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import EXAMPLES_DIR
from pythermonet.components.ground_loads import ground_loads_from_heat_pumps
from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.ground_field import HHEGroundField
from pythermonet.dimensioning.HHE.hhe_workflow import run_hhe_sizing_workflow
from pythermonet.dimensioning.hydraulic_dimensioning import run_pipedimensioning
from pythermonet.dimensioning.sizing_parameters import SizingParameters
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv
from pythermonet.input.read_pipe_catalog import read_pipe_catalog
from pythermonet.input.read_topology import read_undimensioned_topology_tsv_to_network

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EXAMPLE_HHE_DIR = EXAMPLES_DIR / "silkeborg_hhe_full_dimensioning_heat"

# ---------------------------------------------------------------------------
# Module-scoped fixture — runs the full dimensioning once for all tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hhe_dimensioning():
    """Return (hydraulic, result) from the Silkeborg HHE full-dimensioning run."""
    pipe_catalog = read_pipe_catalog()

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

    network = read_undimensioned_topology_tsv_to_network(
        _EXAMPLE_HHE_DIR / "data" / "silkeborg_topology.dat",
        pipe_material=pipe_material_dist,
        roughness=1e-6,
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

    hp_list = read_heat_pumps_tsv(
        path=_EXAMPLE_HHE_DIR / "data" / "silkeborg_heat_pump_heat_only.dat"
    )
    loads = ground_loads_from_heat_pumps(
        hp_list,
        brine,
        peak_hours_heating=4.0,
        peak_fraction_heating_mode="incremental",
        peak_fraction_heating=1.0,
        peak_hours_cooling=4.0,
        peak_fraction_cooling_mode="incremental",
        peak_fraction_cooling=1.0,
    )

    sizing = SizingParameters(thermal_dimensioning_lifetime=30.0)

    hydraulic = run_pipedimensioning(pipe_catalog, brine, network, hp_list)

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

    return hydraulic, result


# ---------------------------------------------------------------------------
# Reference values (from example output)
# ---------------------------------------------------------------------------

# Trace order: Main_line, Inner_distribution_ring_1, Inner_distribution_ring_2,
#              Single_branch, Connection_pipes (x2)

_REF_OD_MM   = [90.0, 75.0, 63.0, 63.0, 40.0, 50.0]
_REF_RE_HEAT = [9714, 7586, 5413, 3926, 2162, 2882]
_REF_RE_COOL = [7254, 5917, 3726, 3062, 1686, 1349]


# ===========================================================================
# Hydraulic dimensioning tests
# ===========================================================================

class TestHydraulicDimensioning:

    def test_diameters_outer(self, hhe_dimensioning):
        hydraulic, _ = hhe_dimensioning
        od_mm = hydraulic.diameters_outer * 1e3
        np.testing.assert_allclose(
            od_mm, _REF_OD_MM, atol=0.1,
            err_msg="Pipe OD selection differs from reference",
        )

    def test_reynolds_heating(self, hhe_dimensioning):
        hydraulic, _ = hhe_dimensioning
        np.testing.assert_allclose(
            hydraulic.reynolds_numbers_heating, _REF_RE_HEAT, rtol=0.01,
            err_msg="Heating Reynolds numbers differ from reference",
        )

    def test_reynolds_cooling(self, hhe_dimensioning):
        hydraulic, _ = hhe_dimensioning
        np.testing.assert_allclose(
            hydraulic.reynolds_numbers_cooling, _REF_RE_COOL, rtol=0.01,
            err_msg="Cooling Reynolds numbers differ from reference",
        )


# ===========================================================================
# HHE thermal dimensioning tests
# ===========================================================================

class TestHHEThermalDimensioning:

    def test_loop_length(self, hhe_dimensioning):
        # sizing.length_element is the one-way pipe segment length;
        # the printed "loop length" is 2 × length_element = 221.81 m.
        _, result = hhe_dimensioning
        L = result.sizing.length_element
        assert abs(L - 110.90) < 0.5, (
            f"HHE segment length {L:.2f} m deviates >0.5 m from 110.90 m"
        )

    def test_governing_mode_cooling(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        assert result.sizing.governing_mode == "cooling", (
            f"Expected governing='cooling', got '{result.sizing.governing_mode}'"
        )

    def test_distribution_fraction_heating(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        f = result.performance_thermonet_heating.load_supply_fraction
        assert abs(f * 100 - 35.5) < 0.5, (
            f"Dist. fraction (heating) = {f*100:.1f}%"
            " deviates >0.5% from 35.5%"
        )

    def test_distribution_fraction_cooling(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        f = result.performance_thermonet_cooling.load_supply_fraction
        assert abs(f * 100 - 33.2) < 0.5, (
            f"Dist. fraction (cooling) = {f*100:.1f}%"
            " deviates >0.5% from 33.2%"
        )

    def test_hhe_pressure_drop_heating(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        dp = result.pressure_loss_hhe_heating
        assert abs(dp - 9_526) < 200, (
            f"HHE ΔP (heating) = {dp:.0f} Pa deviates >200 Pa from 9,526 Pa"
        )

    def test_hhe_pressure_drop_cooling(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        dp = result.pressure_loss_hhe_cooling
        assert abs(dp - 6_602) < 200, (
            f"HHE ΔP (cooling) = {dp:.0f} Pa deviates >200 Pa from 6,602 Pa"
        )

    def test_system_temperature_heating_annual(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        T = result.temperature_system_annual_heating
        assert abs(T - 1.98) < 0.05, (
            f"T_avg_heat_annual = {T:.2f}°C deviates >0.05°C from 1.98°C"
        )

    def test_system_temperature_heating_winter(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        T = result.temperature_system_winter_heating
        assert abs(T - (-1.99)) < 0.05, (
            f"T_avg_heat_winter = {T:.2f}°C deviates >0.05°C from -1.99°C"
        )

    def test_system_temperature_heating_peak(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        T = result.temperature_system_peak_heating
        assert abs(T - (-4.34)) < 0.05, (
            f"T_avg_heat_peak = {T:.2f}°C deviates >0.05°C from -4.34°C"
        )

    def test_system_temperature_cooling_annual(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        T = result.temperature_system_annual_cooling
        assert abs(T - 13.54) < 0.05, (
            f"T_avg_cool_annual = {T:.2f}°C deviates >0.05°C from 13.54°C"
        )

    def test_system_temperature_cooling_summer(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        T = result.temperature_system_summer_cooling
        assert abs(T - 13.55) < 0.05, (
            f"T_avg_cool_summer = {T:.2f}°C deviates >0.05°C from 13.55°C"
        )

    def test_system_temperature_cooling_peak(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        T = result.temperature_system_peak_cooling
        assert abs(T - 13.94) < 0.05, (
            f"T_avg_cool_peak = {T:.2f}°C deviates >0.05°C from 13.94°C"
        )
