"""
test_silkeborg_bhe.py
---------------------
Regression tests for the Silkeborg BHE full-dimensioning example
(heat + cooling, undimensioned topology).

Reference values were captured from:
  examples/silkeborg_bhe_full_dimensioning_heat/main.py

Run with:
    python -m pytest tests/test_silkeborg_bhe.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import EXAMPLES_DIR

from pythermonet.components import (
    BrineTemperatureLimits,
    DistributionNetworkParameters,
    HeatPumpPeakSupplyParameters,
    VHEFieldParameters,
    build_distribution_network,
    build_vhe_field,
    ground_loads_from_heat_pumps,
)
from pythermonet.core import Annulus, HeatCarrier, Material, PipeSegmentParameters, Soil
from pythermonet.dimensioning import (
    SizingParameters,
    run_bhe_sizing_workflow,
    run_pipedimensioning,
)
from pythermonet.input import (
    read_heat_pumps_tsv,
    read_pipe_catalog,
    read_undimensioned_topology_tsv,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EXAMPLE_BHE_DIR = EXAMPLES_DIR / "silkeborg_bhe_full_dimensioning_heat"

# ---------------------------------------------------------------------------
# Module-scoped fixture — runs the full dimensioning once for all tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bhe_dimensioning():
    """Return (hydraulic, result) from the Silkeborg BHE full-dimensioning run."""
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
        thermal_conductivity=2.36,
        thermal_conductivity_shallow_heating=1.25,
        thermal_conductivity_shallow_cooling=1.25,
        geothermal_heat_flux=0.0185,
        temperature_surface_mean=9.03,
        temperature_surface_amplitude=7.9,
    )

    topology = read_undimensioned_topology_tsv(
        _EXAMPLE_BHE_DIR / "data" / "silkeborg_topology.dat",
    )
    network_parameters = DistributionNetworkParameters(
        roughness=1e-6,
        burial_depth=1.2,
        pipe_spacing=0.3,
        n_pipes_parallel=2,
    )
    network = build_distribution_network(
        topology,
        pipe_material=pipe_material_dist,
        network_parameters=network_parameters,
    )

    # Borehole field
    grout = Material(
        density=1500, specific_heat=2e3, thermal_conductivity=1.75
    )
    pipe_mat_bhe = Material(
        density=1000, specific_heat=2e3, thermal_conductivity=0.4
    )
    borehole = Annulus(diameter_outer=0.152, sdr=1000.0)
    pipe_segment_parameters_bhe = PipeSegmentParameters(
        diameter_outer=0.04, sdr=11.0, roughness=1e-6,
    )
    vhe_field_parameters = VHEFieldParameters(
        shank_spacing=0.015 + 2 * 0.02,
        burial_depth=1.0,
        tilt_rad=0.0,
        orientation_rad=0.0,
        heat_exchanger_type="1U",
    )
    n_boreholes = 6
    spacing_m = 15.0
    bhe_field = build_vhe_field(
        segment_parameters=pipe_segment_parameters_bhe,
        field_parameters=vhe_field_parameters,
        pipe_material=pipe_mat_bhe,
        borehole=borehole,
        grout=grout,
        coordinates=[[0.0, i * spacing_m] for i in range(n_boreholes)],
    )

    hp_list = read_heat_pumps_tsv(
        path=_EXAMPLE_BHE_DIR / "data" / "silkeborg_heat_pump_heat_high_cool.dat"
    )
    peak_supply = HeatPumpPeakSupplyParameters(
        peak_hours_heating=4.0,
        peak_fraction_heating_mode="incremental",
        peak_fraction_heating=1.0,
        peak_hours_cooling=4.0,
        peak_fraction_cooling_mode="incremental",
        peak_fraction_cooling=1.0,
    )
    loads = ground_loads_from_heat_pumps(
        hp_list,
        brine=brine,
        peak_supply=peak_supply,
    )

    sizing = SizingParameters(thermal_dimensioning_lifetime=30.0)

    brine_temperature_limits = BrineTemperatureLimits(
        temperature_brine_min_heating=-3.0,
        temperature_brine_max_cooling=20.0,
    )

    hydraulic = run_pipedimensioning(pipe_catalog, brine, network, hp_list)

    result = run_bhe_sizing_workflow(
        ground_loads=loads,
        vhe_field=bhe_field,
        hydraulic=hydraulic,
        brine=brine,
        soil=soil,
        sizing=sizing,
        brine_temperature_limits=brine_temperature_limits,
    )

    return hydraulic, result


# ---------------------------------------------------------------------------
# Reference values (from example output)
# ---------------------------------------------------------------------------

# Trace order: Main_line, Inner_distribution_ring_1, Inner_distribution_ring_2,
#              Single_branch, Connection_pipes_6_kW, Connection_pipes_10_kW

_REF_OD_MM    = [110.0, 90.0, 75.0, 75.0, 50.0, 50.0]
_REF_RE_HEAT  = [7948,  6322, 4547, 3297, 1729, 2882]
_REF_RE_COOL  = [11871, 9862, 6259, 5144, 2698, 2698]


# ===========================================================================
# Hydraulic dimensioning tests
# ===========================================================================

class TestHydraulicDimensioning:

    def test_diameters_outer(self, bhe_dimensioning):
        hydraulic, _ = bhe_dimensioning
        od_mm = hydraulic.diameters_outer * 1e3
        np.testing.assert_allclose(
            od_mm, _REF_OD_MM, atol=0.1,
            err_msg="Pipe OD selection differs from reference",
        )

    def test_reynolds_heating(self, bhe_dimensioning):
        hydraulic, _ = bhe_dimensioning
        np.testing.assert_allclose(
            hydraulic.reynolds_numbers_heating, _REF_RE_HEAT, rtol=0.01,
            err_msg="Heating Reynolds numbers differ from reference",
        )

    def test_reynolds_cooling(self, bhe_dimensioning):
        hydraulic, _ = bhe_dimensioning
        np.testing.assert_allclose(
            hydraulic.reynolds_numbers_cooling, _REF_RE_COOL, rtol=0.01,
            err_msg="Cooling Reynolds numbers differ from reference",
        )

    def test_governing_mode_cooling_dominates_main(self, bhe_dimensioning):
        """Main line should be governed by cooling (larger flow)."""
        hydraulic, _ = bhe_dimensioning
        assert "cooling" in list(hydraulic.governing_mode)


# ===========================================================================
# BHE thermal dimensioning tests
# ===========================================================================

class TestBHEThermalDimensioning:

    def test_borehole_length(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        assert abs(result.sizing.length_element - 168.98) < 0.5, (
            f"Borehole length {result.sizing.length_element:.2f} m"
            " deviates >0.5 m from 168.98 m"
        )

    def test_governing_mode_cooling(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        assert result.sizing.governing_mode == "cooling"

    def test_borehole_resistance_heating(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        rb = result.sizing.thermal_resistance_heating
        assert abs(rb - 0.1339) < 0.002, (
            f"Rb (heating) = {rb:.4f} K·m/W deviates >0.002 from 0.1339"
        )

    def test_borehole_resistance_cooling(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        rb = result.sizing.thermal_resistance_cooling
        assert abs(rb - 0.1248) < 0.002, (
            f"Rb (cooling) = {rb:.4f} K·m/W deviates >0.002 from 0.1248"
        )

    def test_distribution_fraction_heating(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        f = result.performance_thermonet_heating.load_supply_fraction
        assert abs(f * 100 - 36.8) < 0.5

    def test_distribution_fraction_cooling(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        f = result.performance_thermonet_cooling.load_supply_fraction
        assert abs(f * 100 - 26.7) < 0.5

    def test_system_temperature_heating_annual(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        T = result.temperatures_system.temperature_system_mean_annual_heating
        assert abs(T - 3.54) < 0.05

    def test_system_temperature_heating_winter(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        T = result.temperatures_system.temperature_system_mean_winter_heating
        assert abs(T - (-0.41)) < 0.05

    def test_system_temperature_heating_peak(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        T = result.temperatures_system.temperature_system_mean_peak_heating
        assert abs(T - (-2.59)) < 0.05

    def test_bhe_pressure_drop_heating(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        assert abs(result.pressure_loss_bhe_heating - 78_524) < 500

    def test_bhe_pressure_drop_cooling(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        assert abs(result.pressure_loss_bhe_cooling - 155_427) < 1000
