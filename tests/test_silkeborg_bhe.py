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
from pathlib import Path

from pythermonet.components.ground_loads import ground_loads_from_heat_pumps
from pythermonet.components.vhe_field import VHEField
from pythermonet.core.annulus import Annulus
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.BHE.bhe_workflow import run_bhe_sizing_workflow
from pythermonet.dimensioning.hydraulic_dimensioning import run_pipedimensioning
from pythermonet.dimensioning.sizing_parameters import SizingParameters
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv
from pythermonet.input.read_pipe_catalogue import read_pipe_catalogue
from pythermonet.input.read_topology import read_undimensioned_topology_tsv_to_network

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "silkeborg_bhe_full_dimensioning_heat"
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPE_CATALOGUE = _REPO_ROOT / "src" / "pythermonet" / "resources" / "pipe_catalogue.csv"

# ---------------------------------------------------------------------------
# Module-scoped fixture — runs the full dimensioning once for all tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bhe_dimensioning():
    """Return (hydraulic, result) from the Silkeborg BHE full-dimensioning run."""
    pipe_catalogue = read_pipe_catalogue(_PIPE_CATALOGUE)

    pipe_material_dist = Material(rho=975, c=1900, thermalCond=0.4)

    brine = HeatCarrier(
        rho=965, c=4450, thermalCond=0.45, dynamicViscosity=5e-3
    )

    soil = Soil(
        rho=2500,
        c=1000,
        thermalCond=2.36,
        thermalCondShallowHeating=1.25,
        thermalCondShallowCooling=1.25,
        Qgeo=0.0185,
        surfaceTemp=9.03,
        surfaceTempAmp=7.9,
    )

    network = read_undimensioned_topology_tsv_to_network(
        _EXAMPLE_DIR / "data" / "silkeborg_topology.dat",
        pipe_material=pipe_material_dist,
        roughness_height=1e-6,
        burial_depth=1.2,
        pipe_distance=0.3,
        n_parallel_pipes=2,
    )

    # Borehole field
    grout         = Material(rho=1500, c=2e3, thermalCond=1.75)
    pipe_mat_bhe  = Material(rho=1000, c=2e3, thermalCond=0.4)
    borehole      = Annulus(outerDiameter=0.152, SDR=1000.0)
    upipe = PipeSegment(
        outerDiameter=0.04, SDR=11.0, material=pipe_mat_bhe,
        roughnessHeight=1e-6, ID=0, length=100,
    )
    n_boreholes = 6
    spacing_m   = 15.0
    bhe_field = VHEField(
        ID=1, HE="1U",
        pipe=upipe, borehole=borehole, grout=grout,
        coordinates=[[0.0, i * spacing_m] for i in range(n_boreholes)],
        shankSpacing=0.015 + 2 * 0.02,
        H_m=120.0, D_m=1.0,
        tilt_rad=0.0, orientation_rad=0.0,
    )

    hp_list = read_heat_pumps_tsv(
        path=_EXAMPLE_DIR / "data" / "silkeborg_heat_pump_heat_high_cool.dat"
    )
    loads = ground_loads_from_heat_pumps(
        hp_list,
        brine,
        peak_heating_h=4.0,
        peak_fraction_heating_mode="incremental",
        peak_fraction_heating=1.0,
        peak_cooling_h=4.0,
        peak_fraction_cooling_mode="incremental",
        peak_fraction_cooling=1.0,
    )

    sizing = SizingParameters(time_horizon_years=30.0)

    hydraulic = run_pipedimensioning(pipe_catalogue, brine, network, hp_list)

    result = run_bhe_sizing_workflow(
        ground_loads=loads,
        vhe_field=bhe_field,
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
#              Single_branch, Connection_pipes_6_kW, Connection_pipes_10_kW

_REF_OD_MM    = [110.0, 90.0, 75.0, 75.0, 50.0, 50.0]
_REF_RE_HEAT  = [7948,  6322, 4547, 3297, 1729, 2882]
_REF_RE_COOL  = [11871, 9862, 6259, 5144, 2698, 2698]


# ===========================================================================
# Hydraulic dimensioning tests
# ===========================================================================

class TestHydraulicDimensioning:

    def test_pipe_outer_diameters(self, bhe_dimensioning):
        hydraulic, _ = bhe_dimensioning
        od_mm = hydraulic.outer_diameter * 1e3
        np.testing.assert_allclose(
            od_mm, _REF_OD_MM, atol=0.1,
            err_msg="Pipe OD selection differs from reference",
        )

    def test_reynolds_heating(self, bhe_dimensioning):
        hydraulic, _ = bhe_dimensioning
        np.testing.assert_allclose(
            hydraulic.Re_heating, _REF_RE_HEAT, rtol=0.01,
            err_msg="Heating Reynolds numbers differ from reference",
        )

    def test_reynolds_cooling(self, bhe_dimensioning):
        hydraulic, _ = bhe_dimensioning
        np.testing.assert_allclose(
            hydraulic.Re_cooling, _REF_RE_COOL, rtol=0.01,
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
        assert abs(result.sizing.L_m - 168.98) < 0.5, (
            f"Borehole length {result.sizing.L_m:.2f} m deviates >0.5 m from 168.98 m"
        )

    def test_governing_mode_cooling(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        assert result.sizing.governing == "cooling"

    def test_borehole_resistance_heating(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        rb = result.sizing.R_heating_K_m_W
        assert abs(rb - 0.1339) < 0.002, (
            f"Rb (heating) = {rb:.4f} K·m/W deviates >0.002 from 0.1339"
        )

    def test_borehole_resistance_cooling(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        rb = result.sizing.R_cooling_K_m_W
        assert abs(rb - 0.1248) < 0.002, (
            f"Rb (cooling) = {rb:.4f} K·m/W deviates >0.002 from 0.1248"
        )

    def test_distribution_fraction_heating(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        f = result.dist_thermal_heat.F_total
        assert abs(f * 100 - 36.8) < 0.5

    def test_distribution_fraction_cooling(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        f = result.dist_thermal_cool.F_total
        assert abs(f * 100 - 26.7) < 0.5

    def test_system_temperature_heating_annual(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        T = result.sys_temps.T_avg_heat_annual_C
        assert abs(T - 3.54) < 0.05

    def test_system_temperature_heating_winter(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        T = result.sys_temps.T_avg_heat_winter_C
        assert abs(T - (-0.41)) < 0.05

    def test_system_temperature_heating_peak(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        T = result.sys_temps.T_avg_heat_peak_C
        assert abs(T - (-2.59)) < 0.05

    def test_bhe_pressure_drop_heating(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        assert abs(result.bhe_dp_heat_Pa - 78_524) < 500

    def test_bhe_pressure_drop_cooling(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        assert abs(result.bhe_dp_cool_Pa - 155_427) < 1000
