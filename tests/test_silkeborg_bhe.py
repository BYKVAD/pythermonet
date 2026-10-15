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

import json
from dataclasses import replace
from importlib.metadata import version

import numpy as np
import pytest
from conftest import (
    EXAMPLES_DIR,
    REPORT_TIMESTAMP,
    REPORT_VERSION,
    RESULTS_JSON_SECTIONS,
    TESTS_DATA_DIR,
    iter_results_entries,
)

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
from pythermonet.output import (
    bhe_results_to_dict,
    format_bhe_results,
    print_bhe_results,
    save_bhe_results_json,
    save_bhe_results_txt,
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


# ===========================================================================
# Results report (.txt) tests
# ===========================================================================

class TestResultsReport:

    @pytest.fixture
    def fixed_result(self, bhe_dimensioning, monkeypatch):
        """Result with fixed run timestamp and pythermonet version."""
        monkeypatch.setattr(
            "pythermonet.output._report_tables.version", lambda _: REPORT_VERSION
        )
        _, result = bhe_dimensioning
        return replace(result, timestamp_run=REPORT_TIMESTAMP)

    def test_timestamp_run_is_timezone_aware(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        assert result.timestamp_run.tzinfo is not None

    def test_format_matches_pre_refactor_output(self, fixed_result):
        # Expected text = header + stdout of print_bhe_results before the
        # format/print/write refactor (content must be preserved).
        expected_path = TESTS_DATA_DIR / "report_bhe_expected.txt"
        expected = expected_path.read_bytes().decode("utf-8")
        assert format_bhe_results(fixed_result) == expected

    def test_print_matches_format(self, fixed_result, capsys):
        print_bhe_results(fixed_result)
        assert capsys.readouterr().out == "\n" + format_bhe_results(fixed_result)

    def test_save_txt_is_utf8_lf(self, fixed_result, tmp_path):
        path = tmp_path / "results.txt"
        save_bhe_results_txt(fixed_result, path)
        assert path.read_bytes() == format_bhe_results(fixed_result).encode("utf-8")
        assert b"\r\n" not in path.read_bytes()

    def test_save_txt_overwrites_existing_file(self, fixed_result, tmp_path):
        path = tmp_path / "results.txt"
        path.write_bytes(b"old content")
        save_bhe_results_txt(fixed_result, path)
        assert path.read_bytes() == format_bhe_results(fixed_result).encode("utf-8")

    def test_save_txt_missing_folder_raises(self, fixed_result, tmp_path):
        with pytest.raises(FileNotFoundError):
            save_bhe_results_txt(fixed_result, tmp_path / "missing" / "results.txt")


# ===========================================================================
# Results JSON tests
# ===========================================================================

class TestResultsJson:

    @pytest.fixture
    def results(self, bhe_dimensioning):
        _, result = bhe_dimensioning
        return bhe_results_to_dict(result)

    def test_sections(self, results):
        assert set(results) == RESULTS_JSON_SECTIONS
        assert set(results["distribution_network"]) == {"performance", "traces"}

    def test_metadata(self, bhe_dimensioning, results):
        _, result = bhe_dimensioning
        assert results["metadata"] == {
            "ground_field_type": "BHE",
            "pythermonet_version": version("pythermonet"),
            "timestamp_run": result.timestamp_run.isoformat(timespec="seconds"),
        }

    def test_every_entry_has_value_unit_description(self, results):
        for key, entry in iter_results_entries(results):
            assert set(entry) == {"value", "unit", "description"}, key
            assert entry["unit"] and entry["description"], key

    def test_traces_align_with_names(self, results):
        traces = results["distribution_network"]["traces"]
        n_traces = len(traces["names_trace"]["value"])
        for key, entry in traces.items():
            assert len(entry["value"]) == n_traces, key

    def test_no_input_echoes(self, results):
        keys = {key for key, _ in iter_results_entries(results)}
        excluded = {
            "brine", "sdr", "heat_pump_ids_trace", "temperature_thermonet_peak_heating"
        }
        assert not keys & excluded

    def test_values_match_result(self, bhe_dimensioning, results):
        _, result = bhe_dimensioning
        ground_field = results["ground_field"]
        traces = results["distribution_network"]["traces"]
        assert ground_field["length_element"]["value"] == result.sizing.length_element
        assert traces["diameters_outer"]["value"] == (
            result.hydraulic.diameters_outer.tolist()
        )

    def test_absent_mode_is_null_with_same_keys(self, bhe_dimensioning, results):
        _, result = bhe_dimensioning
        without_cooling = bhe_results_to_dict(
            replace(result, performance_thermonet_cooling=None)
        )
        performance = without_cooling["distribution_network"]["performance"]
        assert performance["load_supply_fraction_cooling"]["value"] is None
        assert [key for key, _ in iter_results_entries(without_cooling)] == (
            [key for key, _ in iter_results_entries(results)]
        )

    def test_save_json_equals_dict(self, bhe_dimensioning, results, tmp_path):
        _, result = bhe_dimensioning
        path = tmp_path / "results.json"
        save_bhe_results_json(result, path)
        assert json.loads(path.read_bytes().decode("utf-8")) == results
        assert b"\r\n" not in path.read_bytes()

    def test_save_json_one_line_per_entry(self, bhe_dimensioning, results, tmp_path):
        _, result = bhe_dimensioning
        path = tmp_path / "results.json"
        save_bhe_results_json(result, path)
        lines = path.read_text("utf-8").splitlines()
        entry_lines = [line for line in lines if '"value": ' in line]
        assert len(entry_lines) == len(iter_results_entries(results))
        assert all(line.rstrip(",").endswith("}") for line in entry_lines)

    def test_save_json_nan_raises(self, bhe_dimensioning, tmp_path):
        _, result = bhe_dimensioning
        result_nan = replace(result, pressure_loss_bhe_heating=float("nan"))
        with pytest.raises(ValueError):
            save_bhe_results_json(result_nan, tmp_path / "results.json")

    def test_save_json_missing_folder_raises(self, bhe_dimensioning, tmp_path):
        _, result = bhe_dimensioning
        with pytest.raises(FileNotFoundError):
            save_bhe_results_json(result, tmp_path / "missing" / "results.json")
