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
    HHEFieldParameters,
    build_distribution_network,
    build_hhe_field,
    ground_loads_from_heat_pumps,
)
from pythermonet.core import HeatCarrier, Material, PipeSegmentParameters, Soil
from pythermonet.dimensioning import (
    HHEGroundField,
    SizingParameters,
    run_hhe_sizing_workflow,
    run_pipedimensioning,
)
from pythermonet.input import (
    read_heat_pumps_tsv,
    read_pipe_catalog,
    read_undimensioned_topology_tsv,
)
from pythermonet.output import (
    format_hhe_results,
    hhe_results_to_dict,
    print_hhe_results,
    save_hhe_results_json,
    save_hhe_results_txt,
)

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

    topology = read_undimensioned_topology_tsv(
        _EXAMPLE_HHE_DIR / "data" / "silkeborg_topology.dat",
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

    pipe_segment_parameters_hhe = PipeSegmentParameters(
        diameter_outer=0.040, sdr=17.0, roughness=1e-6,
    )
    hhe_field_parameters = HHEFieldParameters(
        n_pipes_parallel=20,
        pipe_spacing=1.5,
        burial_depth=1.2,
        length_element=100.0,
    )
    pipe_infrastructure_hhe = build_hhe_field(
        segment_parameters=pipe_segment_parameters_hhe,
        field_parameters=hhe_field_parameters,
        pipe_material=pipe_material_dist,
    )
    ground_field_hhe = HHEGroundField(
        pipe_infrastructure=pipe_infrastructure_hhe,
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

    result = run_hhe_sizing_workflow(
        ground_loads=loads,
        hhe_field=ground_field_hhe,
        pipe_infrastructure=pipe_infrastructure_hhe,
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


# ===========================================================================
# Results report (.txt) tests
# ===========================================================================

class TestResultsReport:

    @pytest.fixture
    def fixed_result(self, hhe_dimensioning, monkeypatch):
        """Result with fixed run timestamp and pythermonet version."""
        monkeypatch.setattr(
            "pythermonet.output._report_tables.version", lambda _: REPORT_VERSION
        )
        _, result = hhe_dimensioning
        return replace(result, timestamp_run=REPORT_TIMESTAMP)

    def test_timestamp_run_is_timezone_aware(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        assert result.timestamp_run.tzinfo is not None

    def test_format_matches_pre_refactor_output(self, fixed_result):
        # Expected text = header + stdout of print_hhe_results before the
        # format/print/write refactor (content must be preserved).
        expected_path = TESTS_DATA_DIR / "report_hhe_expected.txt"
        expected = expected_path.read_bytes().decode("utf-8")
        assert format_hhe_results(fixed_result) == expected

    def test_print_matches_format(self, fixed_result, capsys):
        print_hhe_results(fixed_result)
        assert capsys.readouterr().out == "\n" + format_hhe_results(fixed_result)

    def test_save_txt_is_utf8_lf(self, fixed_result, tmp_path):
        path = tmp_path / "results.txt"
        save_hhe_results_txt(fixed_result, path)
        assert path.read_bytes() == format_hhe_results(fixed_result).encode("utf-8")
        assert b"\r\n" not in path.read_bytes()

    def test_save_txt_overwrites_existing_file(self, fixed_result, tmp_path):
        path = tmp_path / "results.txt"
        path.write_bytes(b"old content")
        save_hhe_results_txt(fixed_result, path)
        assert path.read_bytes() == format_hhe_results(fixed_result).encode("utf-8")

    def test_save_txt_missing_folder_raises(self, fixed_result, tmp_path):
        with pytest.raises(FileNotFoundError):
            save_hhe_results_txt(fixed_result, tmp_path / "missing" / "results.txt")


# ===========================================================================
# Results JSON tests
# ===========================================================================

class TestResultsJson:

    @pytest.fixture
    def results(self, hhe_dimensioning):
        _, result = hhe_dimensioning
        return hhe_results_to_dict(result)

    def test_sections(self, results):
        assert set(results) == RESULTS_JSON_SECTIONS
        assert set(results["distribution_network"]) == {"performance", "traces"}

    def test_metadata(self, hhe_dimensioning, results):
        _, result = hhe_dimensioning
        assert results["metadata"] == {
            "ground_field_type": "HHE",
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
        excluded = {"brine", "sdr", "sdr_hhe", "diameter_outer_hhe", "burial_depth_hhe"}
        assert not keys & excluded

    def test_values_match_result(self, hhe_dimensioning, results):
        _, result = hhe_dimensioning
        ground_field = results["ground_field"]
        system = results["system"]
        assert ground_field["length_element"]["value"] == result.sizing.length_element
        assert system["temperature_system_peak_heating"]["value"] == (
            result.temperature_system_peak_heating
        )

    def test_absent_mode_is_null_with_same_keys(self, hhe_dimensioning, results):
        _, result = hhe_dimensioning
        without_cooling = hhe_results_to_dict(
            replace(result, performance_thermonet_cooling=None)
        )
        performance = without_cooling["distribution_network"]["performance"]
        assert performance["load_supply_fraction_cooling"]["value"] is None
        assert [key for key, _ in iter_results_entries(without_cooling)] == (
            [key for key, _ in iter_results_entries(results)]
        )

    def test_save_json_equals_dict(self, hhe_dimensioning, results, tmp_path):
        _, result = hhe_dimensioning
        path = tmp_path / "results.json"
        save_hhe_results_json(result, path)
        assert json.loads(path.read_bytes().decode("utf-8")) == results
        assert b"\r\n" not in path.read_bytes()

    def test_save_json_one_line_per_entry(self, hhe_dimensioning, results, tmp_path):
        _, result = hhe_dimensioning
        path = tmp_path / "results.json"
        save_hhe_results_json(result, path)
        lines = path.read_text("utf-8").splitlines()
        entry_lines = [line for line in lines if '"value": ' in line]
        assert len(entry_lines) == len(iter_results_entries(results))
        assert all(line.rstrip(",").endswith("}") for line in entry_lines)

    def test_save_json_nan_raises(self, hhe_dimensioning, tmp_path):
        _, result = hhe_dimensioning
        result_nan = replace(result, pressure_loss_hhe_heating=float("nan"))
        with pytest.raises(ValueError):
            save_hhe_results_json(result_nan, tmp_path / "results.json")

    def test_save_json_missing_folder_raises(self, hhe_dimensioning, tmp_path):
        _, result = hhe_dimensioning
        with pytest.raises(FileNotFoundError):
            save_hhe_results_json(result, tmp_path / "missing" / "results.json")
