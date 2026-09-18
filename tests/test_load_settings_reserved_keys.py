"""
test_load_settings_reserved_keys.py
------------------------------------
Regression tests for `load_settings()` tolerating reserved
``"qthermonet_*"`` top-level keys (external bookkeeping owned by
QThermonet, not by pythermonet itself).

Run with:
    python -m pytest tests/test_load_settings_reserved_keys.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pythermonet.input import load_settings

_MATERIAL_BLOCK = {
    "type": "material",
    "values": {
        "density": {"value": 2000.0, "unit": "kg/m^3"},
        "specific_heat": {"value": 1000.0, "unit": "J/kg/K"},
        "thermal_conductivity": {"value": 2.4, "unit": "W/m/K"},
    },
}


def _write_settings(path: Path, extra: dict[str, object]) -> Path:
    settings_path = path / "settings.json"
    payload = {"grout": _MATERIAL_BLOCK, **extra}
    settings_path.write_bytes(json.dumps(payload).encode("utf-8"))
    return settings_path


def test_reserved_prefixed_key_is_skipped(tmp_path: Path) -> None:
    """A single "qthermonet_"-prefixed key is ignored, ordinary blocks still load."""
    settings_path = _write_settings(
        tmp_path,
        {
            "qthermonet_hhe_field_parameters": {
                "geojson_path": "field.geojson",
                "connection_node_x": 500000.0,
                "connection_node_y": 6200000.0,
                "rotation_deg": 0.0,
                "mirror_left": False,
            }
        },
    )

    objects = load_settings(settings_path)

    assert set(objects) == {"grout"}


def test_multiple_reserved_prefixed_keys_are_skipped(tmp_path: Path) -> None:
    """Several distinct "qthermonet_*" keys can coexist and are all ignored."""
    settings_path = _write_settings(
        tmp_path,
        {
            "qthermonet_hhe_field_parameters": {"geojson_path": "field.geojson"},
            "qthermonet_placement_options": {"mirror_left": True},
        },
    )

    objects = load_settings(settings_path)

    assert set(objects) == {"grout"}


def test_reserved_prefix_match_is_case_insensitive(tmp_path: Path) -> None:
    """Mixed-case reserved keys (e.g. "QThermonet_...") are also skipped."""
    settings_path = _write_settings(
        tmp_path,
        {"QThermonet_hhe_field_parameters": {"geojson_path": "field.geojson"}},
    )

    objects = load_settings(settings_path)

    assert set(objects) == {"grout"}


def test_unrelated_unrecognized_key_still_raises(tmp_path: Path) -> None:
    """A top-level key that only coincidentally resembles the prefix still fails.

    Guards against the guard being too permissive: "qthermonet" without the
    trailing underscore, and unrelated unknown roles, must still be rejected
    the same way any other malformed/unrecognized block always was.
    """
    settings_path = _write_settings(
        tmp_path, {"qthermonet": {"type": "bogus", "values": {}}}
    )

    with pytest.raises(ValueError, match="bogus"):
        load_settings(settings_path)
