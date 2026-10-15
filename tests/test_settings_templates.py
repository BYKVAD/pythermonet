"""
test_settings_templates.py
--------------------------
Regression tests for the bundled settings templates: each template must
load and contain exactly the roles (with the right types) that the
``*_settings_from_json`` examples read from it.

Guards against a role key being renamed in a template without updating the
code that reads it (as happened with ``pipe_infrastructure_hhe`` ->
``hhe_field_parameters``).

Run with:
    python -m pytest tests/test_settings_templates.py -v
"""
from __future__ import annotations

import pytest

from pythermonet.components import (
    BrineTemperatureLimits,
    DistributionNetworkParameters,
    HeatPumpPeakSupplyParameters,
    HHEFieldParameters,
    VHEFieldParameters,
)
from pythermonet.core import Annulus, HeatCarrier, Material, PipeSegmentParameters, Soil
from pythermonet.dimensioning import SizingParameters
from pythermonet.input import load_settings
from pythermonet.output import save_settings
from pythermonet.resources import SETTINGS_TEMPLATE_BHE_PATH, SETTINGS_TEMPLATE_HHE_PATH

_ROLES_COMMON = {
    "pipe_material_dist": Material,
    "brine": HeatCarrier,
    "soil": Soil,
    "distribution_network_parameters": DistributionNetworkParameters,
    "heat_pump_peak_supply_parameters": HeatPumpPeakSupplyParameters,
    "sizing_parameters": SizingParameters,
    "brine_temperature_limits": BrineTemperatureLimits,
}

_ROLES_BHE = {
    **_ROLES_COMMON,
    "grout": Material,
    "pipe_material_bhe": Material,
    "borehole": Annulus,
    "pipe_segment_bhe": PipeSegmentParameters,
    "vhe_field_parameters": VHEFieldParameters,
}

_ROLES_HHE = {
    **_ROLES_COMMON,
    "pipe_material_hhe": Material,
    "pipe_segment_hhe": PipeSegmentParameters,
    "hhe_field_parameters": HHEFieldParameters,
}


@pytest.mark.parametrize(
    ("template_path", "roles_expected"),
    [
        (SETTINGS_TEMPLATE_BHE_PATH, _ROLES_BHE),
        (SETTINGS_TEMPLATE_HHE_PATH, _ROLES_HHE),
    ],
    ids=["bhe", "hhe"],
)
def test_template_roles_match_examples(template_path, roles_expected):
    settings = load_settings(template_path)

    assert set(settings) == set(roles_expected)
    for role, cls in roles_expected.items():
        assert isinstance(settings[role], cls), (
            f"Role '{role}' is {type(settings[role]).__name__}, expected {cls.__name__}"
        )


@pytest.mark.parametrize(
    "template_path",
    [SETTINGS_TEMPLATE_BHE_PATH, SETTINGS_TEMPLATE_HHE_PATH],
    ids=["bhe", "hhe"],
)
def test_template_is_save_settings_output(template_path, tmp_path):
    # Templates are generated, never hand-edited: loading and re-saving one
    # must reproduce it byte for byte (layout included).
    path = tmp_path / template_path.name
    save_settings(path, load_settings(template_path))
    assert path.read_bytes() == template_path.read_bytes()
