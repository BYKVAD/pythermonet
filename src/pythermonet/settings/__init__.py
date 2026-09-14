"""Schema logic for the project settings file format: registry, validation, presets."""

from __future__ import annotations

from pythermonet.settings.presets import (
    ANNULUS_PRESETS,
    BRINE_TEMPERATURE_LIMITS_PRESETS,
    DISTRIBUTION_NETWORK_PARAMETERS_PRESETS,
    HEAT_CARRIER_PRESETS,
    HEAT_PUMP_PEAK_SUPPLY_PARAMETERS_PRESETS,
    MATERIAL_PRESETS,
    PIPE_INFRASTRUCTURE_PARAMETERS_PRESETS,
    PIPE_SEGMENT_PARAMETERS_PRESETS,
    PRESETS_BY_TYPE,
    PRESETS_SOURCE_NOTE,
    SIZING_PARAMETERS_PRESETS,
    SOIL_PRESETS,
    VHE_FIELD_PARAMETERS_PRESETS,
)
from pythermonet.settings.registry import (
    KNOWN_TYPES,
    class_for_type_name,
    type_name_for,
)
from pythermonet.settings.validation import describe_block_problem, field_names

__all__ = [
    "ANNULUS_PRESETS",
    "BRINE_TEMPERATURE_LIMITS_PRESETS",
    "DISTRIBUTION_NETWORK_PARAMETERS_PRESETS",
    "HEAT_CARRIER_PRESETS",
    "HEAT_PUMP_PEAK_SUPPLY_PARAMETERS_PRESETS",
    "KNOWN_TYPES",
    "MATERIAL_PRESETS",
    "PIPE_INFRASTRUCTURE_PARAMETERS_PRESETS",
    "PIPE_SEGMENT_PARAMETERS_PRESETS",
    "PRESETS_BY_TYPE",
    "PRESETS_SOURCE_NOTE",
    "SIZING_PARAMETERS_PRESETS",
    "SOIL_PRESETS",
    "VHE_FIELD_PARAMETERS_PRESETS",
    "class_for_type_name",
    "describe_block_problem",
    "field_names",
    "type_name_for",
]
