"""Mapping between settings-file ``"type"`` tags and pythermonet domain classes."""

from __future__ import annotations

import re

from pythermonet.components.distribution_network import DistributionNetworkParameters
from pythermonet.components.ground_loads import HeatPumpPeakSupplyParameters
from pythermonet.components.heat_pump import BrineTemperatureLimits
from pythermonet.components.pipe_infrastructure import PipeInfrastructureParameters
from pythermonet.components.vhe_field import VHEFieldParameters
from pythermonet.core.annulus import Annulus
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegmentParameters
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.sizing_parameters import SizingParameters

KNOWN_TYPES: tuple[type, ...] = (
    Material,
    Soil,
    HeatCarrier,
    DistributionNetworkParameters,
    HeatPumpPeakSupplyParameters,
    SizingParameters,
    BrineTemperatureLimits,
    PipeSegmentParameters,
    PipeInfrastructureParameters,
    Annulus,
    VHEFieldParameters,
)


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def type_name_for(cls: type) -> str:
    """Return the settings-file ``"type"`` tag for a known domain class.

    The tag is derived mechanically from the class name (CamelCase to
    snake_case) rather than hand-typed, so it can never drift out of sync
    with the class it names.

    Parameters
    ----------
    cls : type
        A class registered in `KNOWN_TYPES`.

    Returns
    -------
    str
        The snake_case type tag, e.g. ``"heat_carrier"`` for `HeatCarrier`.

    Raises
    ------
    ValueError
        If `cls` is not registered in `KNOWN_TYPES`.

    """
    if cls not in KNOWN_TYPES:
        known = [c.__name__ for c in KNOWN_TYPES]
        raise ValueError(
            f"{cls.__name__} is not a known settings type. Known types: {known}."
        )
    return _camel_to_snake(cls.__name__)


def class_for_type_name(type_name: str) -> type:
    """Return the domain class registered under a settings-file ``"type"`` tag.

    Parameters
    ----------
    type_name : str
        The snake_case type tag read from a settings file, e.g. ``"soil"``.

    Returns
    -------
    type
        The matching class from `KNOWN_TYPES`.

    Raises
    ------
    ValueError
        If `type_name` does not match any class in `KNOWN_TYPES`.

    """
    for cls in KNOWN_TYPES:
        if _camel_to_snake(cls.__name__) == type_name:
            return cls
    known = [_camel_to_snake(c.__name__) for c in KNOWN_TYPES]
    raise ValueError(f"Unknown settings type '{type_name}'. Known types: {known}.")
