"""Core domain classes: materials, fluids, soil, and pipe geometry."""

from __future__ import annotations

from pythermonet.core.annulus import Annulus
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegment, PipeSegmentParameters
from pythermonet.core.soil import Soil

__all__ = [
    "Annulus",
    "HeatCarrier",
    "Material",
    "PipeSegment",
    "PipeSegmentParameters",
    "Soil",
]
