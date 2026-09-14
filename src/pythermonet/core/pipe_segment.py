from __future__ import annotations

from dataclasses import dataclass

from .pipe import Pipe


@dataclass
class PipeSegment(Pipe):
    id_: int
    length: float                # m


@dataclass
class PipeSegmentParameters:
    """Construction parameters for a single pipe segment.

    Parameters
    ----------
    diameter_outer : float
        Outer diameter of the pipe [m].
    sdr : float
        Standard dimension ratio of the pipe [-].
    roughness : float
        Pipe wall roughness [m].

    """

    diameter_outer: float  # m
    sdr: float
    roughness: float       # m
