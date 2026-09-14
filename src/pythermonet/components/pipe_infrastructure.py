from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegment, PipeSegmentParameters


@dataclass
class PipeInfrastructure:
    n_pipes_parallel: int
    segments_trace: List[PipeSegment]  # physical pipe sections making up the thermonet route
    pipe_spacing: float | None    # m
    burial_depth: float            # m


@dataclass
class PipeInfrastructureParameters:
    """Construction parameters for a single-segment-trace pipe infrastructure.

    Parameters
    ----------
    n_pipes_parallel : int
        Number of parallel pipes [-].
    pipe_spacing : float or None
        Wall-to-wall spacing between parallel pipes [m].
    burial_depth : float
        Burial depth of the pipes [m].

    """

    n_pipes_parallel: int
    pipe_spacing: float | None  # m
    burial_depth: float         # m


def build_pipe_infrastructure(
    segment_parameters: PipeSegmentParameters,
    infrastructure_parameters: PipeInfrastructureParameters,
    pipe_material: Material,
) -> PipeInfrastructure:
    """Assemble a single-segment-trace `PipeInfrastructure`.

    Builds the one `PipeSegment` making up `segments_trace` internally —
    only the single-segment-trace case is supported, matching the only
    shape `HHEGroundField` accepts.

    Parameters
    ----------
    segment_parameters : PipeSegmentParameters
        Outer diameter, SDR, and roughness of the pipe segment.
    infrastructure_parameters : PipeInfrastructureParameters
        Parallel-pipe count, spacing, and burial depth.
    pipe_material : Material
        Thermal properties of the pipe wall material.

    Returns
    -------
    PipeInfrastructure
        `segments_trace` holds a single placeholder-length `PipeSegment`
        (`length=100.0`) — sized later by the HHE bisection solver.

    """
    segment = PipeSegment(
        diameter_outer=segment_parameters.diameter_outer,
        sdr=segment_parameters.sdr,
        material=pipe_material,
        roughness=segment_parameters.roughness,
        id_=0,
        length=100.0,  # placeholder — will be sized
    )

    return PipeInfrastructure(
        n_pipes_parallel=infrastructure_parameters.n_pipes_parallel,
        segments_trace=[segment],
        pipe_spacing=infrastructure_parameters.pipe_spacing,
        burial_depth=infrastructure_parameters.burial_depth,
    )
