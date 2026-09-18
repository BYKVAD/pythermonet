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
class HHEFieldParameters:
    """Construction parameters for a single-segment-trace HHE field.

    `length_element` is a placeholder, not a genuine input — it is always
    overwritten by the HHE bisection solver (`pythermonet.dimensioning`)
    regardless of its starting value. It exists here only so a settings
    file can carry a length for a site-placement viewer to render before
    (re)dimensioning has run.

    Parameters
    ----------
    n_pipes_parallel : int
        Number of parallel pipes [-].
    pipe_spacing : float or None
        Wall-to-wall spacing between parallel pipes [m].
    burial_depth : float
        Burial depth of the pipes [m].
    length_element : float
        Length of the single pipe segment representing the field [m].

    """

    n_pipes_parallel: int
    pipe_spacing: float | None  # m
    burial_depth: float         # m
    length_element: float       # m


def build_hhe_field(
    segment_parameters: PipeSegmentParameters,
    field_parameters: HHEFieldParameters,
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
    field_parameters : HHEFieldParameters
        Parallel-pipe count, spacing, burial depth, and starting length.
    pipe_material : Material
        Thermal properties of the pipe wall material.

    Returns
    -------
    PipeInfrastructure
        `segments_trace` holds a single `PipeSegment` at
        `field_parameters.length_element` — sized later by the HHE
        bisection solver, which does not mutate this object in place.

    """
    segment = PipeSegment(
        diameter_outer=segment_parameters.diameter_outer,
        sdr=segment_parameters.sdr,
        material=pipe_material,
        roughness=segment_parameters.roughness,
        id_=0,
        length=field_parameters.length_element,
    )

    return PipeInfrastructure(
        n_pipes_parallel=field_parameters.n_pipes_parallel,
        segments_trace=[segment],
        pipe_spacing=field_parameters.pipe_spacing,
        burial_depth=field_parameters.burial_depth,
    )
