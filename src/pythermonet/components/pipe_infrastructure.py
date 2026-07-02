from dataclasses import dataclass
from typing import List
from pythermonet.core.pipe_segment import PipeSegment

@dataclass
class PipeInfrastructure:
    n_parallel_pipes: int
    trace_segments: List[PipeSegment]  # physical pipe sections making up the thermonet route
    pipe_distance: float | None    # m
    burial_depth: float            # m
