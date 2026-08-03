from dataclasses import dataclass
from typing import List
from pythermonet.core.pipe_segment import PipeSegment

@dataclass
class PipeInfrastructure:
    n_pipes_parallel: int
    segments_trace: List[PipeSegment]  # physical pipe sections making up the thermonet route
    pipe_spacing: float | None    # m
    burial_depth: float            # m
