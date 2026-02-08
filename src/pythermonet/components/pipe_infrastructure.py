from dataclasses import dataclass
from typing import List
from pythermonet.core.pipe_segment import PipeSegment

@dataclass
class PipeInfrastructure:
    NParallelPipes: int
    traceSegments: List[PipeSegment]
    pipeDistance: float | None     # m
    burialDepth: float             # m
