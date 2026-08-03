from dataclasses import dataclass
from .pipe import Pipe

@dataclass
class PipeSegment(Pipe):
    id_: int
    length: float                # m
