from dataclasses import dataclass
from .pipe import Pipe

@dataclass
class PipeSegment(Pipe):
    ID: int
    length: float                # m