from dataclasses import dataclass
from typing import List
from core.pipe_segment import PipeSegment
from core.material import Material

@dataclass
class VHEField:
    ID: int
    HE: str                        # "1U", "2U", "COAX"
    pipe: PipeSegment
    coordinates: List[list]        # [[x, y, z], ...]
    VHERadius: float               # m
    shankSpacing: float            # m
    grout: Material
