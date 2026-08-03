from dataclasses import dataclass
from typing import Optional

@dataclass
class Annulus:
    diameter_outer: Optional[float]  # ingen default
    sdr: float
