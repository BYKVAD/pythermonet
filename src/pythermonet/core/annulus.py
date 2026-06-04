from dataclasses import dataclass
from typing import Optional

@dataclass
class Annulus:
    outer_diameter: Optional[float]  # ingen default
    sdr: float
