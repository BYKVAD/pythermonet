from dataclasses import dataclass
from typing import Optional

@dataclass
class Annulus:
    outerDiameter: Optional[float]  # ingen default
    SDR: float
