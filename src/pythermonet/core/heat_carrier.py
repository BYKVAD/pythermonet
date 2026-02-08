from dataclasses import dataclass
from .material import Material

@dataclass
class HeatCarrier(Material):
    dynamicViscosity: float    # Pa·s
