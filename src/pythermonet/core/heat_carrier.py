from dataclasses import dataclass
from .material import Material

@dataclass
class HeatCarrier(Material):
    dynamic_viscosity: float    # Pa·s
