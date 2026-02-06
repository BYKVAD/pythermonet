from dataclasses import dataclass
from .annulus import Annulus
from .material import Material

@dataclass
class Pipe(Annulus):
    material: Material
    roughnessHeight: float       # m