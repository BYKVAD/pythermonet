from dataclasses import dataclass
from .material import Material

@dataclass
class Soil(Material):
    thermalCondShallowHeating: float   # W/m/K
    thermalCondShallowCooling: float   # W/m/K
    Qgeo: float                        # W/m2
    surfaceTemp: float                # °C
    surfaceTempAmp: float             # K