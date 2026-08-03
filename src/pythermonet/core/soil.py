from dataclasses import dataclass
from .material import Material

@dataclass
class Soil(Material):
    thermal_conductivity_shallow_heating: float  # [W/m/K]
    thermal_conductivity_shallow_cooling: float  # [W/m/K]
    geothermal_heat_flux: float                  # [W/m²]
    temperature_surface_mean: float              # [°C]
    temperature_surface_amplitude: float         # [K]
