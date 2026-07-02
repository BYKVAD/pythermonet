from dataclasses import dataclass
from .material import Material

@dataclass
class Soil(Material):
    thermal_conductivity_shallow_heating: float  # [W/m/K]
    thermal_conductivity_shallow_cooling: float  # [W/m/K]
    geothermal_heat_flux: float                  # [W/m²]
    surface_temperature: float                   # [°C]
    surface_temperature_amplitude: float         # [K]
