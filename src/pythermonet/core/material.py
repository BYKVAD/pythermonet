from dataclasses import dataclass

@dataclass
class Material:
    density: float             # kg/m3
    specific_heat: float       # J/kg/K
    thermal_conductivity: float         # W/m/K