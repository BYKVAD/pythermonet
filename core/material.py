from dataclasses import dataclass

@dataclass
class Material:
    rhoC: float                  # J/m3/K
    thermalCond: float           # W/m/K