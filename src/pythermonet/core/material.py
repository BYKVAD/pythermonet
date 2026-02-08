from dataclasses import dataclass

@dataclass
class Material:
    rho: float                 # kg/m3
    c: float                   # J/kg/K
    thermalCond: float         # W/m/K