from dataclasses import dataclass
from pythermonet.core.heat_carrier import HeatCarrier

@dataclass
class HeatPump:
    ID: int

    annualHeatingLoad: float
    winterHeatingLoad: float
    peakHeatingLoad: float

    annualSCOP: float
    winterSCOP: float
    peakCOP: float

    deltaTHeating: float

    annualCoolingLoad: float
    summerCoolingLoad: float
    peakCoolingLoad: float

    EER: float
    deltaTCooling: float

    sourceHeatCarrier: HeatCarrier

    # --- computed once, used everywhere ---
    annualHeating_ground_load: float = 0.0   # W, + = extracted from ground
    winterHeating_ground_load: float = 0.0   # W
    peakHeating_ground_load: float = 0.0     # W

    annualCooling_ground_load: float = 0.0   # W, - = injected to ground
    summerCooling_ground_load: float = 0.0   # W
    peakCooling_ground_load: float = 0.0     # W
