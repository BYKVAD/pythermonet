from dataclasses import dataclass
from core.heat_carrier import HeatCarrier

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
