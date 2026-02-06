from dataclasses import dataclass
from typing import List
from .heat_pump import HeatPump

@dataclass
class HeatPumps:
    heatPumpList: List[HeatPump]

    TInletHeating: list[float]
    TInletCooling: list[float]

    fractionPeakHeating: list[float]
    durationPeakHeating: float     # s

    fractionPeakCooling: list[float]
    durationPeakCooling: float     # s
