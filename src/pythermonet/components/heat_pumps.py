from dataclasses import dataclass, field
from typing import List
from .heat_pump import HeatPump

@dataclass
class HeatPumps:
    heatPumpList: List[HeatPump]

    TInletHeating: list[float] = field(default_factory=list)
    TInletCooling: list[float] = field(default_factory=list)

    fractionPeakHeating: list[float] = field(default_factory=list)
    durationPeakHeating: float = 0.0     # s

    fractionPeakCooling: list[float] = field(default_factory=list)
    durationPeakCooling: float = 0.0     # s
