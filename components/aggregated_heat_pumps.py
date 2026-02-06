from dataclasses import dataclass
from .heat_pump import HeatPump

@dataclass
class AggregatedHeatPump(HeatPump):
    diversityFactor: float
