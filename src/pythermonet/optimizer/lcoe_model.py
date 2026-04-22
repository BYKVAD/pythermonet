from dataclasses import dataclass, field
from typing import List, Dict

from lcc5gdhc.cost_items import (
    CapexScheduleTS, 
    OpexFixedTS,
    OpexVariableTS,
    EnergyPricePathTS,
    LoadPathTS,
    DebtScheduleTS
)

@dataclass
class LCOE:
    capex_ts: List[CapexScheduleTS]
    opex_fixed_ts: List[OpexFixedTS]
    opex_variable_ts: List[OpexVariableTS]
    price_path_ts: Dict[str, EnergyPricePathTS]
    debt_ts: List[DebtScheduleTS]
    loads_ts: LoadPathTS = None
    years: int = 20
    steps_per_year: int = 1
    r: float = 0.03