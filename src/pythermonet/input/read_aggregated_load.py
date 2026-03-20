from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class AggregatedLoadInput:
    """
    Aggregated heating (and optional cooling) loads for an entire district.

    All loads are on the building-side [W].  Ground loads are derived later
    using COP/EER.  The diversity factor is applied to the peak only.
    """

    n_consumers_heating: int

    load_yearly_heating: float
    load_winter_heating: float
    load_daily_peak_heating: float

    cop_yearly_heating: float
    cop_winter_heating: float
    cop_peak_heating: float
    deltaT_heating: float          # brine ΔT across HP, heating mode [K]

    n_consumers_cooling: int

    load_yearly_cooling: float
    load_summer_cooling: float
    load_daily_peak_cooling: float

    eer_cooling: float
    deltaT_cooling: float          # brine ΔT across HP, cooling mode [K]

    # Derived after init
    has_cooling: bool = field(init=False)

    def __post_init__(self) -> None:
        self.has_cooling = (
            self.load_daily_peak_cooling > 0.0
            and self.eer_cooling > 0.0
            and self.deltaT_cooling > 0.0
        )


def read_aggregated_load_tsv(path: str | Path) -> AggregatedLoadInput:
    """
    Read a single-row TSV with aggregated system loads.

    Required columns
    ----------------
    No. consumers, Yearly_heating_load_(W), Winter_heating_load_(W),
    Daily_heating_load_(W), Year_COP, Winter_COP, Hour_COP, dT_HP_Heating

    Optional cooling columns (all must be present and non-zero for cooling
    to be activated)
    -----------------------------------------------------------------------
    Yearly_cooling_load_(W), Summer_cooling_load_(W), Daily_cooling_load_(W),
    EER, dT_HP_Cooling

    Optional per-mode consumer counts
    ----------------------------------
    No. consumers heating  (defaults to 'No. consumers')
    No. consumers cooling  (defaults to 'No. consumers')
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Aggregated load TSV not found: {p}")

    df = pd.read_csv(p, sep=r"\t+", engine="python")
    df.columns = [c.strip() for c in df.columns]

    if df.empty:
        raise ValueError(f"Aggregated load file is empty: {p}")

    row = df.iloc[0].to_dict()

    def _f(col: str) -> float:
        if col not in row:
            raise ValueError(f"Aggregated load TSV missing required column: '{col}'")
        return float(row[col])

    n_base = int(_f("No. consumers"))
    n_heat = int(row.get("No. consumers heating", n_base))
    n_cool = int(row.get("No. consumers cooling", n_base))

    # Optional cooling columns
    _COOL_COLS = [
        "Yearly_cooling_load_(W)",
        "Summer_cooling_load_(W)",
        "Daily_cooling_load_(W)",
        "EER",
        "dT_HP_Cooling",
    ]
    has_cool_cols = all(c in row for c in _COOL_COLS)

    if has_cool_cols:
        load_y_c = float(row["Yearly_cooling_load_(W)"])
        load_s_c = float(row["Summer_cooling_load_(W)"])
        load_p_c = float(row["Daily_cooling_load_(W)"])
        eer      = float(row["EER"])
        dT_c     = float(row["dT_HP_Cooling"])
    else:
        load_y_c = load_s_c = load_p_c = eer = dT_c = 0.0

    return AggregatedLoadInput(
        n_consumers_heating=n_heat,
        load_yearly_heating=_f("Yearly_heating_load_(W)"),
        load_winter_heating=_f("Winter_heating_load_(W)"),
        load_daily_peak_heating=_f("Daily_heating_load_(W)"),
        cop_yearly_heating=_f("Year_COP"),
        cop_winter_heating=_f("Winter_COP"),
        cop_peak_heating=_f("Hour_COP"),
        deltaT_heating=_f("dT_HP_Heating"),
        n_consumers_cooling=n_cool,
        load_yearly_cooling=load_y_c,
        load_summer_cooling=load_s_c,
        load_daily_peak_cooling=load_p_c,
        eer_cooling=eer,
        deltaT_cooling=dT_c,
    )
