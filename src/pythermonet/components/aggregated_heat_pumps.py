from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.input.read_aggregated_load import AggregatedLoadInput
from pythermonet.system.diversity_factor import diversity_factor_from_n_heat_pumps


@dataclass(frozen=True)
class AggregatedHeatPumps:
    """
    Ground-side thermal loads and flow parameters derived from aggregated
    (district-level) heating and cooling loads.

    Provides the same interface as ``HeatPumps`` so it can be used
    interchangeably throughout the dimensioning pipeline.

    Diversity factor formula
    -----------------------
    The peak load is scaled by::

        S = f_peak * diversity_factor(n_consumers)
            = f_peak * (0.62 + 0.38 / n_consumers)

    where ``f_peak`` is the fraction of the raw peak demand that the heat
    pump system is designed to cover (typically 1.0 for full coverage).
    """

    load_input: AggregatedLoadInput
    brine: HeatCarrier

    f_peak_heating: float = 1.0   # fraction of peak heating load covered [0–1]
    f_peak_cooling: float = 1.0   # fraction of peak cooling load covered [0–1]

    # ---------- computed ----------
    heating_ground_load_W: np.ndarray = field(init=False)   # [W] (3,): annual, winter, peak
    cooling_ground_load_W: Optional[np.ndarray] = field(init=False)  # (3,) or None

    has_cooling: bool = field(init=False)

    deltaT_sys_heat: float = field(init=False)
    deltaT_sys_cool: Optional[float] = field(init=False)

    aggregated_mdot_peak_heat_kg_s: float = field(init=False)
    aggregated_mdot_peak_cool_kg_s: Optional[float] = field(init=False)

    def __post_init__(self) -> None:
        li = self.load_input
        cp = float(self.brine.c)

        if not (0.0 <= self.f_peak_heating <= 1.0):
            raise ValueError(f"f_peak_heating must be in [0, 1]. Got {self.f_peak_heating}")
        if not (0.0 <= self.f_peak_cooling <= 1.0):
            raise ValueError(f"f_peak_cooling must be in [0, 1]. Got {self.f_peak_cooling}")

        # --- Heating ground loads ---
        if li.cop_yearly_heating <= 0 or li.cop_winter_heating <= 0 or li.cop_peak_heating <= 0:
            raise ValueError("All COP values must be > 0")
        if li.deltaT_heating <= 0:
            raise ValueError("deltaT_heating must be > 0")

        P_ann_H  = li.load_yearly_heating  * (1.0 - 1.0 / li.cop_yearly_heating)
        P_win_H  = li.load_winter_heating  * (1.0 - 1.0 / li.cop_winter_heating)
        P_raw_H  = li.load_daily_peak_heating * (1.0 - 1.0 / li.cop_peak_heating)

        S_H = self.f_peak_heating * diversity_factor_from_n_heat_pumps(li.n_consumers_heating)
        P_peak_H = P_raw_H * S_H

        object.__setattr__(
            self,
            "heating_ground_load_W",
            np.array([P_ann_H, P_win_H, P_peak_H], dtype=float),
        )
        object.__setattr__(self, "deltaT_sys_heat", float(li.deltaT_heating))
        object.__setattr__(
            self,
            "aggregated_mdot_peak_heat_kg_s",
            float(P_peak_H / (cp * li.deltaT_heating)),
        )

        # --- Cooling ground loads ---
        object.__setattr__(self, "has_cooling", li.has_cooling)

        if li.has_cooling:
            if li.eer_cooling <= 0:
                raise ValueError("eer_cooling must be > 0 when cooling is active")
            if li.deltaT_cooling <= 0:
                raise ValueError("deltaT_cooling must be > 0 when cooling is active")

            P_ann_C  = li.load_yearly_cooling  * (1.0 + 1.0 / li.eer_cooling)
            P_sum_C  = li.load_summer_cooling  * (1.0 + 1.0 / li.eer_cooling)
            P_raw_C  = li.load_daily_peak_cooling * (1.0 + 1.0 / li.eer_cooling)

            S_C = self.f_peak_cooling * diversity_factor_from_n_heat_pumps(li.n_consumers_cooling)
            P_peak_C = P_raw_C * S_C

            object.__setattr__(
                self,
                "cooling_ground_load_W",
                np.array([P_ann_C, P_sum_C, P_peak_C], dtype=float),
            )
            object.__setattr__(self, "deltaT_sys_cool", float(li.deltaT_cooling))
            object.__setattr__(
                self,
                "aggregated_mdot_peak_cool_kg_s",
                float(P_peak_C / (cp * li.deltaT_cooling)),
            )
        else:
            object.__setattr__(self, "cooling_ground_load_W", None)
            object.__setattr__(self, "deltaT_sys_cool", None)
            object.__setattr__(self, "aggregated_mdot_peak_cool_kg_s", None)
