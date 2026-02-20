from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np

from .heat_pump import HeatPump
from ..core.heat_carrier import HeatCarrier
from ..system.diversity_factor import diversity_factor_from_n_heat_pumps

PeakSupplyMode = Literal["absolute", "incremental"]


@dataclass(frozen=True)
class HeatPumps:
    """
    Varmepumpe-population med aggregerede jordlaster til 3-puls definitionen
    + aggregerede peak flows (q og mdot).

    Output:
      - heating_ground_load_W: [year, winter, peak_eff]
      - cooling_ground_load_W: [year, summer, peak_eff] eller None

    Flows:
      - aggregated_q_peak_heat_m3_s / aggregated_mdot_peak_heat_kg_s svarer til peak_eff (heating)
      - aggregated_q_peak_cool_m3_s / aggregated_mdot_peak_cool_kg_s svarer til peak_eff (cooling)
    """

    heatPumpList: List[HeatPump]
    brine: HeatCarrier  # nødvendig for q (rho) og mdot (cp)

    peak_heating_h: float = 4.0
    peak_cooling_h: Optional[float] = 4.0  # None => cooling peak pulse disabled

    peak_fraction_heating_mode: PeakSupplyMode = "incremental"
    peak_fraction_heating: float = 1.0

    peak_fraction_cooling_mode: PeakSupplyMode = "incremental"
    peak_fraction_cooling: float = 1.0

    # computed
    n_heat_pumps: int = field(init=False)
    diversity_factor: float = field(init=False)

    heating_ground_load_W: np.ndarray = field(init=False)            # (3,)
    cooling_ground_load_W: Optional[np.ndarray] = field(init=False)  # (3,) or None

    aggregated_q_peak_heat_m3_s: float = field(init=False)
    aggregated_mdot_peak_heat_kg_s: float = field(init=False)

    aggregated_q_peak_cool_m3_s: Optional[float] = field(init=False)
    aggregated_mdot_peak_cool_kg_s: Optional[float] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.heatPumpList)
        if n <= 0:
            raise ValueError("HeatPumps requires at least one HeatPump")
        object.__setattr__(self, "n_heat_pumps", n)

        # validate durations
        if self.peak_heating_h <= 0:
            raise ValueError("peak_heating_h must be > 0")
        if self.peak_cooling_h is not None and self.peak_cooling_h <= 0:
            raise ValueError("peak_cooling_h must be > 0 when provided")

        # validate fractions
        if not (0.0 <= self.peak_fraction_heating <= 1.0):
            raise ValueError("peak_fraction_heating must be in [0,1]")
        if not (0.0 <= self.peak_fraction_cooling <= 1.0):
            raise ValueError("peak_fraction_cooling must be in [0,1]")

        if self.peak_fraction_heating_mode not in ("absolute", "incremental"):
            raise ValueError("peak_fraction_heating_mode must be 'absolute' or 'incremental'")
        if self.peak_fraction_cooling_mode not in ("absolute", "incremental"):
            raise ValueError("peak_fraction_cooling_mode must be 'absolute' or 'incremental'")

        # validate brine props
        rho = float(self.brine.rho)
        cp = float(self.brine.c)
        if rho <= 0 or cp <= 0:
            raise ValueError(f"brine.rho and brine.c must be > 0. Got rho={rho}, c={cp}.")

        f = float(diversity_factor_from_n_heat_pumps(n))
        if not (0.0 < f <= 1.0):
            raise ValueError(f"diversity_factor must be in (0,1]. Got {f}.")
        object.__setattr__(self, "diversity_factor", f)

        # -----------------------
        # HEATING loads (W): [year, winter, peak_eff]
        # -----------------------
        P_year = float(sum(hp.annualHeating_ground_load for hp in self.heatPumpList))
        P_wint = float(sum(hp.winterHeating_ground_load for hp in self.heatPumpList))
        P_peak_raw = float(sum(hp.peakHeating_ground_load for hp in self.heatPumpList))  # raw population peak

        P_peak_div = f * P_peak_raw
        P_peak_eff = self._apply_peak_fraction(
            P_base=P_wint,
            P_peak=P_peak_div,
            mode=self.peak_fraction_heating_mode,
            alpha=self.peak_fraction_heating,
        )

        object.__setattr__(
            self,
            "heating_ground_load_W",
            np.asarray([P_year, P_wint, P_peak_eff], dtype=float),
        )

        # -----------------------
        # HEATING aggregated peak flows (match peak_eff)
        # -----------------------
        # Sum of individual (raw) peak flows (no diversity, no peak fraction)
        sum_q_peak_raw = 0.0
        sum_mdot_peak_raw = 0.0
        for hp in self.heatPumpList:
            dT = float(hp.deltaTHeating)
            if dT <= 0:
                raise ValueError(f"HeatPump ID={hp.ID}: deltaTHeating must be > 0. Got {dT}.")
            P = float(hp.peakHeating_ground_load)  # W (positive)
            sum_q_peak_raw += P / (dT * rho * cp)     # m3/s
            sum_mdot_peak_raw += P / (dT * cp)        # kg/s
        print(rho,cp)
        # Apply diversity to flows (since raw sums correspond to raw peak power)
        q_peak_div = f * sum_q_peak_raw
        mdot_peak_div = f * sum_mdot_peak_raw

        # Apply peak fraction scaling to reach peak_eff
        scale_H = 0.0 if P_peak_div == 0.0 else (P_peak_eff / P_peak_div)
        object.__setattr__(self, "aggregated_q_peak_heat_m3_s", float(scale_H * q_peak_div))
        object.__setattr__(self, "aggregated_mdot_peak_heat_kg_s", float(scale_H * mdot_peak_div))

        # -----------------------
        # COOLING (optional)
        # -----------------------
        has_cooling = any(
            (hp.peakCoolingLoad is not None and hp.peakCoolingLoad > 0)
            for hp in self.heatPumpList
        )
        if not has_cooling:
            object.__setattr__(self, "cooling_ground_load_W", None)
            object.__setattr__(self, "aggregated_q_peak_cool_m3_s", None)
            object.__setattr__(self, "aggregated_mdot_peak_cool_kg_s", None)
            return

        # NOTE: I mange datasæt er cooling ground load negativ (injektion).
        # Vi aggregerer som magnituder for flows.
        P_year_c = float(sum(self._mag(hp.annualCooling_ground_load) for hp in self.heatPumpList))
        P_summ_c = float(sum(self._mag(hp.summerCooling_ground_load) for hp in self.heatPumpList))
        P_peak_c_raw = float(sum(self._mag(hp.peakCooling_ground_load) for hp in self.heatPumpList))

        P_peak_c_div = f * P_peak_c_raw

        if self.peak_cooling_h is None:
            # Policy: peak pulse disabled -> peak level = season level
            P_peak_c_eff = P_summ_c
        else:
            P_peak_c_eff = self._apply_peak_fraction(
                P_base=P_summ_c,
                P_peak=P_peak_c_div,
                mode=self.peak_fraction_cooling_mode,
                alpha=self.peak_fraction_cooling,
            )

        object.__setattr__(
            self,
            "cooling_ground_load_W",
            np.asarray([P_year_c, P_summ_c, P_peak_c_eff], dtype=float),
        )

        # Cooling aggregated flows (match peak_c_eff)
        sum_q_peak_c_raw = 0.0
        sum_mdot_peak_c_raw = 0.0
        for hp in self.heatPumpList:
            dT = float(hp.deltaTCooling)
            if dT <= 0:
                # hvis en HP ikke har cooling, kan deltaTCooling være 0/None i nogle datasæt
                # i så fald bør du sikre upstream at de er sat meningsfuldt for cooling-HP'er
                continue
            Pmag = self._mag(float(hp.peakCooling_ground_load))
            if Pmag == 0.0:
                continue
            sum_q_peak_c_raw += Pmag / (dT * rho * cp)
            sum_mdot_peak_c_raw += Pmag / (dT * cp)

        q_peak_c_div = f * sum_q_peak_c_raw
        mdot_peak_c_div = f * sum_mdot_peak_c_raw

        scale_C = 0.0 if P_peak_c_div == 0.0 else (P_peak_c_eff / P_peak_c_div)
        object.__setattr__(self, "aggregated_q_peak_cool_m3_s", float(scale_C * q_peak_c_div))
        object.__setattr__(self, "aggregated_mdot_peak_cool_kg_s", float(scale_C * mdot_peak_c_div))

    @staticmethod
    def _apply_peak_fraction(*, P_base: float, P_peak: float, mode: PeakSupplyMode, alpha: float) -> float:
        if mode == "absolute":
            return float(alpha * P_peak)
        # incremental
        return float(P_base + alpha * (P_peak - P_base))

    @staticmethod
    def _mag(x: float) -> float:
        # magnitude helper (handles negative cooling loads)
        return float(abs(x))
