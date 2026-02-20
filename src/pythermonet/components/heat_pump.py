from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from pythermonet.core.heat_carrier import HeatCarrier


@dataclass
class HeatPump:
    ID: int

    # Loads [W] (vælg én enhed konsekvent – her antager jeg W)
    annualHeatingLoad: float
    winterHeatingLoad: float
    peakHeatingLoad: float

    annualSCOP: float
    winterSCOP: float
    peakCOP: float

    deltaTHeating: float  # [K] brine ΔT ved heating

    annualCoolingLoad: float
    summerCoolingLoad: float
    peakCoolingLoad: float

    EER: float
    deltaTCooling: float  # [K] brine ΔT ved cooling

    sourceHeatCarrier: HeatCarrier

    # --- computed once, used everywhere ---
    annualHeating_ground_load: float = field(init=False)   # [W], + = extracted from ground
    winterHeating_ground_load: float = field(init=False)
    peakHeating_ground_load: float = field(init=False)

    annualCooling_ground_load: float = field(init=False)   # [W], - = injected to ground (valgfrit fortegn)
    summerCooling_ground_load: float = field(init=False)
    peakCooling_ground_load: float = field(init=False)

    q_peak_heat_m3_s: float = field(init=False)
    q_peak_cool_m3_s: Optional[float] = field(init=False)

    mdot_peak_heat_kg_s: float = field(init=False)
    mdot_peak_cool_kg_s: Optional[float] = field(init=False)

    def __post_init__(self) -> None:
        # --- basic validation (fail fast) ---
        if self.deltaTHeating <= 0:
            raise ValueError("deltaTHeating must be > 0")
        if self.peakCOP <= 0:
            raise ValueError("peakCOP must be > 0")
        if self.sourceHeatCarrier.rho <= 0 or self.sourceHeatCarrier.c <= 0:
            raise ValueError("HeatCarrier rho and c must be > 0")

        # 1) Ground loads (heating)
        # Antagelse: peakHeatingLoad er leveret til bygning (sink-side).
        # El-forbrug = P_load / COP
        # Ground extraction = P_load - P_el = P_load*(1 - 1/COP)
        self.peakHeating_ground_load = self.peakHeatingLoad * (1.0 - 1.0 / self.peakCOP)
        self.winterHeating_ground_load = self.winterHeatingLoad * (1.0 - 1.0 / self.winterSCOP)
        self.annualHeating_ground_load = self.annualHeatingLoad * (1.0 - 1.0 / self.annualSCOP)

        # 2) Ground loads (cooling)
        # Her SKAL du definere konvention:
        # - Hvis peakCoolingLoad er "køleeffekt leveret" (fjernet fra bygning),
        #   så er varmeafgivelsen til jord typisk: P_reject = P_cool*(1 + 1/EER) (for aktiv køling)
        #   For passiv køling er det anderledes.
        #
        # Jeg implementerer en simpel aktiv-køling konvention:
        if self.peakCoolingLoad > 0:
            if self.EER <= 0:
                raise ValueError("EER must be > 0 when peakCoolingLoad > 0")
            if self.deltaTCooling <= 0:
                raise ValueError("deltaTCooling must be > 0 when peakCoolingLoad > 0")

            self.peakCooling_ground_load = self.peakCoolingLoad * (1.0 + 1.0 / self.EER)
            self.summerCooling_ground_load = self.summerCoolingLoad * (1.0 + 1.0 / self.EER)
            self.annualCooling_ground_load = self.annualCoolingLoad * (1.0 + 1.0 / self.EER)
        else:
            self.peakCooling_ground_load = 0.0
            self.summerCooling_ground_load = 0.0
            self.annualCooling_ground_load = 0.0

        # 3) Convert ground loads to flow (heating peak)
        # m_dot = P_ground / (cp * ΔT), Q = m_dot / rho
        cp = self.sourceHeatCarrier.c
        rho = self.sourceHeatCarrier.rho

        self.mdot_peak_heat_kg_s = self.peakHeating_ground_load / (cp * self.deltaTHeating)
        self.q_peak_heat_m3_s = self.mdot_peak_heat_kg_s / rho

        if self.peakCoolingLoad > 0:
            self.mdot_peak_cool_kg_s = self.peakCooling_ground_load / (cp * self.deltaTCooling)
            self.q_peak_cool_m3_s = self.mdot_peak_cool_kg_s / rho
        else:
            self.mdot_peak_cool_kg_s = None
            self.q_peak_cool_m3_s = None
