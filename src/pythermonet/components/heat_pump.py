from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class HeatPump:
    ID: int

    # Loads [W] (sink-side: heating delivered, cooling delivered)
    annualHeatingLoad: float
    winterHeatingLoad: float
    peakHeatingLoad: float

    annualSCOP: float
    winterSCOP: float
    peakCOP: float

    deltaTHeating: float  # [K] brine ΔT in heating (local HP assumption)

    annualCoolingLoad: float
    summerCoolingLoad: float
    peakCoolingLoad: float

    EER: float
    deltaTCooling: float  # [K] brine ΔT in cooling (local HP assumption)

    # --- computed once, used everywhere ---
    annualHeating_ground_load: float = field(init=False)   # [W], + extracted from ground
    winterHeating_ground_load: float = field(init=False)
    peakHeating_ground_load: float = field(init=False)

    annualCooling_ground_load: float = field(init=False)   # [W], + rejected to ground (active cooling convention)
    summerCooling_ground_load: float = field(init=False)
    peakCooling_ground_load: float = field(init=False)

    def __post_init__(self) -> None:
        # --- basic validation (fail fast) ---
        if self.annualHeatingLoad < 0 or self.winterHeatingLoad < 0 or self.peakHeatingLoad < 0:
            raise ValueError("Heating loads must be >= 0")

        if self.annualSCOP <= 0 or self.winterSCOP <= 0 or self.peakCOP <= 0:
            raise ValueError("COP/SCOP must be > 0")

        if self.deltaTHeating <= 0:
            raise ValueError("deltaTHeating must be > 0")

        if self.annualCoolingLoad < 0 or self.summerCoolingLoad < 0 or self.peakCoolingLoad < 0:
            raise ValueError("Cooling loads must be >= 0")

        has_cooling = self.peakCoolingLoad > 0

        if has_cooling:
            if self.EER <= 0:
                raise ValueError("EER must be > 0 when peakCoolingLoad > 0")
            if self.deltaTCooling <= 0:
                raise ValueError("deltaTCooling must be > 0 when peakCoolingLoad > 0")

        # 1) Ground loads (heating)
        # Antagelse: HeatingLoad er leveret til bygning (sink-side).
        # El-forbrug = P_load / COP
        # Ground extraction = P_load - P_el = P_load*(1 - 1/COP)
        object.__setattr__(self, "peakHeating_ground_load",   self.peakHeatingLoad   * (1.0 - 1.0 / self.peakCOP))
        object.__setattr__(self, "winterHeating_ground_load", self.winterHeatingLoad * (1.0 - 1.0 / self.winterSCOP))
        object.__setattr__(self, "annualHeating_ground_load", self.annualHeatingLoad * (1.0 - 1.0 / self.annualSCOP))

        # 2) Ground loads (cooling) – aktiv køling konvention
        # Hvis CoolingLoad er køleeffekt leveret (fjernet fra bygning),
        # så er varmeafgivelse til jord: P_reject = P_cool + P_el = P_cool*(1 + 1/EER)
        if has_cooling:
            object.__setattr__(self, "peakCooling_ground_load",   self.peakCoolingLoad   * (1.0 + 1.0 / self.EER))
            object.__setattr__(self, "summerCooling_ground_load", self.summerCoolingLoad * (1.0 + 1.0 / self.EER))
            object.__setattr__(self, "annualCooling_ground_load", self.annualCoolingLoad * (1.0 + 1.0 / self.EER))
        else:
            object.__setattr__(self, "peakCooling_ground_load",   0.0)
            object.__setattr__(self, "summerCooling_ground_load", 0.0)
            object.__setattr__(self, "annualCooling_ground_load", 0.0)