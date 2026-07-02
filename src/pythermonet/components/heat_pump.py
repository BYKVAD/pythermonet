from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class HeatPump:
    id_: int

    # Loads [W] (sink-side: heating delivered, cooling delivered)
    annual_load_heating: float
    winter_load_heating: float
    peak_load_heating: float

    annual_scop: float
    winter_scop: float
    peak_cop: float

    delta_temperature_heating: float  # [K] brine ΔT in heating (local HP assumption)

    annual_load_cooling: float
    summer_load_cooling: float
    peak_load_cooling: float

    eer: float
    delta_temperature_cooling: float  # [K] brine ΔT in cooling (local HP assumption)

    # --- computed once, used everywhere ---
    annual_ground_load_heating: float = field(init=False)   # [W], + extracted from ground
    winter_ground_load_heating: float = field(init=False)
    peak_ground_load_heating: float = field(init=False)

    annual_ground_load_cooling: float = field(init=False)   # [W], + rejected to ground (active cooling convention)
    summer_ground_load_cooling: float = field(init=False)
    peak_ground_load_cooling: float = field(init=False)

    def __post_init__(self) -> None:
        # --- basic validation (fail fast) ---
        if self.annual_load_heating < 0 or self.winter_load_heating < 0 or self.peak_load_heating < 0:
            raise ValueError("Heating loads must be >= 0")

        if self.annual_scop <= 0 or self.winter_scop <= 0 or self.peak_cop <= 0:
            raise ValueError("COP/SCOP must be > 0")

        if self.delta_temperature_heating <= 0:
            raise ValueError("delta_temperature_heating must be > 0")

        if self.annual_load_cooling < 0 or self.summer_load_cooling < 0 or self.peak_load_cooling < 0:
            raise ValueError("Cooling loads must be >= 0")

        has_cooling = self.peak_load_cooling > 0

        if has_cooling:
            if self.eer <= 0:
                raise ValueError("eer must be > 0 when peak_load_cooling > 0")
            if self.delta_temperature_cooling <= 0:
                raise ValueError("delta_temperature_cooling must be > 0 when peak_load_cooling > 0")

        # 1) Ground loads (heating)
        # Antagelse: HeatingLoad er leveret til bygning (sink-side).
        # El-forbrug = P_load / COP
        # Ground extraction = P_load - P_el = P_load*(1 - 1/COP)
        object.__setattr__(self, "peak_ground_load_heating",   self.peak_load_heating   * (1.0 - 1.0 / self.peak_cop))
        object.__setattr__(self, "winter_ground_load_heating", self.winter_load_heating * (1.0 - 1.0 / self.winter_scop))
        object.__setattr__(self, "annual_ground_load_heating", self.annual_load_heating * (1.0 - 1.0 / self.annual_scop))

        # 2) Ground loads (cooling) – aktiv køling konvention
        # Hvis CoolingLoad er køleeffekt leveret (fjernet fra bygning),
        # så er varmeafgivelse til jord: P_reject = P_cool + P_el = P_cool*(1 + 1/eer)
        if has_cooling:
            object.__setattr__(self, "peak_ground_load_cooling",   self.peak_load_cooling   * (1.0 + 1.0 / self.eer))
            object.__setattr__(self, "summer_ground_load_cooling", self.summer_load_cooling * (1.0 + 1.0 / self.eer))
            object.__setattr__(self, "annual_ground_load_cooling", self.annual_load_cooling * (1.0 + 1.0 / self.eer))
        else:
            object.__setattr__(self, "peak_ground_load_cooling",   0.0)
            object.__setattr__(self, "summer_ground_load_cooling", 0.0)
            object.__setattr__(self, "annual_ground_load_cooling", 0.0)
