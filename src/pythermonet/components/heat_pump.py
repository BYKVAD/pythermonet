from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class HeatPump:
    id_: int

    # Loads [W] (sink-side: heating delivered, cooling delivered)
    load_annual_heating: float
    load_winter_heating: float
    load_peak_heating: float

    scop_annual: float
    scop_winter: float
    cop_peak: float

    temperature_delta_heating: float  # [K] brine ΔT in heating (local HP assumption)

    load_annual_cooling: float
    load_summer_cooling: float
    load_peak_cooling: float

    eer: float
    temperature_delta_cooling: float  # [K] brine ΔT in cooling (local HP assumption)

    # --- computed once, used everywhere ---
    load_ground_annual_heating: float = field(init=False)   # [W], + extracted from ground
    load_ground_winter_heating: float = field(init=False)
    load_ground_peak_heating: float = field(init=False)

    load_ground_annual_cooling: float = field(init=False)   # [W], + rejected to ground (active cooling convention)
    load_ground_summer_cooling: float = field(init=False)
    load_ground_peak_cooling: float = field(init=False)

    def __post_init__(self) -> None:
        # --- basic validation (fail fast) ---
        if self.load_annual_heating < 0 or self.load_winter_heating < 0 or self.load_peak_heating < 0:
            raise ValueError("Heating loads must be >= 0")

        if self.scop_annual <= 0 or self.scop_winter <= 0 or self.cop_peak <= 0:
            raise ValueError("COP/SCOP must be > 0")

        if self.temperature_delta_heating <= 0:
            raise ValueError("temperature_delta_heating must be > 0")

        if self.load_annual_cooling < 0 or self.load_summer_cooling < 0 or self.load_peak_cooling < 0:
            raise ValueError("Cooling loads must be >= 0")

        has_cooling = self.load_peak_cooling > 0

        if has_cooling:
            if self.eer <= 0:
                raise ValueError("eer must be > 0 when load_peak_cooling > 0")
            if self.temperature_delta_cooling <= 0:
                raise ValueError("temperature_delta_cooling must be > 0 when load_peak_cooling > 0")

        # 1) Ground loads (heating)
        # Antagelse: HeatingLoad er leveret til bygning (sink-side).
        # El-forbrug = P_load / COP
        # Ground extraction = P_load - P_el = P_load*(1 - 1/COP)
        object.__setattr__(self, "load_ground_peak_heating",   self.load_peak_heating   * (1.0 - 1.0 / self.cop_peak))
        object.__setattr__(self, "load_ground_winter_heating", self.load_winter_heating * (1.0 - 1.0 / self.scop_winter))
        object.__setattr__(self, "load_ground_annual_heating", self.load_annual_heating * (1.0 - 1.0 / self.scop_annual))

        # 2) Ground loads (cooling) – aktiv køling konvention
        # Hvis CoolingLoad er køleeffekt leveret (fjernet fra bygning),
        # så er varmeafgivelse til jord: P_reject = P_cool + P_el = P_cool*(1 + 1/eer)
        if has_cooling:
            object.__setattr__(self, "load_ground_peak_cooling",   self.load_peak_cooling   * (1.0 + 1.0 / self.eer))
            object.__setattr__(self, "load_ground_summer_cooling", self.load_summer_cooling * (1.0 + 1.0 / self.eer))
            object.__setattr__(self, "load_ground_annual_cooling", self.load_annual_cooling * (1.0 + 1.0 / self.eer))
        else:
            object.__setattr__(self, "load_ground_peak_cooling",   0.0)
            object.__setattr__(self, "load_ground_summer_cooling", 0.0)
            object.__setattr__(self, "load_ground_annual_cooling", 0.0)


@dataclass
class BrineTemperatureLimits:
    """Allowed brine operating temperature limits.

    Parameters
    ----------
    temperature_brine_min_heating : float
        Minimum allowed brine temperature during heating — HP evaporator
        inlet limit [°C].
    temperature_brine_max_cooling : float
        Maximum allowed brine temperature during cooling — HP condenser
        inlet limit [°C].

    """

    temperature_brine_min_heating: float  # °C
    temperature_brine_max_cooling: float  # °C
