from dataclasses import dataclass

@dataclass
class Climate:
    degree_days_fraction_winter_heating: float
    degree_days_fraction_summer_cooling: float
