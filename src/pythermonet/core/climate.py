from dataclasses import dataclass

@dataclass
class Climate:
    heating_degree_days_fraction_winter: float
    cooling_degree_days_fraction_summer: float
