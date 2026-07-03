from dataclasses import dataclass, fields
from typing import NamedTuple


from pythermonet.components.heat_pump import HeatPump

season_periods = {
    "winter": [1, 2, 3, 4, 11, 12],
    "summer": [5, 6, 7, 8, 9, 10]
}

@dataclass(frozen=True)
class Loads():
    winter_load: float
    summer_load: float


def period_loads(hps: list[HeatPump]) -> Loads:
    yearly_load = sum(item.annualHeatingLoad for item in hps)
    winter_load = sum(item.winterHeatingLoad for item in hps)

    winter_load = winter_load / (12 / len(season_periods["winter"]))

    # monthly_summer_load = loads.summer_load / ( 12 / len(season_periods["summer"]))

    # TODO Current assumption is two season, should be dynamic 
    # and handle more detailed seasons
    non_winter_load = yearly_load-winter_load
    return Loads(winter_load, non_winter_load)


def get_monthly_demand(hps: list[HeatPump]) -> list[float]:
    loads = period_loads(hps)

    season_to_load = {}
    months = [None] * 12

    for field in fields(loads):
        if "summer" in field.name:
            season_to_load["summer"] = getattr(loads, field.name)
        elif "winter" in field.name:
            season_to_load["winter"] = getattr(loads, field.name)

    for season in season_periods:
        seasonal_months = season_periods[season]

        for months_nr in seasonal_months:
            # get load based on season
            season_load = season_to_load[season]

            # divide load by len(seasonal_months)
            monthly_load = season_load / len(seasonal_months)

            # add load to months by months_nr - 1
            months[months_nr-1] = monthly_load
            
    return months
    

def monthly_elec_loads(hps: list[HeatPump], cops: list[float]):
    loads = get_monthly_demand(hps)

    monthly_heat_distribution = [load * 8760.0 / 1000000.0 for load in loads]

    elec_demand = []
    for index in range(0,len(monthly_heat_distribution)):
        elec_demand.append(monthly_heat_distribution[index] / cops[index])
    
    return elec_demand

