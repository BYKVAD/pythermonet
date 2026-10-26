import json

from pythermonet.components.heat_pump import HeatPump



def read_file(path):
    with open(path) as f:
        file = json.load(f)

    if file == None:
        return "not a file"

    return file


def fast_load_to_hps(file_path, annual_cop, winter_cop, energy_effiency_ratio, peak_demand_cop):
    
    fast_load_res = read_file(file_path)

    HPs = []
    for heat_load in fast_load_res:
        id = heat_load["id"]
        sim = heat_load["result"]

        annual_heat_load = (sim["heating"]*1000)/8760
        annual_cool_load = (sim["cooling"]*1000)/8760
        
        winter_heat_load = (sim["winter_heating_load"]*1000) / sim["winter_months"]*24
        summer_cool_load = (sim["winter_heating_load"]*1000) / sim["winter_months"]*24

        daily_heat_load = (sim["daily_peak_heating_load"]*1000) / 24
        daily_cool_load = (sim["daily_peak_cooling_load"]*1000) / 24

        HPs.append(HeatPump(
            ID=id,
            annualHeatingLoad=annual_heat_load,
            annualCoolingLoad=annual_cool_load,
            winterHeatingLoad=winter_heat_load,
            summerCoolingLoad=summer_cool_load,
            peakHeatingLoad=daily_heat_load,
            peakCoolingLoad=daily_cool_load,
            annualSCOP=annual_cop,
            winterSCOP=winter_cop,
            EER=energy_effiency_ratio,
            peakCOP=peak_demand_cop,
            deltaTCooling=4,
            deltaTHeating=5
        ))

    return HPs
