from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pythermonet.components import (
    build_distribution_network,
    build_pipe_infrastructure,
    ground_loads_from_heat_pumps,
)
from pythermonet.dimensioning import HHEGroundField, run_hhe_sizing_workflow, run_pipedimensioning
from pythermonet.input import (
    load_settings,
    read_heat_pumps_tsv,
    read_pipe_catalog,
    read_undimensioned_topology_tsv,
)
from pythermonet.output import print_hhe_results

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

heat_pump_file = PROJECT_DIR / "data/silkeborg_heat_pump_heat_only.dat"
topology_file = PROJECT_DIR / "data/silkeborg_topology.dat"
settings_file = PROJECT_DIR / "data/settings.json"


# -----------------------------------------------------------------------------
# 1) Read pipe catalog, load materials + fluid + soil from the settings file
# -----------------------------------------------------------------------------
# Same values as silkeborg_hhe_full_dimensioning_heat, but sourced from a
# settings file instead of hardcoded kwargs — see pythermonet.input.load_settings
# and pythermonet.output.save_settings.
pipe_catalog = read_pipe_catalog()

settings = load_settings(settings_file)
pipe_material_dist = settings["pipe_material_dist"]
brine = settings["brine"]
soil = settings["soil"]

# -----------------------------------------------------------------------------
# 2) Distribution network
# -----------------------------------------------------------------------------
network_parameters = settings["distribution_network_parameters"]

topology = read_undimensioned_topology_tsv(topology_file)
distribution_network_undimensioned = build_distribution_network(
    topology,
    pipe_material=pipe_material_dist,
    network_parameters=network_parameters,
)

# -----------------------------------------------------------------------------
# 3) HHE field  (20 parallel pipes = 10 loops, horizontal, 1.2 m burial depth)
# -----------------------------------------------------------------------------
pipe_material_hhe = settings["pipe_material_hhe"]
pipe_segment_parameters_hhe = settings["pipe_segment_hhe"]
pipe_infrastructure_parameters_hhe = settings["pipe_infrastructure_hhe"]

pipe_infrastructure_hhe = build_pipe_infrastructure(
    segment_parameters=pipe_segment_parameters_hhe,
    infrastructure_parameters=pipe_infrastructure_parameters_hhe,
    pipe_material=pipe_material_hhe,
)

ground_field_hhe = HHEGroundField(
    pipe_infrastructure=pipe_infrastructure_hhe,
    soil_thermal_conductivity_heating=float(soil.thermal_conductivity_shallow_heating),
    soil_thermal_conductivity_cooling=float(soil.thermal_conductivity_shallow_cooling),
)

# -----------------------------------------------------------------------------
# 4) Heat pumps
# -----------------------------------------------------------------------------
hp_list = read_heat_pumps_tsv(path=heat_pump_file)

peak_supply = settings["heat_pump_peak_supply_parameters"]

loads = ground_loads_from_heat_pumps(
    hp_list,
    brine=brine,
    peak_supply=peak_supply,
)

sizing = settings["sizing_parameters"]

# -----------------------------------------------------------------------------
# 5) Brine temperature limits
# -----------------------------------------------------------------------------
brine_temperature_limits = settings["brine_temperature_limits"]

# -----------------------------------------------------------------------------
# 6) Hydraulic pipe network sizing (mode-specific)
# -----------------------------------------------------------------------------
hydraulic = run_pipedimensioning(
    pipe_catalog,
    brine,
    distribution_network_undimensioned,
    hp_list,
)

# -----------------------------------------------------------------------------
# 7) HHE sizing workflow + results
# -----------------------------------------------------------------------------
result = run_hhe_sizing_workflow(
    ground_loads=loads,
    hhe_field=ground_field_hhe,
    pipe_infrastructure=pipe_infrastructure_hhe,
    hydraulic=hydraulic,
    brine=brine,
    soil=soil,
    sizing=sizing,
    brine_temperature_limits=brine_temperature_limits,
)
print_hhe_results(result, pipe_infrastructure_hhe)
