from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pythermonet.components import (
    build_distribution_network,
    build_vhe_field,
    ground_loads_from_heat_pumps,
    localize_borefield_coordinates,
)
from pythermonet.dimensioning import run_bhe_sizing_workflow, run_pipedimensioning
from pythermonet.input import (
    load_settings,
    read_borefield_coordinates_tsv,
    read_heat_pumps_tsv,
    read_pipe_catalog,
    read_undimensioned_topology_tsv,
)
from pythermonet.output import print_bhe_results
from pythermonet.resources import SETTINGS_TEMPLATE_BHE_PATH

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

heat_pump_file = PROJECT_DIR / "data/silkeborg_heat_pump_heat_cool.dat"
topology_file = PROJECT_DIR / "data/silkeborg_topology.dat"
settings_file = SETTINGS_TEMPLATE_BHE_PATH
borefield_coordinates_file = PROJECT_DIR / "data/silkeborg_borefield_coordinates.dat"


# -----------------------------------------------------------------------------
# 1) Read pipe catalog, load materials + fluid + soil from the settings file
# -----------------------------------------------------------------------------
# Same values as silkeborg_bhe_full_dimensioning_heat, but sourced from a
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
# 3) Borehole field (only identical boreholes supported for now)
# -----------------------------------------------------------------------------
# Coordinates come from a real-world borefield layout (WKT/EWKT, EPSG:25832),
# e.g. as handed off from Qthermonet/PostGIS — see
# pythermonet.input.read_borefield_coordinates_tsv and
# pythermonet.components.localize_borefield_coordinates.
borefield_input = read_borefield_coordinates_tsv(borefield_coordinates_file)
coordinates = localize_borefield_coordinates(borefield_input)
grout = settings["grout"]
pipe_material_bhe = settings["pipe_material_bhe"]
borehole = settings["borehole"]
pipe_segment_parameters_bhe = settings["pipe_segment_bhe"]
vhe_field_parameters = settings["vhe_field_parameters"]

BHEfield = build_vhe_field(
    segment_parameters=pipe_segment_parameters_bhe,
    field_parameters=vhe_field_parameters,
    pipe_material=pipe_material_bhe,
    borehole=borehole,
    grout=grout,
    coordinates=coordinates,
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
# 8) BHE sizing workflow + results
# -----------------------------------------------------------------------------
result = run_bhe_sizing_workflow(
    ground_loads=loads,
    vhe_field=BHEfield,
    hydraulic=hydraulic,
    brine=brine,
    soil=soil,
    sizing=sizing,
    brine_temperature_limits=brine_temperature_limits,
)
print_bhe_results(result)
