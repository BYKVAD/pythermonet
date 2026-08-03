from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pythermonet.components.ground_loads import ground_loads_from_heat_pumps
from pythermonet.core.material import Material
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.input.read_pipe_catalogue import read_pipe_catalogue
from pythermonet.input.read_topology import read_undimensioned_topology_tsv_to_network
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv
from pythermonet.dimensioning.hydraulic_dimensioning import run_pipedimensioning

from pythermonet.components.vhe_field import VHEField
from pythermonet.core.annulus import Annulus
from pythermonet.core.pipe_segment import PipeSegment

from pythermonet.dimensioning.BHE.bhe_workflow import run_bhe_sizing_workflow, print_bhe_results
from pythermonet.dimensioning.sizing_parameters import SizingParameters

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[2]

pipe_catalogue_file = REPO_ROOT / "PythermonetII/src/pythermonet/resources/pipe_catalogue.csv"
heat_pump_file = PROJECT_DIR / "data/silkeborg_heat_pump_heat_high_cool.dat"
topology_file = PROJECT_DIR / "data/silkeborg_topology.dat"


# -----------------------------------------------------------------------------
# 1) Read pipe catalogue, define materials + fluids + soil
# -----------------------------------------------------------------------------
pipe_catalogue = read_pipe_catalogue(pipe_catalogue_file)

pipe_material_dist = Material(
    density=975,
    specific_heat=1900,
    thermal_conductivity=0.4,
)

brine = HeatCarrier(
    density=965,
    specific_heat=4450,
    thermal_conductivity=0.45,
    dynamic_viscosity=5e-3,
)

soil = Soil(
    density=2500,
    specific_heat=1000,
    thermal_conductivity=2.36,
    thermal_conductivity_shallow_heating=1.25,
    thermal_conductivity_shallow_cooling=1.25,
    geothermal_heat_flux=0.0185,
    temperature_surface_mean=9.03,
    temperature_surface_amplitude=7.9,
)

# -----------------------------------------------------------------------------
# 2) Distribution network
# -----------------------------------------------------------------------------
distribution_network_undimensioned = read_undimensioned_topology_tsv_to_network(
    topology_file,
    pipe_material=pipe_material_dist,
    roughness=1e-6,
    burial_depth=1.2,
    pipe_distance=0.3,
    n_parallel_pipes=2,
)

# -----------------------------------------------------------------------------
# 3) Borehole field (only identical boreholes supported for now)
# -----------------------------------------------------------------------------
n_boreholes = 6
spacing_m = 15.0
coordinates = [[0.0, i * spacing_m] for i in range(n_boreholes)]
borehole_diameter_m = 0.152
u_pipe_outer_diameter_m = 0.04
u_pipe_sdr = 11.0
grout = Material(density=1500, specific_heat=2e3, thermal_conductivity=1.75)
pipe_material_bhe = Material(density=1000, specific_heat=2e3, thermal_conductivity=0.4)
borehole = Annulus(diameter_outer=borehole_diameter_m, sdr=1000.0)

upipe = PipeSegment(
    diameter_outer=u_pipe_outer_diameter_m,
    sdr=u_pipe_sdr,
    material=pipe_material_bhe,
    roughness=1e-6,
    id_=0,
    length=100,  # placeholder
)

BHEfield = VHEField(
    id_=1,
    heat_exchanger_type='1U',
    pipe=upipe,
    borehole=borehole,
    grout=grout,
    coordinates=coordinates,
    shank_spacing=0.015 + 2 * 0.02,
    length_borehole=120.0,
    burial_depth=1.0,
    tilt_rad=0.0,
    orientation_rad=0.0,
)

# -----------------------------------------------------------------------------
# 4) Heat pumps
# -----------------------------------------------------------------------------
hp_list = read_heat_pumps_tsv(path=heat_pump_file)

loads = ground_loads_from_heat_pumps(
    hp_list,
    brine,
    peak_hours_heating=4.0,
    peak_fraction_heating_mode="incremental",
    peak_fraction_heating=1.0,
    peak_hours_cooling=4.0,
    peak_fraction_cooling_mode="incremental",
    peak_fraction_cooling=1.0,
)

sizing = SizingParameters(thermal_dimensioning_lifetime=30.0)

# -----------------------------------------------------------------------------
# 5) Brine temperature limits
# -----------------------------------------------------------------------------
T_BRINE_MIN_HEAT = -3.0   # HP evaporator inlet limit [°C]
T_BRINE_MAX_COOL = 20.0   # HP condenser inlet limit [°C]

# -----------------------------------------------------------------------------
# 6) Hydraulic pipe network sizing (mode-specific)
# -----------------------------------------------------------------------------
hydraulic = run_pipedimensioning(
    pipe_catalogue,
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
    T_brine_min_heat=T_BRINE_MIN_HEAT,
    T_brine_max_cool=T_BRINE_MAX_COOL,
)
print_bhe_results(result)
