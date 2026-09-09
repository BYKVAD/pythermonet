from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pythermonet.components.ground_loads import ground_loads_from_heat_pumps
from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.core.material import Material
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.ground_field import HHEGroundField
from pythermonet.dimensioning.HHE.hhe_workflow import run_hhe_sizing_workflow, print_hhe_results
from pythermonet.dimensioning.hydraulic_dimensioning import run_pipedimensioning
from pythermonet.dimensioning.sizing_parameters import SizingParameters
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv
from pythermonet.input.read_pipe_catalog import read_pipe_catalog
from pythermonet.input.read_topology import read_undimensioned_topology_tsv_to_network

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

heat_pump_file = PROJECT_DIR / "data/silkeborg_heat_pump_heat_only.dat"
topology_file = PROJECT_DIR / "data/silkeborg_topology.dat"

# -----------------------------------------------------------------------------
# 1) Read pipe catalog, define materials + fluids + soil
# -----------------------------------------------------------------------------
pipe_catalog = read_pipe_catalog()

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
    thermal_conductivity=1.25,
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
# 3) HHE field  (20 parallel pipes = 10 loops, horizontal, 1.2 m burial depth)
# -----------------------------------------------------------------------------
hhe_pipe_material = pipe_material_dist

hhe_segment = PipeSegment(
    diameter_outer=0.040,   # 40 mm OD
    sdr=17.0,
    material=hhe_pipe_material,
    roughness=1e-6,
    id_=0,
    length=100.0,           # placeholder — will be sized
)

pipe_infrastructure = PipeInfrastructure(
    n_pipes_parallel=20,      # 10 loops (outgoing + return)
    segments_trace=[hhe_segment],
    pipe_spacing=1.5,        # lateral spacing between pipes [m]
    burial_depth=1.2,        # m
)

hhe_field = HHEGroundField(
    pipe_infrastructure=pipe_infrastructure,
    soil_thermal_conductivity_heating=float(soil.thermal_conductivity_shallow_heating),
    soil_thermal_conductivity_cooling=float(soil.thermal_conductivity_shallow_cooling),
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
    hhe_field=hhe_field,
    pipe_infrastructure=pipe_infrastructure,
    hydraulic=hydraulic,
    brine=brine,
    soil=soil,
    sizing=sizing,
    T_brine_min_heat=T_BRINE_MIN_HEAT,
    T_brine_max_cool=T_BRINE_MAX_COOL,
)
print_hhe_results(result, pipe_infrastructure)
