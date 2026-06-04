from __future__ import annotations

from pathlib import Path

from pythermonet.core.material import Material
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.components.vhe_field import VHEField
from pythermonet.core.annulus import Annulus
from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.components.ground_loads import ground_loads_from_district
from pythermonet.input.read_aggregated_load import read_aggregated_load_tsv
from pythermonet.input.read_dimensioned_topology import read_dimensioned_topology_tsv_to_hydraulic
from pythermonet.dimensioning.BHE.bhe_workflow import run_bhe_sizing_workflow, print_bhe_results
from pythermonet.dimensioning.sizing_parameters import SizingParameters

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
agg_load_file = PROJECT_DIR / "data/silkeborg_aggregated_load_heat.dat"
topology_file = PROJECT_DIR / "data/silkeborg_topology_dimensioned_heat.dat"

# -----------------------------------------------------------------------------
# 1) Materials, brine, soil
# -----------------------------------------------------------------------------
pipe_material = Material(density=975, specific_heat=1900, thermal_conductivity=0.4)

brine = HeatCarrier(
    density=965,
    specific_heat=4450,
    thermal_conductivity=0.45,
    dynamic_viscosity=5e-3,
)

soil = Soil(
    density=2650,
    specific_heat=1000,
    thermal_conductivity=2.36,
    thermal_conductivity_shallow_heating=1.25,
    thermal_conductivity_shallow_cooling=1.25,
    geothermal_heat_flux=0.0185,
    surface_temperature=9.03,
    surface_temperature_amplitude=7.9,
)

# -----------------------------------------------------------------------------
# 2) Distribution network (pre-sized, no hydraulic dimensioning)
# -----------------------------------------------------------------------------
network, hydraulic = read_dimensioned_topology_tsv_to_hydraulic(
    topology_file,
    pipe_material=pipe_material,
    brine=brine,
    burial_depth=1.2,
    pipe_distance=0.3,
    n_parallel_pipes=2,
)

# -----------------------------------------------------------------------------
# 3) Borehole field
# -----------------------------------------------------------------------------
n_boreholes = 6
spacing_m   = 15.0
coordinates = [[0.0, i * spacing_m] for i in range(n_boreholes)]

pipe_material_bhe = Material(density=1000, specific_heat=2e3, thermal_conductivity=0.4)
grout    = Material(density=1500, specific_heat=2e3, thermal_conductivity=1.75)
borehole = Annulus(outer_diameter=0.152, sdr=1000.0)
upipe    = PipeSegment(
    outer_diameter=0.04,
    sdr=11.0,
    material=pipe_material_bhe,
    roughnessHeight=1e-6,
    ID=0,
    length=100,  # placeholder
)

BHEfield = VHEField(
    ID=1,
    HE="1U",
    pipe=upipe,
    borehole=borehole,
    grout=grout,
    coordinates=coordinates,
    shankSpacing=0.015 + 2 * 0.02,
    H_m=120.0,
    D_m=0,
    tilt_rad=0.0,
    orientation_rad=0.0,
)

# -----------------------------------------------------------------------------
# 4) Aggregated heat pump loads (mode-specific)
# -----------------------------------------------------------------------------
agg_load_input = read_aggregated_load_tsv(agg_load_file)

loads = ground_loads_from_district(
    agg_load_input,
    brine,
    f_peak_heating=1.0,
    f_peak_cooling=1.0,
    peak_heating_h=4.0,
    peak_cooling_h=4.0,
)

sizing = SizingParameters(time_horizon_years=30.0)

# -----------------------------------------------------------------------------
# 5) Brine temperature limits
# -----------------------------------------------------------------------------
T_BRINE_MIN_HEAT = -3.0   # HP evaporator inlet limit [°C]
T_BRINE_MAX_COOL = 25.0   # HP condenser inlet limit [°C]

# -----------------------------------------------------------------------------
# 6) BHE sizing workflow + results
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
