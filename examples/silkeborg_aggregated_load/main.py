from __future__ import annotations

from pathlib import Path

from pythermonet.components import (
    AggregatedLoadPeakSupplyParameters,
    BrineTemperatureLimits,
    VHEField,
    ground_loads_from_aggregated_load,
)
from pythermonet.core import Annulus, HeatCarrier, Material, PipeSegment, Soil
from pythermonet.dimensioning import SizingParameters, run_bhe_sizing_workflow
from pythermonet.input import (
    read_aggregated_load_tsv,
    read_dimensioned_topology_tsv_to_hydraulic,
)
from pythermonet.output import print_bhe_results

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
    temperature_surface_mean=9.03,
    temperature_surface_amplitude=7.9,
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
borehole = Annulus(diameter_outer=0.152, sdr=1000.0)
upipe    = PipeSegment(
    diameter_outer=0.04,
    sdr=11.0,
    material=pipe_material_bhe,
    roughness=1e-6,
    id_=0,
    length=100,  # placeholder
)

BHEfield = VHEField(
    id_=1,
    heat_exchanger_type="1U",
    pipe=upipe,
    borehole=borehole,
    grout=grout,
    coordinates=coordinates,
    shank_spacing=0.015 + 2 * 0.02,
    length_borehole=120.0,
    burial_depth=0,
    tilt_rad=0.0,
    orientation_rad=0.0,
)

# -----------------------------------------------------------------------------
# 4) Aggregated heat pump loads (mode-specific)
# -----------------------------------------------------------------------------
agg_load_input = read_aggregated_load_tsv(agg_load_file)

peak_supply = AggregatedLoadPeakSupplyParameters(
    peak_hours_heating=4.0,
    f_peak_heating=1.0,
    peak_hours_cooling=4.0,
    f_peak_cooling=1.0,
)

loads = ground_loads_from_aggregated_load(
    agg_load_input,
    brine=brine,
    peak_supply=peak_supply,
)

sizing = SizingParameters(thermal_dimensioning_lifetime=30.0)

# -----------------------------------------------------------------------------
# 5) Brine temperature limits
# -----------------------------------------------------------------------------
brine_temperature_limits = BrineTemperatureLimits(
    temperature_brine_min_heating=-3.0,
    temperature_brine_max_cooling=25.0,
)

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
    brine_temperature_limits=brine_temperature_limits,
)
print_bhe_results(result)
