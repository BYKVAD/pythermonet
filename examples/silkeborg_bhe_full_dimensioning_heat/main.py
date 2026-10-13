from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pythermonet.components import (
    BrineTemperatureLimits,
    DistributionNetworkParameters,
    HeatPumpPeakSupplyParameters,
    VHEFieldParameters,
    build_distribution_network,
    build_vhe_field,
    ground_loads_from_heat_pumps,
)
from pythermonet.core import Annulus, HeatCarrier, Material, PipeSegmentParameters, Soil
from pythermonet.dimensioning import (
    SizingParameters,
    run_bhe_sizing_workflow,
    run_pipedimensioning,
)
from pythermonet.input import (
    read_heat_pumps_tsv,
    read_pipe_catalog,
    read_undimensioned_topology_tsv,
)
from pythermonet.output import (
    print_bhe_results,
    save_bhe_results_json,
    save_bhe_results_txt,
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

heat_pump_file = PROJECT_DIR / "data/silkeborg_heat_pump_heat_high_cool.dat"
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
network_parameters = DistributionNetworkParameters(
    roughness=1e-6,
    burial_depth=1.2,
    pipe_spacing=0.3,
    n_pipes_parallel=2,
)

topology = read_undimensioned_topology_tsv(topology_file)
distribution_network_undimensioned = build_distribution_network(
    topology,
    pipe_material=pipe_material_dist,
    network_parameters=network_parameters,
)

# -----------------------------------------------------------------------------
# 3) Borehole field (only identical boreholes supported for now)
# -----------------------------------------------------------------------------
n_boreholes = 6
spacing_m = 15.0
coordinates = [[0.0, i * spacing_m] for i in range(n_boreholes)]
grout = Material(density=1500, specific_heat=2e3, thermal_conductivity=1.75)
pipe_material_bhe = Material(density=1000, specific_heat=2e3, thermal_conductivity=0.4)
borehole = Annulus(diameter_outer=0.152, sdr=1000.0)

pipe_segment_parameters_bhe = PipeSegmentParameters(
    diameter_outer=0.04,
    sdr=11.0,
    roughness=1e-6,
)

vhe_field_parameters = VHEFieldParameters(
    shank_spacing=0.015 + 2 * 0.02,
    burial_depth=1.0,
    tilt_rad=0.0,
    orientation_rad=0.0,
    heat_exchanger_type="1U",
)

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

peak_supply = HeatPumpPeakSupplyParameters(
    peak_hours_heating=4.0,
    peak_fraction_heating_mode="incremental",
    peak_fraction_heating=1.0,
    peak_hours_cooling=4.0,
    peak_fraction_cooling_mode="incremental",
    peak_fraction_cooling=1.0,
)

loads = ground_loads_from_heat_pumps(
    hp_list,
    brine=brine,
    peak_supply=peak_supply,
)

sizing = SizingParameters(thermal_dimensioning_lifetime=30.0)

# -----------------------------------------------------------------------------
# 5) Brine temperature limits
# -----------------------------------------------------------------------------
brine_temperature_limits = BrineTemperatureLimits(
    temperature_brine_min_heating=-3.0,
    temperature_brine_max_cooling=20.0,
)

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

out_dir = PROJECT_DIR / "out"
out_dir.mkdir(exist_ok=True)
save_bhe_results_txt(result, out_dir / "results.txt")
save_bhe_results_json(result, out_dir / "results.json")
