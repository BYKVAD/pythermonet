from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pythermonet.components import (
    BrineTemperatureLimits,
    DistributionNetworkParameters,
    HeatPumpPeakSupplyParameters,
    HHEFieldParameters,
    build_distribution_network,
    build_hhe_field,
    ground_loads_from_heat_pumps,
)
from pythermonet.core import HeatCarrier, Material, PipeSegmentParameters, Soil
from pythermonet.dimensioning import (
    HHEGroundField,
    SizingParameters,
    run_hhe_sizing_workflow,
    run_pipedimensioning,
)
from pythermonet.input import (
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
# 3) HHE field  (20 parallel pipes = 10 loops, horizontal, 1.2 m burial depth)
# -----------------------------------------------------------------------------
pipe_material_hhe = pipe_material_dist

pipe_segment_parameters_hhe = PipeSegmentParameters(
    diameter_outer=0.040,   # 40 mm OD
    sdr=17.0,
    roughness=1e-6,
)

hhe_field_parameters = HHEFieldParameters(
    n_pipes_parallel=20,      # 10 loops (outgoing + return)
    pipe_spacing=1.5,        # lateral spacing between pipes [m]
    burial_depth=1.2,        # m
    length_element=100.0,    # placeholder — sized by the HHE bisection solver
)

pipe_infrastructure_hhe = build_hhe_field(
    segment_parameters=pipe_segment_parameters_hhe,
    field_parameters=hhe_field_parameters,
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
