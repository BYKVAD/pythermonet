from __future__ import annotations

from pathlib import Path

from pythermonet.components.heat_pumps import HeatPumps
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

from pythermonet.physics.bhe_resistance import compute_rb_for_vhe_field
from pythermonet.simulation.run_distribution_pipe_thermal_model import compute_distribution_pipe_thermal_capacity, print_pipe_thermal_table
import numpy as np

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[2]

pipe_catalogue_file = REPO_ROOT / "PythermonetII/src/pythermonet/resources/pipe_catalogue.csv"
heat_pump_file = PROJECT_DIR / "data/silkeborg_heat_pump_heat.dat"
topology_file = PROJECT_DIR / "data/silkeborg_topology.dat"


# -----------------------------------------------------------------------------
# 1) Read pipe catalogue, define materials + fluids + soil
# -----------------------------------------------------------------------------
pipe_catalogue = read_pipe_catalogue(pipe_catalogue_file)

pipe_material_dist = Material(
    rho=975,
    c=1900,
    thermalCond=0.4,
)

brine = HeatCarrier(
    rho=965,
    c=4450,
    thermalCond=0.45,
    dynamicViscosity=5e-3,
)

soil = Soil(
    rho=2500,
    c=1000,
    thermalCond=2.36,
    thermalCondShallowHeating=1.25,
    thermalCondShallowCooling=1.25,
    Qgeo=0.01,
    surfaceTemp=9,
    surfaceTempAmp=7.9,
)

# -----------------------------------------------------------------------------
# 2) Distribution network
# -----------------------------------------------------------------------------
distribution_network_undimensioned = read_undimensioned_topology_tsv_to_network(
    topology_file,
    pipe_material=pipe_material_dist,
    roughness_height=1e-6,
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
r_b_m = borehole_diameter_m / 2.0
H_m = 150.0
D_m = 1.5
u_pipe_outer_diameter_m = 0.04
u_pipe_sdr = 11.0
grout = Material(rho=1500, c=2e3, thermalCond=1.75)
pipe_material_bhe = Material(rho=1000, c=2e3, thermalCond=0.4)
borehole = Annulus(outerDiameter=borehole_diameter_m, SDR=1000.0)

# Create the U-pipe for the borehole
upipe = PipeSegment(
    outerDiameter=u_pipe_outer_diameter_m,
    SDR=u_pipe_sdr,
    material=pipe_material_bhe,
    roughnessHeight=1e-6,
    ID=0,
    length=100,  # placeholder
)

BHEfield = VHEField(
    ID = 1,
    HE='1U',
    pipe = upipe,
    borehole = borehole,
    grout = grout,
    coordinates=coordinates,
    shankSpacing=0.015 + 2 * 0.02,
    H_m=120.0,
    D_m=1.0,
    r_b_m=0.075,
    tilt_rad=0.0,
    orientation_rad=0.0,
)

# -----------------------------------------------------------------------------
# 4) Heat pumps
# -----------------------------------------------------------------------------
hp_list = read_heat_pumps_tsv(path=heat_pump_file)

heat_pumps = HeatPumps(
    heatPumpList=hp_list,
    brine=brine,
    peak_heating_h=4.0,
    peak_fraction_heating_mode="incremental",
    peak_fraction_heating=1.0,
    peak_cooling_h=4.0,
    peak_fraction_cooling_mode="incremental",
    peak_fraction_cooling=1.0,
)

# -----------------------------------------------------------------------------
# 5) Simulation time vector
# -----------------------------------------------------------------------------
times_s = [
    4 * 3600,
    4 * 3600 + 86400 * 365.25 / 4,
    4 * 3600 + 86400 * 365.25 / 4 + 30 * 365.25 * 86400,
]

# -----------------------------------------------------------------------------
# 6) Hydraulic pipe network sizing
# -----------------------------------------------------------------------------
distribution_network = run_pipedimensioning(
    pipe_catalogue,
    brine,
    distribution_network_undimensioned,
    heat_pumps,
)

#print_pipe_thermal_table(distribution_network, 2)

# -----------------------------------------------------------------------------
# 7) Distribution pipe thermal simulation
# -----------------------------------------------------------------------------
dist_thermal = compute_distribution_pipe_thermal_capacity(
    network=distribution_network,
    brine=brine,
    soil=soil,
    heat_pumps=heat_pumps,
    times_heat_s=np.flip(times_s),
    times_cool_s=np.flip(times_s),
    T_brine_min_heat=-3.0,
    T_brine_max_cool=25.0,
)

print(dist_thermal)
### Status: dist_thermal objektet har properties der er per trace. Overvej om de skal appendes på netværksobjektet

# -----------------------------------------------------------------------------
# 8) Compute g-functions for VHE field
# -----------------------------------------------------------------------------

g_values = BHEfield.compute_pygfunctions(
    times_s=times_s,
    alpha_m2_s=soil.thermalCond / soil.rho / soil.c,
    method="equivalent",
    boundary_condition="UHTR",
)

# -----------------------------------------------------------------------------
# 9) Compute BHE thermal resistance from flow simulations (Rb)
# -----------------------------------------------------------------------------

rb = compute_rb_for_vhe_field(
    vhe_field=BHEfield,
    brine=brine,
    soil=soil,
    L_bhe_m=H_m,
    m_dot_kg_s=heat_pumps.aggregated_mdot_peak_heat_kg_s / n_boreholes,
    use_flow_length_correction=True,
)

print("Rb [K*m/W] =", rb.Rb_K_m_W)
print("Re, Pr     =", rb.Re, rb.Pr)

# -----------------------------------------------------------------------------
# 10) Compute fractions of thermal loads supplied by the boreholes
# -----------------------------------------------------------------------------
P_heating = (1 - dist_thermal["heating"].F_total)*heat_pumps.heating_ground_load_W
P_cooling = (1 - dist_thermal["cooling"].F_total)*heat_pumps.cooling_ground_load_W