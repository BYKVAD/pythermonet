import json
from pathlib import Path

from pythermonet.components.heat_pumps import HeatPumps
from pythermonet.core.material import Material
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.input.read_pipe_catalogue import read_pipe_catalogue
from pythermonet.input.read_topology import read_undimensioned_topology_tsv_to_network
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv
from pythermonet.components.vhe_field import VHEField
from pythermonet.core.annulus import Annulus
from pythermonet.dimensioning.hydraulic_dimensioning import run_pipedimensioning
from pythermonet.components.vhe_field import VHEField
from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegment

from pythermonet.components.distribution_network import DistributionNetwork
from pythermonet.simulation.run_distribution_pipe_thermal import run_distribution_pipe_thermal
import numpy as np

from pythermonet.gfunctions.models import GFunctionRequest
from pythermonet.gfunctions.service import GFunctionsService

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[2]

pipe_catalogue_file = REPO_ROOT / "PythermonetII/src/pythermonet/resources/pipe_catalogue.csv"
heat_pump_file = PROJECT_DIR / "data/silkeborg_heat_pump_heat.dat"
topology_file = PROJECT_DIR / "data/silkeborg_topology.dat"

# 1) Pipe catalogue
pipe_catalogue = read_pipe_catalogue(pipe_catalogue_file)

# 2) Materials
pipe_material = Material(
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
    rho=2650,
    c=1000,
    thermalCond=2.36,
    thermalCondShallowHeating=1.25,
    thermalCondShallowCooling=1.25,
    Qgeo=0.01,
    surfaceTemp=9,
    surfaceTempAmp=7.9,
    )

# 3) Pipe distribution grid infrastructure
net = read_undimensioned_topology_tsv_to_network(
    topology_file,
    pipe_material=pipe_material,
    roughness_height=1e-6,
    burial_depth=1.2,
    pipe_distance=0.3,
    n_parallel_pipes=2,
    )

hp = HeatPumps(
    heatPumpList=read_heat_pumps_tsv(
    path=heat_pump_file,
    source_heat_carrier=brine,),
    brine=brine,
    peak_heating_h=4.0,
    peak_fraction_heating_mode="incremental",
    peak_fraction_heating=1,
    peak_cooling_h=4.0,
    peak_fraction_cooling_mode="incremental",
    peak_fraction_cooling=1 ,
)


# 5) Borehole field
n_boreholes = 6
spacing_m = 15.0
borehole_diameter_m = 0.152
r_b_m = borehole_diameter_m / 2.0
H_m = 150.0   # borehole length, eksempel
D_m = 1.5     # top depth under ground, eksempel

times_s = [4*3600, 86400*365.25/4, 30*365.25*86400 ]
alpha_m2_s = soil.thermalCond / (soil.rho * soil.c)  # forudsætter c = J/kg/K

# U-pipe (fra legacy: r_p=0.02 => outer diameter 0.04 m, SDR=11)
u_pipe_outer_diameter_m = 0.04
u_pipe_sdr = 11.0

# Grout (fra legacy: l_g=1.75, rhoc_g=3e6) – din Material bruger rhoC
grout = Material(rho=1500, c=2e3, thermalCond=1.75)

# Pipe material (fra legacy: l_p=0.4) – rhoC skal du sætte til dit standardvalg
pipe_material = Material(rho=1000, c=2e3, thermalCond=0.4)  # justér rhoC efter din konvention

# Borehole/casing geometri: Annulus repræsenterer borehullet
borehole = Annulus(outerDiameter=borehole_diameter_m, SDR=1000.0)

# PipeSegment i VHEField: bruges her kun som rørgeometri + materialer (ikke "trace segment")
pipe = PipeSegment(
    outerDiameter=u_pipe_outer_diameter_m,
    SDR=u_pipe_sdr,
    material=pipe_material,
    roughnessHeight=1e-6,
    ID=0,
    length=1.0,  # placeholder; længde bruges ikke i VHEField-semantik
)

# Koordinater: 6 borehuller på linje langs y-aksen (x=0)
coordinates = [[0.0, i * spacing_m, 0.0] for i in range(n_boreholes)]

# 4) Run pipe sizing
network = read_undimensioned_topology_tsv_to_network(
    topology_file,
    pipe_material=pipe_material,
    roughness_height=1e-6,
    burial_depth=1.2,
    pipe_distance=0.3,
    n_parallel_pipes=2,
)

distribution_network = run_pipedimensioning(pipe_catalogue, brine, network, hp)
print("Dimensioned distribution network:", distribution_network.infrastructure.traceSegments)

vhe_field = VHEField(
    ID=1,
    HE="1U",
    pipe=pipe,
    borehole=borehole,
    grout=grout,
    coordinates=coordinates,
    shankSpacing=0.015 + 2 * 0.02,  # legacy: s = 2*r_p + D_pipes (D_pipes=0.015, r_p=0.02)
)

# Define boreholes for g-function computation using VHEField adapter

boreholes = vhe_field.to_pygfunction_boreholes(
    H_m=H_m,
    D_m=D_m,
    r_b_m=r_b_m,
    use_z_as_depth=False,
)

req = GFunctionRequest(
    times_s=times_s,
    boreholes=boreholes,
    alpha_m2_s=alpha_m2_s,
    method="equivalent", #['detailed', 'similarities', 'equivalent']
    boundary_condition='UHTR',
    options={},
)

# 5) Compute g-functions

gset = GFunctionsService(cache=None).compute(req)

print("Computed g-functions:", gset.g_values[:5])

from pythermonet.physics.bhe_resistance import compute_rb_for_vhe_field

Q_BHE = hp.aggregated_q_peak_heat_m3_s  # eksempelvis, eller tag det fra distribution_network efter dimensionering
print("Peak volumetric flow rate [m3/s]:", Q_BHE)

rb = compute_rb_for_vhe_field(
    vhe_field=vhe_field,
    brine=brine,
    soil=soil,
    L_bhe_m=150.0,
    m_dot_kg_s=hp.aggregated_mdot_peak_heat_kg_s/n_boreholes,
    use_flow_length_correction=True,
)

print("Rb [K*m/W] =", rb.Rb_K_m_W)
print("Re, Pr     =", rb.Re, rb.Pr)

print("N HP:", hp.n_heat_pumps)
print("Diversity:", hp.diversity_factor)

# P_peak_heat = hp.aggre
# q_peak_heat = hp.aggregated_q_peak_heat_m3_s
# print("Aggregated peak heating ground load [W]:", P_peak_heat)

from pythermonet.simulation.run_distribution_pipe_thermal import run_distribution_pipe_thermal

dist_therm = run_distribution_pipe_thermal(
    network=distribution_network,
    brine=brine,
    soil=soil,
    heat_pumps=hp,
    Ti_heat_C=-3.0,
    To_heat_C=-6.0,
    peak_heating_h=4.0,
)

print("T_dimv (heating) [°C]:", dist_therm.heating.T_dimv_C)
print("F_total (heating):", dist_therm.heating.F_total)