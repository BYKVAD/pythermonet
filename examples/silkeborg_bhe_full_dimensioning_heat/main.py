from pathlib import Path

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
import numpy as np

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
    rho=2000,
    c=800,
    thermalCond=1.5,
    thermalCondShallowHeating=1.5,
    thermalCondShallowCooling=1.2,
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

# 4) Heat pumps
hp = read_heat_pumps_tsv(
    path=heat_pump_file,
    source_heat_carrier=brine,
)

# 5) Borehole field
n_boreholes = 6
spacing_m = 15.0

# Borehole diameter (fra legacy: r_b = 0.152/2)
borehole_diameter_m = 0.152

# U-pipe (fra legacy: r_p=0.02 => outer diameter 0.04 m, SDR=11)
u_pipe_outer_diameter_m = 0.04
u_pipe_sdr = 11.0

# Grout (fra legacy: l_g=1.75, rhoc_g=3e6) – din Material bruger rhoC
grout = Material(rho=1500, c=2e6, thermalCond=1.75)

# Pipe material (fra legacy: l_p=0.4) – rhoC skal du sætte til dit standardvalg
pipe_material = Material(rho=2000, c=1e3, thermalCond=0.4)  # justér rhoC efter din konvention

# Borehole/casing geometri: Annulus repræsenterer borehullet
# "tynd casing" kan modelleres ved stor SDR (ikke kritisk nu, da rb kommer fra outerDiameter)
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
# Hvis du vil have z med, kan du sætte 0.0 som tredje kolonne.
coordinates = [[0.0, i * spacing_m] for i in range(n_boreholes)]
# coordinates = [[0.0, i * spacing_m, 0.0] for i in range(n_boreholes)]  # hvis du vil have z med

vhe_field = VHEField(
    ID=1,
    HE="1U",
    pipe=pipe,
    borehole=borehole,
    grout=grout,
    coordinates=coordinates,
    shankSpacing=0.015 + 2 * 0.02,  # legacy: s = 2*r_p + D_pipes (D_pipes=0.015, r_p=0.02)
)

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